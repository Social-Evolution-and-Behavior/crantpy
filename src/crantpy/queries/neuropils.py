# -*- coding: utf-8 -*-
"""
This module contains functions to query neuropil information from the CRANTb dataset.

"""

import datetime
import logging
from typing import Dict, List, Optional, Union, TYPE_CHECKING
import pandas as pd
import numpy as np
import trimesh as tm
from crantpy.utils.config import CRANT_VALID_DATASETS
from crantpy.utils.decorators import inject_dataset, parse_neuroncriteria
from crantpy.utils.helpers import parse_root_ids
from crantpy.queries.connections import get_synapses
from crantpy.viz.mesh import (
    get_supported_neuropil_mesh_labels,
    load_neuropil_mesh,
    resolve_neuropil_mesh_label_ids,
)

from crantpy.utils.config import SCALE_X, SCALE_Y, SCALE_Z, SYN_V3_RES_X, SYN_V3_RES_Y, SYN_V3_RES_Z

if TYPE_CHECKING:
    from crantpy.queries.neurons import NeuronCriteria

logger = logging.getLogger(__name__)

# CAVE silently truncates query results at this row count
CAVE_ROW_LIMIT = 200_000
_MAX_SUBDIVISION_DEPTH = 8
_MESH_CONTAINS_BATCH_SIZE = 50_000

# synapses_v3 native coordinate resolution as a numpy array (nm per unit).
# Multiply v3 position columns by this to obtain nanometer coordinates.
_SYN_V3_RES = np.array([SYN_V3_RES_X, SYN_V3_RES_Y, SYN_V3_RES_Z], dtype=float)


def _filter_by_neuron_count(
    syn: pd.DataFrame, min_count: int, label: str = ""
) -> pd.DataFrame:
    """Drop synapses unless both pre- and post-neuron have >= *min_count* synapses."""
    pre_counts = syn["pre_pt_root_id"].value_counts()
    post_counts = syn["post_pt_root_id"].value_counts()
    all_neuron_counts = pre_counts.add(post_counts, fill_value=0)
    valid_neurons = all_neuron_counts[all_neuron_counts >= min_count].index
    syn = syn[
        (syn["pre_pt_root_id"].isin(valid_neurons))
        & (syn["post_pt_root_id"].isin(valid_neurons))
    ]
    logger.info(
        "After neuron filtering%s: %d synapses from %d neurons with >= %d synapses",
        f" in {label}" if label else "",
        len(syn),
        len(valid_neurons),
        min_count,
    )
    return syn


def _filter_by_pair_count(
    syn: pd.DataFrame, min_count: int, label: str = ""
) -> pd.DataFrame:
    """Drop synapses unless their pre/post pair has >= *min_count* synapses."""
    pair_counts = syn.groupby(["pre_pt_root_id", "post_pt_root_id"]).size()
    valid_pairs = pair_counts[pair_counts >= min_count].index
    syn = syn.set_index(["pre_pt_root_id", "post_pt_root_id"])
    syn = syn.loc[syn.index.isin(valid_pairs)]
    syn = syn.reset_index()
    logger.info(
        "After pair filtering%s: %d synapses with >= %d synapses per pair",
        f" in {label}" if label else "",
        len(syn),
        min_count,
    )
    return syn


def _normalize_synapse_root_ids(
    syn: pd.DataFrame, dataset: Optional[str] = None
) -> pd.DataFrame:
    """Map synapse root-ID columns to their latest roots.

    Mesh queries return the root IDs stored in the queried materialization.
    To align results with current proofreading/annotation tables, normalize
    all returned root IDs to their latest roots before downstream filtering.
    """
    id_columns = [c for c in ("pre_pt_root_id", "post_pt_root_id") if c in syn.columns]
    if syn.empty or not id_columns:
        return syn

    unique_ids = pd.unique(
        pd.concat([syn[col] for col in id_columns], ignore_index=True).dropna()
    )
    unique_ids = np.asarray(unique_ids)
    unique_ids = unique_ids[unique_ids != 0]
    if len(unique_ids) == 0:
        return syn

    from crantpy.utils.cave.segmentation import update_ids as _update_ids

    logger.info(
        "Normalizing %d unique synapse root ID(s) to latest roots", len(unique_ids)
    )
    updates = _update_ids(
        unique_ids.tolist(),
        dataset=dataset,
        progress=False,
        clear_cache=True,
    )

    if updates.empty:
        logger.warning(
            "Root-ID normalization returned no mapping data; leaving synapse IDs unchanged"
        )
        return syn

    id_map = dict(zip(updates["old_id"], updates["new_id"]))
    unresolved = (
        int((updates["confidence"] == 0).sum())
        if "confidence" in updates.columns
        else 0
    )
    changed = int(updates["changed"].sum()) if "changed" in updates.columns else 0
    if unresolved:
        logger.warning(
            "Could not normalize %d synapse root ID(s); leaving those IDs unchanged",
            unresolved,
        )
    if changed:
        logger.info("Updated %d synapse root ID(s) to newer roots", changed)

    syn = syn.copy()
    for col in id_columns:
        syn[col] = syn[col].map(id_map).fillna(syn[col]).astype(np.int64)
    return syn


def _validate_neuropil_names(names: List[str]) -> None:
    """Raise ``ValueError`` if any name is not a supported neuropil label."""
    for name in names:
        try:
            resolve_neuropil_mesh_label_ids(name)
        except ValueError as exc:
            raise ValueError(
                f"Invalid neuropil name: {name!r}. "
                f"Available: {get_supported_neuropil_mesh_labels()}"
            ) from exc


def _batched_mesh_contains(
    mesh: tm.Trimesh, points: np.ndarray, batch_size: int = _MESH_CONTAINS_BATCH_SIZE
) -> np.ndarray:
    """Run ``mesh.contains`` in batches to avoid OOM on large point sets."""
    n = len(points)
    if n <= batch_size:
        return mesh.contains(points)
    result = np.empty(n, dtype=bool)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        logger.debug(f"mesh.contains batch {start}-{end} of {n}")
        result[start:end] = mesh.contains(points[start:end])
    return result


class _CircuitBreaker:
    """Lightweight circuit breaker for CAVE API calls.

    States:
      - CLOSED  (normal):  requests flow through
      - OPEN    (tripped): requests are blocked; raises immediately
      - HALF_OPEN (probe): one probe request is allowed through

    Transitions:
      CLOSED  -> OPEN       after ``failure_threshold`` consecutive failures
      OPEN    -> HALF_OPEN  after ``recovery_timeout`` seconds
      HALF_OPEN -> CLOSED   if the probe request succeeds
      HALF_OPEN -> OPEN     if the probe request fails
    """

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time = None

    @property
    def state(self):
        if self._state == self.OPEN and self._last_failure_time is not None:
            import time as _time

            elapsed = _time.time() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                self._state = self.HALF_OPEN
                logger.info(
                    f"Circuit breaker OPEN -> HALF_OPEN after "
                    f"{elapsed:.0f}s cooldown, allowing probe request"
                )
        return self._state

    def record_success(self):
        if self._state in (self.HALF_OPEN, self.OPEN):
            logger.info("Circuit breaker -> CLOSED (probe succeeded)")
        self._failure_count = 0
        self._state = self.CLOSED

    def record_failure(self):
        import time as _time

        self._failure_count += 1
        self._last_failure_time = _time.time()
        if self._state == self.HALF_OPEN:
            self._state = self.OPEN
            logger.warning(
                "Circuit breaker HALF_OPEN -> OPEN (probe failed), "
                f"blocking requests for {self.recovery_timeout}s"
            )
        elif self._failure_count >= self.failure_threshold:
            self._state = self.OPEN
            logger.warning(
                f"Circuit breaker -> OPEN after {self._failure_count} consecutive "
                f"failures, blocking requests for {self.recovery_timeout}s"
            )

    def __repr__(self):
        return (
            f"_CircuitBreaker(state={self.state}, failures={self._failure_count}/"
            f"{self.failure_threshold})"
        )


# Shared circuit breaker for all CAVE synapse queries
_cave_breaker = _CircuitBreaker(failure_threshold=5, recovery_timeout=120)


def _query_with_breaker(func, *args, operation=None, **kwargs):
    """Execute a CAVE call through the circuit breaker + retry logic.

    If the breaker is OPEN, waits for the recovery timeout before attempting
    a probe. Uses aggressive retries (15 attempts, 10s linear backoff) within
    each circuit-breaker cycle.
    """
    import time as _time
    import requests
    from crantpy.utils.helpers import retry

    operation_name = operation or getattr(func, "__name__", repr(func))
    state = _cave_breaker.state
    logger.debug(
        "Executing CAVE query '%s' through breaker (state=%s, args=%d, kwargs=%s)",
        operation_name,
        state,
        len(args),
        sorted(kwargs),
    )
    if state == _CircuitBreaker.OPEN:
        wait = _cave_breaker.recovery_timeout
        if _cave_breaker._last_failure_time is not None:
            elapsed = _time.time() - _cave_breaker._last_failure_time
            wait = max(0, _cave_breaker.recovery_timeout - elapsed)
        logger.warning(
            "Circuit breaker is OPEN - waiting %.0fs before probe request for '%s'",
            wait,
            operation_name,
        )
        _time.sleep(wait)
        # After sleeping, state should transition to HALF_OPEN
        _ = _cave_breaker.state

    attempt_count = 0
    saw_request_failure = False
    logged_retry_cycle = False

    def _logged_func(*inner_args, **inner_kwargs):
        nonlocal attempt_count, saw_request_failure, logged_retry_cycle

        attempt_count += 1
        logger.debug(
            "CAVE query '%s' attempt %d/%d",
            operation_name,
            attempt_count,
            15,
        )
        try:
            return func(*inner_args, **inner_kwargs)
        except requests.RequestException as exc:
            saw_request_failure = True
            if not logged_retry_cycle:
                logger.info(
                    "CAVE query '%s' failed on attempt %d; retrying with up to %d attempts and %.0fs linear backoff",
                    operation_name,
                    attempt_count,
                    15,
                    10.0,
                )
                logged_retry_cycle = True
            logger.debug(
                "CAVE query '%s' attempt %d/%d failed with %s: %s",
                operation_name,
                attempt_count,
                15,
                type(exc).__name__,
                exc,
            )
            raise

    try:
        result = retry(_logged_func, retries=15, cooldown=10)(*args, **kwargs)
        if saw_request_failure:
            logger.info(
                "CAVE query '%s' succeeded after %d attempts (%d retries)",
                operation_name,
                attempt_count,
                attempt_count - 1,
            )
        _cave_breaker.record_success()
        return result
    except requests.RequestException as exc:
        logger.error(
            "CAVE query '%s' failed after %d attempts with %s: %s",
            operation_name,
            attempt_count,
            type(exc).__name__,
            exc,
        )
        _cave_breaker.record_failure()
        raise


def _query_synapses_in_bbox(
    client, bbox, materialization, materialization_version=None, depth=0
):
    """
    Query synapses within a bounding box, auto-subdividing if CAVE truncates.

    CAVE silently truncates results at ~200,000 rows. This function detects
    truncation (result count >= CAVE_ROW_LIMIT) and recursively bisects the
    bounding box along its longest axis, querying each half separately.

    Uses a circuit breaker to avoid hammering the CAVE server when it is
    persistently unavailable. After 5 consecutive failures the breaker trips
    OPEN, blocking requests for 120 s before sending a single probe.

    Parameters
    ----------
    client : CAVEclient
        An initialized CAVE client.
    bbox : list
        Bounding box as [[min_x, min_y, min_z], [max_x, max_y, max_z]].
    materialization : str
        Either 'live' or 'latest'.
    materialization_version : int, optional
        Resolved materialization version (used for 'latest' mode).
        If None and materialization is 'latest', it will be resolved.
    depth : int
        Current recursion depth (safety guard).

    Returns
    -------
    tuple of (pd.DataFrame, int)
        The synapse DataFrame and the resolved materialization_version.
    """
    # Resolve materialization version once at the top level
    if materialization == "latest" and materialization_version is None:
        materialization_version = _query_with_breaker(
            client.materialize.most_recent_version,
            operation="most_recent_version",
        )

    filter_spatial_dict = {"ctr_pt_position": bbox}

    if materialization == "live":
        syn = _query_with_breaker(
            client.materialize.live_query,
            operation="live_query",
            table="synapses_v3",
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            filter_spatial_dict=filter_spatial_dict,
        )
    else:
        syn = _query_with_breaker(
            client.materialize.query_table,
            operation="query_table",
            table="synapses_v3",
            materialization_version=materialization_version,
            filter_spatial_dict=filter_spatial_dict,
        )

    # Check for truncation
    if len(syn) >= CAVE_ROW_LIMIT and depth < _MAX_SUBDIVISION_DEPTH:
        min_coords = np.array(bbox[0], dtype=float)
        max_coords = np.array(bbox[1], dtype=float)
        extents = max_coords - min_coords
        axis = int(np.argmax(extents))
        midpoint = (min_coords[axis] + max_coords[axis]) / 2.0

        logger.info(
            f"CAVE returned {len(syn)} rows (>= {CAVE_ROW_LIMIT}), "
            f"subdividing along axis {axis} at depth {depth}"
        )

        # First half: min..mid along the split axis
        max_a = max_coords.copy()
        max_a[axis] = midpoint
        bbox_a = [min_coords.tolist(), max_a.tolist()]

        # Second half: mid..max along the split axis
        min_b = min_coords.copy()
        min_b[axis] = midpoint
        bbox_b = [min_b.tolist(), max_coords.tolist()]

        syn_a, materialization_version = _query_synapses_in_bbox(
            client, bbox_a, materialization, materialization_version, depth + 1
        )
        syn_b, materialization_version = _query_synapses_in_bbox(
            client, bbox_b, materialization, materialization_version, depth + 1
        )

        syn = pd.concat([syn_a, syn_b], ignore_index=True)
        # Deduplicate by id column (synapses on the boundary may appear in both)
        if "id" in syn.columns:
            syn = syn.drop_duplicates(subset="id")
        logger.info(f"After merging subdivisions at depth {depth}: {len(syn)} synapses")

    return syn, materialization_version


@parse_neuroncriteria()
@inject_dataset(allowed=CRANT_VALID_DATASETS)
def count_synapses_in_mesh(
    neuron_ids: Union[int, str, List[Union[int, str]], "NeuronCriteria"],
    neuropil_mesh_names: Union[str, List[str]],
    min_synapses_per_neuron: int = 1,
    min_synapses_per_pair: int = 1,
    materialization: Optional[str] = "latest",
    update_ids: bool = True,
    dataset: Optional[str] = None,
    loc: str = "pre",
) -> pd.DataFrame:
    """
    Count the number of presynaptic outputs from specified neurons within neuropil meshes.

    This function queries all synapses where the specified neurons are presynaptic,
    then counts how many of these synapses fall within each specified neuropil mesh region.

    Parameters
    ----------
    neuron_ids : int, str, list of int/str, or NeuronCriteria
        Neuron root ID(s) to query. These neurons will be treated as presynaptic neurons.
        Can be a single ID, list of IDs, or NeuronCriteria object.
    neuropil_mesh_names : str or list of str
        Name(s) of neuropil mesh(es) to load and check against. Must be valid neuropil
        labels from the NEUROPIL_MESH_DICT configuration.
    min_synapses_per_neuron : int, default 1
        Minimum number of synapses required from a presynaptic neuron within the mesh
        to be retained. Presynaptic neurons with fewer synapses are filtered out entirely.
    min_synapses_per_pair : int, default 1
        Minimum number of synapses required between a neuron pair (pre-post) to be retained.
        Synaptic connections (pre-post pairs) with fewer synapses are filtered out.
    materialization : str, default 'latest'
        Materialization version to use. 'latest' (default) or 'live' for live table.
        This is passed to get_synapses().
    update_ids : bool, default True
        Whether to automatically update outdated root IDs to their latest versions
        before querying. This is passed to get_synapses().
    dataset : str, optional
        Dataset to use for the query. If None, uses the default dataset.
    loc : {"pre", "ctr"}, default "pre"
        Which synapse position column to use for point-in-mesh testing.
        ``"pre"`` uses the presynaptic terminal position (``pre_pt_position``).
        ``"ctr"`` uses the synapse center point (``ctr_pt_position``).

    Returns
    -------
    pd.DataFrame
        A DataFrame with neurons as rows (indexed by neuron ID) and neuropil meshes
        as columns. Each cell contains the count of synapses from that neuron that
        fall within that neuropil mesh region.

    Raises
    ------
    ValueError
        If neuropil_mesh_names contains invalid neuropil labels.

    Examples
    --------
    >>> import crantpy as cp
    >>> # Count synapses in a single neuropil for one neuron
    >>> counts = cp.count_synapses_in_mesh(
    ...     neuron_ids=576460752641833774,
    ...     neuropil_mesh_names='LH'
    ... )
    >>>
    >>> # Count synapses in multiple neuropils for multiple neurons
    >>> counts = cp.count_synapses_in_mesh(
    ...     neuron_ids=[576460752641833774, 576460752777916050],
    ...     neuropil_mesh_names='antennal_lobe_left', 'mushroom_body_pedunculus_and_lobes_right']
    ... )
    >>>
    >>> # Use with NeuronCriteria
    >>> neurons = cp.NeuronCriteria(cell_type='PN')
    >>> counts = cp.count_synapses_in_mesh(
    ...     neuron_ids=neurons,
    ...     neuropil_mesh_names=['antennal_lobe_left', 'mushroom_body_pedunculus_and_lobes_right']
    ... )

    Notes
    -----
    - This function only considers synapses where the query neurons are **presynaptic**.
    - By default, synapse positions are extracted from the ``pre_pt_position`` column.
      Set ``loc="ctr"`` to use the synapse center point instead.
    - Coordinates are automatically converted to nanometers for comparison with meshes.
    - Point-in-mesh testing uses ray casting, so meshes should be closed/watertight.
    - The function may take some time for large numbers of synapses or complex meshes.

    See Also
    --------
    get_synapses : Retrieve raw synapse data
    load_neuropil_mesh : Load neuropil mesh by name
    """
    # Parse neuron IDs - keep as List[Union[int, str]] for compatibility with get_synapses
    query_ids_parsed = [int(x) for x in parse_root_ids(neuron_ids)]

    # Normalize neuropil mesh names to list
    if isinstance(neuropil_mesh_names, str):
        neuropil_mesh_names = [neuropil_mesh_names]

    # Verify neuropil mesh names
    _validate_neuropil_names(neuropil_mesh_names)

    if loc not in {"pre", "ctr"}:
        raise ValueError("loc must be either 'pre' or 'ctr'")

    # Get synapses where query neurons are presynaptic
    logger.info(f"Fetching synapses for {len(query_ids_parsed)} neuron(s)...")
    synapses = get_synapses(
        pre_ids=query_ids_parsed,
        post_ids=None,  # Don't filter by postsynaptic neurons
        min_size=None,
        materialization=materialization,
        return_pixels=False,
        clean=True,
        update_ids=update_ids,
        dataset=dataset,
    )

    # Initialize result DataFrame
    result_df = pd.DataFrame(
        0,
        index=pd.Index(query_ids_parsed, name="neuron_id"),
        columns=neuropil_mesh_names,
        dtype=int,
    )

    # If no synapses found, return empty result
    if synapses.empty:
        logger.warning("No synapses found for the specified neurons")
        return result_df

    # Apply neuron-level filtering if specified
    if min_synapses_per_neuron > 1:
        synapses = _filter_by_neuron_count(synapses, min_synapses_per_neuron)

    # Apply pair-level filtering if specified
    if min_synapses_per_pair > 1:
        synapses = _filter_by_pair_count(synapses, min_synapses_per_pair)

    # If no synapses remaining after filtering, return empty result
    if synapses.empty:
        logger.warning("No synapses remaining after filtering")
        return result_df

    # Load neuropil meshes
    logger.info(f"Loading {len(neuropil_mesh_names)} neuropil mesh(es)...")
    meshes = {}
    for neuropil_name in neuropil_mesh_names:
        try:
            meshes[neuropil_name] = load_neuropil_mesh(neuropil_name)
            logger.debug(f"Loaded mesh for {neuropil_name}")
        except Exception as e:
            logger.error(f"Failed to load mesh for {neuropil_name}: {e}")
            raise

    # Extract synapse positions. get_synapses(return_pixels=False) already returns nm.
    position_column = "ctr_pt_position" if loc == "ctr" else "pre_pt_position"
    logger.info(f"Processing {len(synapses)} synapses using {position_column}...")

    # Coordinates are already in nm (get_synapses converts v3 units → nm internally)
    synapse_coords = np.vstack(synapses[position_column].values)

    # Add coordinates and neuron IDs to a working dataframe (in nm, matching mesh space)
    synapse_df = pd.DataFrame(
        {
            "neuron_id": synapses["pre_pt_root_id"].values,
            "x": synapse_coords[:, 0],
            "y": synapse_coords[:, 1],
            "z": synapse_coords[:, 2],
        }
    )

    # For each neuropil mesh, check which synapses are inside
    for neuropil_name, mesh in meshes.items():
        logger.info(f"Checking synapses against {neuropil_name} mesh...")

        # Check each synapse position against the mesh
        points = synapse_df[["x", "y", "z"]].values
        inside_mask = mesh.contains(points)

        # Count synapses per neuron that are inside this mesh
        synapse_df["inside"] = inside_mask
        counts_per_neuron = synapse_df[synapse_df["inside"]].groupby("neuron_id").size()

        # Update result DataFrame for this neuropil
        for neuron_id in query_ids_parsed:
            if neuron_id in counts_per_neuron.index:
                count_value = counts_per_neuron[neuron_id]
                result_df.loc[result_df.index == neuron_id, neuropil_name] = int(
                    count_value
                )

        logger.debug(f"Found {inside_mask.sum()} synapses in {neuropil_name}")

    logger.info("Synapse counting complete")
    return result_df.reset_index()


@inject_dataset(allowed=CRANT_VALID_DATASETS)
def get_synapses_in_mesh(
    mesh: tm.Trimesh,
    min_synapses_per_neuron: int = 1,
    min_synapses_per_pair: int = 1,
    min_size: Optional[int] = None,
    loc: str = "ctr",
    materialization: Optional[str] = "latest",
    return_pixels: bool = True,
    clean: bool = True,
    mesh_coordinates: str = "nm",
    dataset: Optional[str] = None,
) -> pd.DataFrame:
    """
    Query all synapses within a given mesh volume.

    This function queries the synapse table and filters for synapses whose center point
    coordinates fall within the provided mesh volume. This is useful for analyzing
    synaptic connectivity within a specific anatomical region.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        A trimesh object representing the volume to query. Synapses whose center point
        coordinates fall within this mesh will be returned. The mesh should be in the
        same coordinate space as the synapse data (typically nanometers).
    min_synapses_per_neuron : int, default 1
        Minimum number of synapses required from a presynaptic neuron within the mesh
        to be retained. Presynaptic neurons with fewer synapses are filtered out entirely.
    min_synapses_per_pair : int, default 1
        Minimum number of synapses required between a neuron pair (pre-post) to be retained.
        Synaptic connections (pre-post pairs) with fewer synapses are filtered out.
    min_size : int, optional
        Minimum size for filtering synapses. If specified, only synapses with size
        greater than or equal to this value will be included.
    loc : {"ctr", "all"}, default "ctr"
        Which synapse location(s) must fall inside the mesh.
        ``"ctr"`` keeps synapses whose center point is inside the mesh.
        ``"all"`` requires center, presynaptic, and postsynaptic points all to be
        inside the mesh.
    materialization : str, default 'latest'
        Materialization version to use. 'latest' (default) or 'live' for live table.
    return_pixels : bool, default True
        Whether to convert coordinate columns from nanometers to pixels.
        If True (default), coordinates in ctr_pt_position, pre_pt_position, and
        post_pt_position are converted using dataset scale factors.
        If False, coordinates remain in nanometer units.
    clean : bool, default True
        Whether to perform cleanup of the synapse data:
        - Remove autapses (self-connections)
        - Remove connections involving neuron ID 0 (background)
    mesh_coordinates : str, default "nm"
        Coordinate system of the mesh. Either "nm" (nanometers) or "voxels" (pixels).
        If "nm", both the query and mesh.contains() will use nanometer coordinates.
        If "voxels", the mesh bounds will be converted to nanometers for the database
        query, and synapse coordinates will be converted to voxels for mesh.contains().
    dataset : str, optional
        Dataset to use for the query. If None, uses the default dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame of synapses within the mesh, with the same columns as returned by
        get_synapses(), including an additional boolean column 'in_mesh' that is True
        for all returned synapses.

    Examples
    --------
    >>> import crantpy as cp
    >>> import trimesh as tm
    >>> # Load a neuropil mesh
    >>> mesh = cp.load_neuropil_mesh('antennal_lobe_left')
    >>>
    >>> # Get all synapses within the mesh
    >>> synapses = cp.get_synapses_in_mesh(mesh)
    >>>
    >>> # Get synapses with minimum pair threshold (only pairs with 3+ synapses)
    >>> synapses = cp.get_synapses_in_mesh(mesh, min_synapses_per_pair=3)
    >>>
    >>> # Get synapses from neurons with at least 10 total synapses in the mesh
    >>> synapses = cp.get_synapses_in_mesh(mesh, min_synapses_per_neuron=10)
    >>>
    >>> # Get synapses with minimum size threshold
    >>> synapses = cp.get_synapses_in_mesh(mesh, min_size=50)
    >>>
    >>> # Require center, pre, and post points all to be inside the mesh
    >>> synapses = cp.get_synapses_in_mesh(mesh, loc='all')
    >>>
    >>> # Get live data without cleaning
    >>> synapses = cp.get_synapses_in_mesh(mesh, materialization='live', clean=False)
    >>>
    >>> # Use a mesh in voxel coordinates instead of nanometers
    >>> synapses = cp.get_synapses_in_mesh(mesh_voxels, mesh_coordinates='voxels')

    Notes
    -----
    - This function first filters synapses using the mesh's bounding box, then performs
      precise point-in-mesh testing only on synapses within the bounding box. This is
      much more efficient than querying all synapses in the dataset.
    - The loc parameter controls whether mesh membership is defined by
      'ctr_pt_position' only or by all of 'ctr_pt_position', 'pre_pt_position',
      and 'post_pt_position'.
    - The mesh_coordinates parameter controls coordinate conversion: meshes from
      load_neuropil_mesh() are typically in nanometers, while custom meshes may be
      in voxel space.
    - The min_synapses_per_neuron parameter filters out entire presynaptic neurons if
      they have fewer than the specified number of synapses within the mesh.
    - The min_synapses_per_pair parameter filters connection pairs (pre-post neuron pairs),
      keeping only pairs with the minimum number of synapses specified.
    - Both filters are applied independently; a synapse is retained only if its presynaptic
      neuron meets the min_synapses_per_neuron threshold AND its pre-post pair meets
      the min_synapses_per_pair threshold.
    - The mesh.contains() method uses ray casting, so meshes should be closed/watertight
      for accurate results.
    - The return_pixels parameter only affects the output coordinates, not the query.
    - Returned root IDs are normalized to latest roots so they remain aligned with
      current proofreading/annotation tables.

    See Also
    --------
    get_synapses : Query synapses by pre/post neuron IDs
    count_synapses_in_mesh : Count synapses from specific neurons within meshes
    load_neuropil_mesh : Load neuropil mesh by name
    """
    from crantpy.utils.cave.load import get_cave_client

    # Always bypass the cached client here so materialization-backed queries
    # do not reuse stale client state across calls.
    client = get_cave_client(dataset=dataset, clear_cache=True)

    # Validate mesh_coordinates parameter
    if mesh_coordinates not in ["nm", "voxels"]:
        raise ValueError("mesh_coordinates must be either 'nm' or 'voxels'")
    if loc not in {"ctr", "all"}:
        raise ValueError("loc must be either 'ctr' or 'all'")

    # Get mesh bounding box for efficient spatial filtering
    min_coords, max_coords = mesh.bounds
    logger.info(
        f"Mesh bounding box ({mesh_coordinates}): min={min_coords}, max={max_coords}"
    )

    _SCALE = np.array([SCALE_X, SCALE_Y, SCALE_Z], dtype=float)

    # Convert mesh bounds to nanometers if needed
    if mesh_coordinates == "voxels":
        min_coords_nm = np.asarray(min_coords, dtype=float) * _SCALE
        max_coords_nm = np.asarray(max_coords, dtype=float) * _SCALE
    else:
        min_coords_nm = np.asarray(min_coords, dtype=float)
        max_coords_nm = np.asarray(max_coords, dtype=float)

    # Convert nm bounding box to synapses_v3 native units for the CAVE spatial filter
    _v3_res = _SYN_V3_RES
    bbox_v3 = [
        (min_coords_nm / _v3_res).tolist(),
        (max_coords_nm / _v3_res).tolist(),
    ]
    logger.info(f"CAVE query bbox (v3 units): min={bbox_v3[0]}, max={bbox_v3[1]}")


    # Validate materialization parameter
    if materialization not in ("live", "latest"):
        raise ValueError("materialization must be either 'live' or 'latest'")

    # Query synapses with auto-subdivision for large bounding boxes
    logger.info("Querying synapses within mesh bounding box...")
    syn, _ = _query_synapses_in_bbox(client, bbox_v3, materialization)


    if syn.empty:
        logger.warning("No synapses found in bounding box")
        return syn

    logger.info(f"Retrieved {len(syn)} synapses within bounding box")

    point_columns = (
        ["ctr_pt_position"]
        if loc == "ctr"
        else [
            "ctr_pt_position",
            "pre_pt_position",
            "post_pt_position",
        ]
    )
    for col in point_columns:
        if col not in syn.columns:
            raise ValueError(f"Expected column '{col}' not found in synapse data")

        # Convert synapse coords from v3 native units to the mesh coordinate space
        synapse_coords = np.vstack(syn[col].values).astype(float)

        if mesh_coordinates == "voxels":
            # v3 units → nm → voxels
            synapse_coords_for_mesh = synapse_coords * _SYN_V3_RES / _SCALE
        else:
            # v3 units → nm
            synapse_coords_for_mesh = synapse_coords * _SYN_V3_RES

        logger.info(
            "Checking %d synapses against mesh using %s...",
            len(synapse_coords_for_mesh),
            col,
        )
        inside_mask = _batched_mesh_contains(mesh, synapse_coords_for_mesh)
        logger.info("Found %d synapses inside mesh (%s)", inside_mask.sum(), col)

        syn = syn[inside_mask].copy()

        if syn.empty:
            break

    if syn.empty:
        logger.warning("No synapses found within mesh")
        return syn

    # Apply size filter if specified
    if min_size is not None and "size" in syn.columns:
        syn = syn[syn["size"] >= min_size]
        logger.info(f"After size filtering: {len(syn)} synapses")

    syn = _normalize_synapse_root_ids(syn, dataset=dataset)

    # Convert coordinates from v3 native units to nm before any downstream use
    from crantpy.queries.connections import _convert_v3_coordinates_to_nm, _convert_coordinates_to_pixels
    syn = _convert_v3_coordinates_to_nm(syn)

    # Clean up synapses if requested
    if clean:
        # Remove autapses (self-connections)
        syn = syn[syn["pre_pt_root_id"] != syn["post_pt_root_id"]]
        # Remove connections involving background (ID 0)
        syn = syn[(syn["pre_pt_root_id"] != 0) & (syn["post_pt_root_id"] != 0)]
        logger.info(f"After cleaning: {len(syn)} synapses")

    if syn.empty:
        logger.warning("No synapses remaining after filtering")
        return syn

    # Apply neuron-level filtering if specified
    if min_synapses_per_neuron > 1:
        syn = _filter_by_neuron_count(syn, min_synapses_per_neuron)

    # Apply pair-level filtering if specified
    if min_synapses_per_pair > 1:
        syn = _filter_by_pair_count(syn, min_synapses_per_pair)

    if syn.empty:
        logger.warning("No synapses remaining after filtering")
        return syn

    # Convert nm coordinates to pixels if requested
    if return_pixels:
        syn = _convert_coordinates_to_pixels(syn)

    logger.info(f"Returning {len(syn)} synapses within mesh")
    return syn


@inject_dataset(allowed=CRANT_VALID_DATASETS)
def get_synapses_in_neuropils(
    neuropil_names: List[str],
    min_synapses_per_neuron: int = 1,
    min_synapses_per_pair: int = 1,
    min_size: Optional[int] = None,
    materialization: Optional[str] = "latest",
    return_pixels: bool = True,
    clean: bool = True,
    cache_path: Optional[str] = None,
    dataset: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Query synapses across multiple neuropil regions with a single CAVE request.

    Instead of querying each neuropil separately (N CAVE requests), this function
    computes the union bounding box of all requested neuropils and makes one query
    (with auto-subdivision if needed). It then assigns synapses to each neuropil
    locally using ``mesh.contains()``, and applies filtering independently per region.

    Parameters
    ----------
    neuropil_names : list of str
        Names of neuropil meshes to query. Must be valid labels from NEUROPIL_MESH_DICT.
    min_synapses_per_neuron : int, default 1
        Minimum number of synapses required from a neuron within each neuropil.
        Applied independently per neuropil.
    min_synapses_per_pair : int, default 1
        Minimum number of synapses required between a neuron pair within each neuropil.
        Applied independently per neuropil.
    min_size : int, optional
        Minimum synapse size threshold.
    materialization : str, default 'latest'
        Materialization version: 'latest' or 'live'.
    return_pixels : bool, default True
        Whether to convert coordinates from nanometers to pixels.
    clean : bool, default True
        Whether to remove autapses and background connections.
    cache_path : str, optional
        Path to a parquet file for caching the raw CAVE query result. If the file
        exists, synapse data is loaded from disk (skipping the CAVE query). If it
        does not exist, data is queried from CAVE and saved to this path. Filtering
        and mesh.contains() are always re-applied on top of the cached data, so you
        can change filtering parameters without re-downloading.
    dataset : str, optional
        Dataset to use. If None, uses the default dataset.

    Returns
    -------
    dict of str -> pd.DataFrame
        Mapping from neuropil name to a DataFrame of synapses within that neuropil.

    Examples
    --------
    >>> import crantpy as cp
    >>> result = cp.get_synapses_in_neuropils(
    ...     neuropil_names=['ellipsoid_body', 'protocerebral_bridge', 'fan_shaped_body'],
    ...     min_synapses_per_neuron=15,
    ...     min_synapses_per_pair=5,
    ...     cache_path='synapse_cache.parquet',
    ... )
    >>> result['ellipsoid_body'].head()

    See Also
    --------
    get_synapses_in_mesh : Query synapses within a single mesh
    load_neuropil_mesh : Load neuropil mesh by name

    Notes
    -----
    - Returned root IDs are normalized to latest roots so they remain aligned with
      current proofreading/annotation tables.
    """
    import os
    from crantpy.utils.cave.load import get_cave_client

    if materialization not in ("live", "latest"):
        raise ValueError("materialization must be either 'live' or 'latest'")

    _validate_neuropil_names(neuropil_names)

    # Load all meshes
    logger.info(f"Loading {len(neuropil_names)} neuropil meshes...")
    meshes: Dict[str, tm.Trimesh] = {}
    for name in neuropil_names:
        meshes[name] = load_neuropil_mesh(name)

    # Compute union bounding box across all meshes
    all_mins = np.array([m.bounds[0] for m in meshes.values()])
    all_maxs = np.array([m.bounds[1] for m in meshes.values()])
    union_min = all_mins.min(axis=0)
    union_max = all_maxs.max(axis=0)
    logger.info(f"Union bounding box (mesh nm): min={union_min}, max={union_max}")

    # Convert nm bounding box to synapses_v3 native units for the CAVE spatial filter
    bbox = [
        (union_min / _SYN_V3_RES).tolist(),
        (union_max / _SYN_V3_RES).tolist(),
    ]
    logger.info(f"CAVE query bbox (v3 units): min={bbox[0]}, max={bbox[1]}")


    # Load from cache or query CAVE
    # The cache stores the raw synapse query for a specific union bounding box.
    # We validate that the cached bbox covers the current request to avoid
    # silently returning incomplete data when neuropils change.
    meta_path = f"{cache_path}.meta.json" if cache_path else None
    cache_valid = False
    if cache_path and os.path.exists(cache_path):
        if meta_path and os.path.exists(meta_path):
            import json

            with open(meta_path) as f:
                meta = json.load(f)
            cached_min = np.array(meta.get("bbox_min", []))
            cached_max = np.array(meta.get("bbox_max", []))
            if (
                cached_min.shape == (3,)
                and cached_max.shape == (3,)
                and np.all(cached_min <= union_min)
                and np.all(cached_max >= union_max)
            ):
                cache_valid = True
            else:
                logger.warning(
                    "Cached bbox does not cover the current neuropil set; "
                    "re-querying CAVE (delete %s to silence this warning)",
                    cache_path,
                )
        else:
            # Legacy cache without metadata — trust it but warn
            logger.warning(
                "Cache file %s has no metadata sidecar; cannot verify bbox coverage. "
                "Delete the cache file to force a fresh query.",
                cache_path,
            )
            cache_valid = True

    if cache_valid:
        logger.info("Cache hit for raw union-bbox synapse data at %s", cache_path)
        syn = pd.read_parquet(cache_path)
        logger.info(
            "Loaded %d raw synapses from cache; downstream filtering and mesh.contains() will still be applied",
            len(syn),
        )
    else:
        if cache_path:
            logger.info("Cache miss for raw union-bbox synapse data at %s", cache_path)
        client = get_cave_client(dataset=dataset, clear_cache=True)
        logger.info("Querying synapses within union bounding box...")
        syn, _ = _query_synapses_in_bbox(client, bbox, materialization)
        if cache_path:
            logger.info(
                "Retrieved raw union-bbox synapse data from CAVE; downstream filtering and mesh.contains() will still be applied",
            )

        if cache_path and not syn.empty:
            import json

            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            syn.to_parquet(cache_path, index=False)
            with open(meta_path, "w") as f:
                json.dump(
                    {"bbox_min": union_min.tolist(), "bbox_max": union_max.tolist()}, f
                )
            logger.info("Cached %d raw synapses to %s", len(syn), cache_path)

    if syn.empty:
        logger.warning("No synapses found in union bounding box")
        return {name: syn.copy() for name in neuropil_names}

    logger.info(f"Retrieved {len(syn)} synapses within union bounding box")

    # Apply size filter globally (before per-neuropil work)
    if min_size is not None and "size" in syn.columns:
        syn = syn[syn["size"] >= min_size]
        logger.info(f"After size filtering: {len(syn)} synapses")

    syn = _normalize_synapse_root_ids(syn, dataset=dataset)

    # Clean globally
    if clean:
        syn = syn[syn["pre_pt_root_id"] != syn["post_pt_root_id"]]
        syn = syn[(syn["pre_pt_root_id"] != 0) & (syn["post_pt_root_id"] != 0)]
        logger.info(f"After cleaning: {len(syn)} synapses")

    if syn.empty:
        logger.warning("No synapses remaining after global filtering")
        return {name: syn.copy() for name in neuropil_names}

    # Convert coordinates from v3 native units to nm before mesh comparison and pixel conversion
    from crantpy.queries.connections import _convert_v3_coordinates_to_nm
    syn = _convert_v3_coordinates_to_nm(syn)

    # Extract center-point coordinates (now in nm, matching mesh space)
    ctr_coords = np.vstack(syn["ctr_pt_position"].values).astype(float)

    # Assign synapses to each neuropil and apply per-neuropil filtering
    results: Dict[str, pd.DataFrame] = {}
    for name, mesh in meshes.items():
        logger.info(f"Checking synapses against {name} mesh...")
        inside_mask = _batched_mesh_contains(mesh, ctr_coords)
        neuropil_syn = syn[inside_mask].copy()
        logger.info(f"Found {len(neuropil_syn)} synapses inside {name}")

        if neuropil_syn.empty:
            results[name] = neuropil_syn
            continue

        # Per-neuropil neuron filtering
        if min_synapses_per_neuron > 1:
            neuropil_syn = _filter_by_neuron_count(
                neuropil_syn, min_synapses_per_neuron, label=name
            )

        # Per-neuropil pair filtering
        if min_synapses_per_pair > 1 and not neuropil_syn.empty:
            neuropil_syn = _filter_by_pair_count(
                neuropil_syn, min_synapses_per_pair, label=name
            )

        # Convert nm coordinates to pixels if requested (syn is already in nm)
        if return_pixels and not neuropil_syn.empty:
            from crantpy.queries.connections import _convert_coordinates_to_pixels
            neuropil_syn = _convert_coordinates_to_pixels(neuropil_syn)

        results[name] = neuropil_syn

    total = sum(len(df) for df in results.values())
    logger.info(
        f"Returning {total} total synapses across {len(neuropil_names)} neuropils"
    )
    return results
