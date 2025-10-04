# -*- coding: utf-8 -*-
"""
This module provides utility functions for cave-related segmentation operations.

Ported and adapted from fafbseg-py (https://github.com/navis-org/fafbseg-py)
for use with CRANT datasets.

Available Functions
-------------------
Utilities:
    - _check_bounds_coverage: Check if bounds are within CloudVolume coverage (internal)

Core Segmentation:
    - roots_to_supervoxels: Get supervoxels making up given neurons
    - supervoxels_to_roots: Get root IDs for given supervoxels
    - update_ids: Update root IDs to their latest versions

Location-based Queries:
    - locs_to_supervoxels: Get supervoxel IDs at given locations
    - locs_to_segments: Get root IDs at given locations
    - snap_to_id: Snap locations to the correct segmentation ID

Neuron Analysis:
    - neuron_to_segments: Get root IDs overlapping with a neuron
    - get_lineage_graph: Get lineage graph showing edit history

Voxel Operations:
    - get_voxels: Fetch voxels making up a given root ID
    - get_segmentation_cutout: Fetch cutout of segmentation

Temporal Analysis:
    - find_common_time: Find time when root IDs co-existed

Note: is_valid_root, is_valid_supervoxel, and is_latest_roots are in helpers.py

Coordinate System Notes
-----------------------
Understanding coordinate systems is critical when working with segmentation data:

**CAVE API (ChunkedGraph/Materialization):**
    - Uses nanometers for all spatial coordinates
    - Base resolution: [8, 8, 42] nm/voxel (X, Y, Z)
    - Full dataset coverage

**CloudVolume API:**
    - Works in voxel space at the current MIP (scale) level
    - **IMPORTANT**: `vol.bounds` returns voxel coordinates, NOT nanometers
    - Resolution varies by MIP level:
        * MIP 0: [16, 16, 42] nm/voxel (missing CAVE's 8nm base layer!)
        * MIP 1: [32, 32, 42] nm/voxel
    - Limited spatial coverage: Only ~360 x 344 x 257 µm region
    - Does NOT cover the full CAVE dataset

**Coordinate Conversions:**
    - nm → voxels: divide by `vol.scale["resolution"]`
    - voxels → nm: multiply by `vol.scale["resolution"]`
    - Always use CAVE base resolution for voxel coordinate inputs
    - CloudVolume resolution changes with MIP level

**Best Practices:**
    1. Always provide explicit bounds for voxel operations
    2. Use small regions (< 10 µm) for `get_voxels` queries
    3. For full neurons, use `get_l2_skeleton` or `get_mesh` instead
    4. Test bounds with `get_segmentation_cutout` before large queries
    5. When in doubt, use nanometers (the CAVE standard)
"""
import os
import time
import numpy as np
from diskcache import Cache
from typing import Optional, Union, Iterable, Dict
import pandas as pd

import navis

from datetime import datetime

from crantpy.utils.config import (
    CRANT_CACHE_DIR,
    CRANT_VALID_DATASETS,
)
from crantpy.utils.decorators import (
    inject_dataset,
)
from crantpy.utils.cave.load import get_cloudvolume, get_cave_client
from crantpy.utils.cave.helpers import parse_root_ids, is_latest_roots
from crantpy.utils.helpers import retry, make_iterable, parse_timestamp
from crantpy.utils.types import IDs, Neurons, Timestamp
from crantpy.utils.seatable import get_all_seatable_annotations


def _check_bounds_coverage(
    bounds: np.ndarray,
    mip: int,
    dataset: Optional[str] = None,
) -> tuple[bool, dict]:
    """Check if given bounds are within CloudVolume coverage.

    Parameters
    ----------
    bounds : np.ndarray
        Bounding box in nanometers, shape (3, 2) or (2, 3)
    mip : int
        MIP level to check
    dataset : str, optional
        Dataset to use

    Returns
    -------
    is_valid : bool
        True if bounds are within CloudVolume coverage
    info : dict
        Dictionary with details: 'within_bounds', 'requested_voxels',
        'available_voxels', 'resolution'
    """
    from crantpy.utils.cave.load import get_cloudvolume, get_cave_client

    bounds = np.asarray(bounds)
    if bounds.shape == (2, 3):
        bounds = bounds.T

    vol = get_cloudvolume(dataset)
    vol.mip = mip

    resolution = np.array(vol.scale["resolution"])
    bounds_voxels = (bounds / resolution[:, None]).round().astype(int)

    vol_bounds = vol.bounds
    vol_bounds_mn = np.array(
        [vol_bounds.minpt[0], vol_bounds.minpt[1], vol_bounds.minpt[2]]
    )
    vol_bounds_mx = np.array(
        [vol_bounds.maxpt[0], vol_bounds.maxpt[1], vol_bounds.maxpt[2]]
    )

    within_bounds = np.all(bounds_voxels[:, 0] >= vol_bounds_mn) and np.all(
        bounds_voxels[:, 1] <= vol_bounds_mx
    )

    info = {
        "within_bounds": within_bounds,
        "requested_voxels": bounds_voxels,
        "available_voxels": np.column_stack([vol_bounds_mn, vol_bounds_mx]),
        "resolution": resolution,
        "mip": mip,
    }

    return within_bounds, info


@inject_dataset(allowed=CRANT_VALID_DATASETS)
def roots_to_supervoxels(
    neurons: Neurons,
    clear_cache: bool = False,
    progress: bool = True,
    *,
    dataset: Optional[str] = None,
):
    """Get supervoxels making up given neurons.

    Parameters
    ----------
    neurons :      Neurons = str | int | np.int64 | navis.BaseNeuron | Iterables of previous types | navis.NeuronList | NeuronCriteria
    clear_cache :  bool
                   If True, bypasses the cache and fetches a new volume.
    progress :     bool
                   If True, show progress bar.
    dataset :      str
                    The dataset to use. If not provided, uses the default dataset.

    Returns
    -------
    dict
                    A dictionary mapping neuron IDs to lists of supervoxel IDs.

    Examples
    --------
    >>> from crantpy.utils.cave.segmentation import roots_to_supervoxels
    >>> roots_to_supervoxels([123456, 789012], dataset='latest')
    {123456: [1, 2, 3], 789012: [4, 5, 6]}

    """
    # get the neuron IDs
    neurons = parse_root_ids(neurons)

    # Make sure we're not getting bogged down with duplicates
    neurons = np.unique(neurons)

    if len(neurons) <= 1:
        progress = False

    # Get the volume
    vol = get_cloudvolume(dataset, clear_cache=clear_cache, check_stale=True)

    svoxels = {}
    # See what we can get from cache
    if not clear_cache:
        # Cache for root -> supervoxels
        # Grows to max 1Gb by default and persists across sessions
        with Cache(directory=os.path.join(CRANT_CACHE_DIR, "svoxel_cache")) as sv_cache:
            # See if we have any of these roots cached
            with sv_cache.transact():
                # N.B. np.isin does seem to say "True" when in fact it's not
                is_cached = np.array([int(i) in sv_cache for i in neurons])

            # Add supervoxels from cache if we have any
            if np.any(is_cached):
                # Get values from cache
                with sv_cache.transact():
                    # N.B. int(i) is required because of stupid numpy 2.0
                    svoxels.update({i: sv_cache[int(i)] for i in neurons[is_cached]})

    # Get the supervoxels for the roots that are still missing
    # We need to convert keys to integer array because otherwise there is a
    # mismatch in types (int vs np.int?) which causes all root IDs to be in miss
    # -> I think that's because of the way disk cache works
    miss = neurons[~np.isin(neurons, np.array(list(svoxels.keys()), dtype=np.int64))]
    get_leaves = retry(vol.get_leaves)
    with navis.config.tqdm(
        desc="Querying", total=len(neurons), disable=not progress, leave=False
    ) as pbar:
        # Update for those for which we had cached data
        pbar.update(len(svoxels))

        for i in miss:
            svoxels[i] = get_leaves(i, bbox=vol.meta.bounds(0), mip=0)
            pbar.update()

    # Update cache
    if not clear_cache:
        with Cache(directory=os.path.join(CRANT_CACHE_DIR, "svoxel_cache")) as sv_cache:
            with sv_cache.transact():
                for i in miss:
                    sv_cache[int(i)] = svoxels[i]

    return svoxels


@inject_dataset(allowed=CRANT_VALID_DATASETS)
def supervoxels_to_roots(
    ids: IDs,
    timestamp: Optional[Union[str, Timestamp]] = "mat",
    clear_cache: bool = False,
    batch_size: int = 10_000,
    stop_layer: int = 8,
    retry: bool = True,
    progress: bool = True,
    *,
    dataset: Optional[str] = None,
):
    """Get root(s) for given supervoxel(s).

    Parameters
    ----------
    ids :           IDs = str | int | np.int64 | Iterables of previous types
                    Supervoxel ID(s) to find the root(s) for. Also works for e.g. L2 IDs.
    timestamp :     Timestamp = str | int | np.int64 | datetime | np.datetime64 | pd.Timestamp or str starting with "mat"
                    Get roots at given date (and time). Int must be unix timestamp. String must be ISO 8601 - e.g. '2021-11-15'.
                    "mat" will use the timestamp of the most recent materialization. You can also use e.g. "mat_<version>" to get the
                    root ID at a specific materialization.
    clear_cache :   bool
                    If True, bypasses the cache and fetches a new volume.
    batch_size :    int
                    Max number of supervoxel IDs per query. Reduce batch size if you experience time outs.
    stop_layer :    int
                    Set e.g. to ``2`` to get L2 IDs instead of root IDs.
    retry :         bool
                    Whether to retry if a batched query fails.
    progress :      bool
                    If True, show progress bar.
    dataset :       str
                    The dataset to use. If not provided, uses the default dataset.

    Returns
    -------
    roots  :        numpy array
                    Roots corresponding to supervoxels in `x`.

    Examples
    --------
    >>> from crantpy.utils.cave.segmentation import supervoxels_to_roots
    >>> supervoxels_to_roots([123456, 789012], dataset='latest')
    [1, 2]

    """
    # Make sure we are working with an array of integers
    ids = make_iterable(ids, force_type=np.int64)

    # Parse the volume
    vol = get_cloudvolume(dataset, clear_cache=clear_cache, check_stale=True)

    # Prepare results array
    roots = np.zeros(ids.shape, dtype=np.int64)

    if isinstance(timestamp, str) and timestamp.startswith("mat"):
        client = get_cave_client(dataset=dataset)
        if timestamp == "mat" or timestamp == "mat_latest":
            timestamp = client.materialize.get_timestamp()
        else:
            # Split e.g. 'mat_432' to extract version and query timestamp
            version = int(timestamp.split("_")[1])
            timestamp = client.materialize.get_timestamp(version)

    if timestamp is not None:
        # Parse timestamp but keep it as integer for cloudvolume
        timestamp = int(parse_timestamp(timestamp))

    with navis.config.tqdm(
        desc="Fetching roots",
        leave=False,
        total=len(ids),
        disable=not progress or len(ids) < batch_size,
    ) as pbar:
        for i in range(0, len(ids), int(batch_size)):
            # This batch
            batch = ids[i : i + batch_size]

            # get_roots() doesn't like to be asked for zeros - causes server error
            not_zero = batch != 0
            try:
                roots[i : i + batch_size][not_zero] = vol.get_roots(
                    batch[not_zero], stop_layer=stop_layer, timestamp=timestamp
                )
            except KeyboardInterrupt:
                raise
            except BaseException:
                if not retry:
                    raise
                time.sleep(1)
                roots[i : i + batch_size][not_zero] = vol.get_roots(
                    batch[not_zero], stop_layer=stop_layer, timestamp=timestamp
                )

            pbar.update(len(batch))

    return roots


@inject_dataset(allowed=CRANT_VALID_DATASETS)
def update_ids(
    x: Union[Neurons, pd.DataFrame],
    supervoxels: Optional[IDs] = None,
    timestamp: Optional[Union[str, Timestamp]] = None,
    stop_layer: int = 2,
    progress: bool = True,
    dataset: Optional[str] = None,
    use_annotations: bool = True,
    clear_cache: bool = False,
) -> pd.DataFrame:
    """Update root IDs to their latest versions.

    This function prioritizes using supervoxel IDs from annotations when available,
    falling back to chunkedgraph methods only when necessary.

    Parameters
    ----------
    x : Neurons or pd.DataFrame
        Root IDs to update. If DataFrame, must contain 'root_id' column and
        optionally 'supervoxel_id' column.
    supervoxels : IDs, optional
        Supervoxel IDs corresponding to the root IDs. If provided, these will
        be used instead of looking up annotations.
    timestamp : Timestamp, optional
        Target timestamp. Can be "mat" for latest materialization.
    stop_layer : int, default 2
        Stop layer for chunkedgraph operations when supervoxels unavailable.
    progress : bool, default True
        Whether to show progress bar.
    dataset : str, optional
        Dataset to use.
    use_annotations : bool, default True
        Whether to look up supervoxels from annotations when not provided.
    clear_cache : bool, default False
        Whether to clear annotation cache.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: old_id, new_id, confidence, changed

    Examples
    --------
    >>> from crantpy.utils.cave.segmentation import update_ids
    >>> update_ids([123456789, 987654321])
       old_id      new_id  confidence  changed
    0  123456789  123456789     1.0     False
    1  987654321  999999999     0.85    True

    >>> # With supervoxels
    >>> update_ids([123456789], supervoxels=[111222333])
       old_id      new_id  confidence  changed
    0  123456789  123456789     1.0     False
    """
    import logging

    # Handle DataFrame input
    if isinstance(x, pd.DataFrame):
        if "root_id" not in x.columns:
            raise ValueError("DataFrame must contain 'root_id' column")

        raw_root_ids = x["root_id"].values
        raw_supervoxels = (
            x["supervoxel_id"].values if "supervoxel_id" in x.columns else None
        )
    else:
        # Handle Neurons input - this already handles None/NaN through parse_neuroncriteria
        try:
            raw_root_ids = parse_root_ids(x)
            raw_supervoxels = None
        except Exception as e:
            logging.warning(f"Failed to parse root IDs: {e}")
            return pd.DataFrame(columns=["old_id", "new_id", "confidence", "changed"])

    # Filter out None/NaN root IDs - we can't work with invalid root IDs
    valid_root_mask = pd.notna(raw_root_ids) & (raw_root_ids != 0)

    if not valid_root_mask.any():
        logging.warning("No valid root IDs found. All root IDs are None, NaN, or zero.")
        return pd.DataFrame(columns=["old_id", "new_id", "confidence", "changed"])

    # Extract valid root IDs and convert to proper format
    valid_indices = np.where(valid_root_mask)[0]
    root_ids = raw_root_ids[valid_root_mask]

    try:
        root_ids = make_iterable(root_ids, force_type=np.int64)
    except Exception as e:
        logging.error(f"Failed to convert root IDs to int64: {e}")
        return pd.DataFrame(columns=["old_id", "new_id", "confidence", "changed"])

    # Handle supervoxels with None/NaN filtering
    supervoxel_mask = np.zeros(len(root_ids), dtype=bool)
    supervoxels_clean = None

    if supervoxels is not None:
        # User provided supervoxels
        raw_supervoxels = supervoxels
    elif raw_supervoxels is not None:
        # Got supervoxels from DataFrame
        raw_supervoxels = raw_supervoxels[
            valid_root_mask
        ]  # Only keep supervoxels for valid root IDs
    elif use_annotations:
        # Try to get supervoxels from annotations
        try:
            annotations = get_all_seatable_annotations(
                dataset=dataset, clear_cache=clear_cache
            )
            if (
                "supervoxel_id" in annotations.columns
                and "root_id" in annotations.columns
            ):
                # Filter out NaN values and create clean DataFrame
                clean_annotations = annotations[["root_id", "supervoxel_id"]].dropna()

                # Check for duplicates and warn if found
                duplicates = clean_annotations.duplicated(
                    subset=["root_id"], keep=False
                )
                if duplicates.any():
                    duplicate_roots = clean_annotations[duplicates]["root_id"].unique()
                    logging.warning(
                        f"Multiple supervoxel IDs found for {len(duplicate_roots)} root IDs. Using first occurrence for each."
                    )

                # Create mapping using drop_duplicates (keeps first occurrence)
                supervoxel_map = clean_annotations.drop_duplicates(
                    subset=["root_id"]
                ).set_index("root_id")["supervoxel_id"]

                # Map supervoxels to our root IDs using pandas vectorized operations
                root_ids_series = pd.Series(root_ids)
                supervoxels_series = root_ids_series.map(supervoxel_map)

                # Convert back to numpy arrays, handling NaN values
                raw_supervoxels = supervoxels_series.values

                if pd.notna(raw_supervoxels).any():
                    logging.info(
                        f"Found supervoxel IDs for {pd.notna(raw_supervoxels).sum()}/{len(root_ids)} root IDs from annotations"
                    )
            else:
                raw_supervoxels = None
                logging.info("No supervoxel_id column found in annotations")
        except Exception as e:
            logging.warning(f"Could not retrieve supervoxels from annotations: {e}")
            raw_supervoxels = None
    else:
        raw_supervoxels = None

    # Process supervoxels if we have them
    if raw_supervoxels is not None:
        # First convert raw_supervoxels to a proper numpy array to avoid indexing issues
        try:
            raw_supervoxels = np.asarray(raw_supervoxels)
        except Exception as e:
            logging.warning(
                f"Failed to convert supervoxels to array: {e}. "
                "Expected input: a list, numpy array, or pandas Series of integer supervoxel IDs (not None, NaN, or zero). "
                "Example: [12345, 67890, 13579]. Please check your input format."
            )
            raw_supervoxels = None

        # Check length match
        if raw_supervoxels is not None and len(raw_supervoxels) != len(root_ids):
            logging.warning(
                f"Number of supervoxels ({len(raw_supervoxels)}) does not match root IDs ({len(root_ids)}). Ignoring supervoxels."
            )
            raw_supervoxels = None
        elif raw_supervoxels is not None:
            # Filter out None/NaN/zero supervoxels
            supervoxel_mask = pd.notna(raw_supervoxels) & (raw_supervoxels != 0)

            if supervoxel_mask.any():
                try:
                    # Only convert valid supervoxels
                    valid_supervoxels = raw_supervoxels[supervoxel_mask]
                    supervoxels_clean = make_iterable(
                        valid_supervoxels, force_type=np.int64
                    )
                    logging.info(
                        f"Using supervoxels for {supervoxel_mask.sum()}/{len(root_ids)} root IDs"
                    )
                except Exception as e:
                    logging.warning(
                        f"Failed to convert supervoxels to int64: {e}. Falling back to chunkedgraph method."
                    )
                    supervoxel_mask = np.zeros(len(root_ids), dtype=bool)
                    supervoxels_clean = None

    # Check which valid IDs are already latest
    is_latest_valid = np.zeros(len(root_ids), dtype=bool)  # For valid IDs only
    if len(root_ids) > 0:  # Only check if there are valid IDs
        is_latest_valid = is_latest_roots(
            root_ids,
            timestamp=timestamp,
            dataset=dataset,
            progress=progress,
            validate_ids=False,
        )

    # Initialize result DataFrame for ALL original indices (including invalid ones)
    result = pd.DataFrame(
        {
            "old_id": raw_root_ids,
            "new_id": raw_root_ids.copy(),
            "confidence": np.ones(len(raw_root_ids), dtype=float),
            "changed": np.zeros(len(raw_root_ids), dtype=bool),
        }
    )

    # Mark invalid root IDs with 0 confidence
    result.loc[~valid_root_mask, "confidence"] = 0.0

    # Find valid IDs that need updating (indices into root_ids array)
    valid_needs_update = ~is_latest_valid

    if not valid_needs_update.any():
        logging.info("All valid root IDs are already the latest. No updates needed.")
        return result

    # Handle materialization timestamps
    if isinstance(timestamp, str) and timestamp.startswith("mat"):
        client = get_cave_client(dataset=dataset)
        if timestamp == "mat" or timestamp == "mat_latest":
            timestamp = client.materialize.get_timestamp()
        else:
            try:
                version = int(timestamp.split("_")[1])
                timestamp = client.materialize.get_timestamp(version)
            except (IndexError, ValueError):
                raise ValueError(
                    f"Invalid materialization timestamp format: {timestamp}"
                )

    # Parse timestamp
    if timestamp is not None:
        # Parse timestamp but keep it as integer for cloudvolume
        timestamp = int(parse_timestamp(timestamp))

    # Get indices that need updating (indices into root_ids array, not raw_root_ids)
    update_indices = np.where(valid_needs_update)[0]

    # Split into those with and without supervoxels
    has_supervoxels = (
        supervoxel_mask[update_indices]
        if supervoxels_clean is not None
        else np.zeros(len(update_indices), dtype=bool)
    )

    with navis.config.tqdm(
        desc="Updating IDs",
        total=len(update_indices),
        disable=not progress or len(update_indices) <= 1,
        leave=False,
    ) as pbar:

        # Update using supervoxels (fast path)
        if has_supervoxels.any() and supervoxels_clean is not None:
            svoxel_update_indices = update_indices[has_supervoxels]

            # Create mapping from update indices to supervoxel indices
            supervoxel_indices = np.where(supervoxel_mask)[0]
            svoxel_lookup = {}
            svoxel_idx = 0
            for i, has_sv in enumerate(supervoxel_mask):
                if has_sv:
                    svoxel_lookup[i] = svoxel_idx
                    svoxel_idx += 1

            # Get supervoxel IDs for the indices that need updating
            svoxel_ids = []
            for idx in svoxel_update_indices:
                if idx in svoxel_lookup:
                    svoxel_ids.append(supervoxels_clean[svoxel_lookup[idx]])

            if svoxel_ids:
                try:
                    new_roots = supervoxels_to_roots(
                        svoxel_ids,
                        timestamp=timestamp,
                        dataset=dataset,
                        progress=progress,
                    )

                    for i, (update_idx, new_root) in enumerate(
                        zip(svoxel_update_indices, new_roots)
                    ):
                        original_idx = valid_indices[
                            update_idx
                        ]  # Map back to original DataFrame index
                        result.loc[original_idx, "new_id"] = new_root
                        result.loc[original_idx, "confidence"] = 1.0
                        result.loc[original_idx, "changed"] = (
                            new_root != root_ids[update_idx]
                        )
                        pbar.update(1)

                except Exception as e:
                    logging.warning(f"Failed to update IDs using supervoxels: {e}")
                    import traceback

                    traceback.print_exc()
                    # Fall back to chunkedgraph method for these
                    has_supervoxels[:] = False

        # Update using chunkedgraph suggest_latest_roots (robust but slower)
        no_supervoxels = update_indices[~has_supervoxels]

        if len(no_supervoxels) > 0:
            client = get_cave_client(dataset=dataset)
            suggest_latest = retry(client.chunkedgraph.suggest_latest_roots)

            for update_idx in no_supervoxels:
                original_idx = valid_indices[
                    update_idx
                ]  # Map back to original DataFrame index
                old_id = root_ids[update_idx]

                try:
                    suggestions, overlap = suggest_latest(
                        old_id,
                        timestamp=timestamp,
                        stop_layer=stop_layer,
                        return_all=True,
                        return_fraction_overlap=True,
                    )

                    if suggestions:
                        # Get the suggestion with highest overlap
                        candidates = [
                            {"new_root_id": s, "overlap_fraction": f}
                            for s, f in zip(suggestions, overlap)
                        ]
                        best_suggestion = max(
                            candidates, key=lambda x: x.get("overlap_fraction", 0)
                        )
                        new_id = best_suggestion["new_root_id"]
                        confidence = best_suggestion.get("overlap_fraction", 0.0)
                    else:
                        # No suggestions found, keep original
                        new_id = old_id
                        confidence = 0.0

                    result.loc[original_idx, "new_id"] = new_id
                    result.loc[original_idx, "confidence"] = confidence
                    result.loc[original_idx, "changed"] = new_id != old_id

                except Exception as e:
                    logging.warning(f"Failed to update ID {old_id}: {e}")
                    import traceback

                    traceback.print_exc()
                    # Keep original ID with 0 confidence
                    result.loc[original_idx, "new_id"] = old_id
                    result.loc[original_idx, "confidence"] = 0.0
                    result.loc[original_idx, "changed"] = False

                pbar.update(1)

    return result


@inject_dataset(allowed=CRANT_VALID_DATASETS)
def locs_to_supervoxels(
    locs: Union[np.ndarray, pd.DataFrame],
    mip: int = 0,
    coordinates: str = "nm",
    progress: bool = True,
    *,
    dataset: Optional[str] = None,
):
    """Retrieve supervoxel IDs at given location(s).

    Parameters
    ----------
    locs :          array-like | pandas.DataFrame
                    Array of x/y/z coordinates. If DataFrame must contain
                    'x', 'y', 'z' columns.
    mip :           int
                    Scale to query. Lower mip = more precise but slower;
                    higher mip = faster but less precise. The default is 0
                    which is the highest resolution.
    coordinates :   "nm" | "voxel"
                    Units in which your coordinates are in. "nm" for nanometers,
                    "voxel" for voxel coordinates.
    progress :      bool
                    If True, show progress bar.
    dataset :       str
                    The dataset to use. If not provided, uses the default dataset.

    Returns
    -------
    numpy.array
                    List of supervoxel IDs in the same order as ``locs``.
                    Invalid locations will be returned with ID 0.

    Examples
    --------
    >>> from crantpy.utils.cave.segmentation import locs_to_supervoxels
    >>> locs = [[133131, 55615, 3289], [132802, 55661, 3289]]
    >>> locs_to_supervoxels(locs, dataset='latest')
    array([79801454835332154, 79731086091150780], dtype=uint64)
    """
    if isinstance(locs, pd.DataFrame):
        if np.all(np.isin(["x", "y", "z"], locs.columns)):
            locs = locs[["x", "y", "z"]].values
        else:
            raise ValueError("`locs` as pandas.DataFrame must have [x, y, z] columns.")

        # Make sure we are working with numbers
        if not np.issubdtype(locs.dtype, np.number):
            locs = locs.astype(np.float64)

    locs = np.asarray(locs)

    # Get the cloudvolume
    vol = get_cloudvolume(dataset)
    vol.mip = mip

    # Convert coordinates if needed
    if coordinates in ("voxel", "voxels"):
        # Get resolution info from client
        client = get_cave_client(dataset=dataset)
        res = np.array(
            [
                client.info.get_datastack_info()["viewer_resolution_x"],
                client.info.get_datastack_info()["viewer_resolution_y"],
                client.info.get_datastack_info()["viewer_resolution_z"],
            ]
        )
        locs = locs * res

    # Download the segmentation at these locations
    # We use vol.download_point_cloud which is efficient for sparse points
    try:
        from cloudvolume import CloudVolume

        result = np.zeros(len(locs), dtype=np.uint64)

        # Process in chunks for better memory management
        chunk_size = 1000
        with navis.config.tqdm(
            total=len(locs),
            desc="Fetching supervoxels",
            disable=not progress,
            leave=False,
        ) as pbar:
            for i in range(0, len(locs), chunk_size):
                chunk_locs = locs[i : i + chunk_size]

                # CloudVolume expects coordinates in nanometers
                for j, loc in enumerate(chunk_locs):
                    try:
                        # Convert to voxel coordinates for the given mip
                        voxel_loc = (loc / vol.scale["resolution"]).astype(int)

                        # Download a single voxel
                        seg = vol[
                            voxel_loc[0] : voxel_loc[0] + 1,
                            voxel_loc[1] : voxel_loc[1] + 1,
                            voxel_loc[2] : voxel_loc[2] + 1,
                        ]
                        result[i + j] = seg[0, 0, 0, 0]
                    except Exception:
                        result[i + j] = 0

                pbar.update(len(chunk_locs))

        return result

    except Exception as e:
        raise RuntimeError(f"Failed to fetch supervoxels: {e}")


@inject_dataset(allowed=CRANT_VALID_DATASETS)
def locs_to_segments(
    locs: Union[np.ndarray, pd.DataFrame],
    timestamp: Optional[Union[str, Timestamp]] = None,
    coordinates: str = "nm",
    progress: bool = True,
    *,
    dataset: Optional[str] = None,
):
    """Retrieve segment (i.e. root) IDs at given location(s).

    Parameters
    ----------
    locs :          array-like | pandas.DataFrame
                    Array of x/y/z coordinates. If DataFrame must contain
                    'x', 'y', 'z' columns.
    timestamp :     Timestamp, optional
                    Get roots at given date (and time). Int must be unix
                    timestamp. String must be ISO 8601 - e.g. '2021-11-15'.
                    "mat" will use the timestamp of the most recent
                    materialization.
    coordinates :   "nm" | "voxel"
                    Units in which your coordinates are in.
    progress :      bool
                    If True, show progress bar.
    dataset :       str
                    The dataset to use. If not provided, uses the default dataset.

    Returns
    -------
    numpy.array
                    List of root IDs in the same order as ``locs``.

    Examples
    --------
    >>> from crantpy.utils.cave.segmentation import locs_to_segments
    >>> locs = [[133131, 55615, 3289], [132802, 55661, 3289]]
    >>> locs_to_segments(locs, dataset='latest')
    array([720575940631693610, 720575940631693610])
    """
    svoxels = locs_to_supervoxels(
        locs, coordinates=coordinates, dataset=dataset, progress=progress
    )

    return supervoxels_to_roots(
        svoxels, timestamp=timestamp, dataset=dataset, progress=progress
    )


@inject_dataset(allowed=CRANT_VALID_DATASETS)
def get_lineage_graph(
    x: Union[int, np.int64],
    progress: bool = True,
    *,
    dataset: Optional[str] = None,
):
    """Get lineage graph for given neuron.

    Parameters
    ----------
    x :         int
                A single root ID.
    progress :  bool
                If True, show progress bar.
    dataset :   str
                The dataset to use. If not provided, uses the default dataset.

    Returns
    -------
    networkx.DiGraph
                The lineage graph showing the history of edits for this root ID.

    Examples
    --------
    >>> from crantpy.utils.cave.segmentation import get_lineage_graph
    >>> G = get_lineage_graph(720575940621039145)
    >>> len(G.nodes())
    150
    """
    import networkx as nx

    x = np.int64(x)

    client = get_cave_client(dataset=dataset)
    G = client.chunkedgraph.get_lineage_graph(x, as_nx_graph=True)

    # Remap operation ID
    op_remapped = {}
    for n in G:
        pred = list(G.predecessors(n))
        if pred:
            op_remapped[n] = G.nodes[pred[0]]["operation_id"]

    # Remove existing operation IDs
    for n in G.nodes:
        G.nodes[n].pop("operation_id", None)
    # Apply new IDs
    nx.set_node_attributes(G, op_remapped, name="operation_id")

    return G


@inject_dataset(allowed=CRANT_VALID_DATASETS)
def neuron_to_segments(
    x: Union[navis.TreeNeuron, navis.NeuronList],
    short: bool = False,
    coordinates: str = "nm",
    *,
    dataset: Optional[str] = None,
):
    """Get root IDs overlapping with a given neuron.

    Parameters
    ----------
    x :             Neuron/List
                    Neurons for which to return root IDs. Neurons must be
                    in the correct coordinate space for the dataset.
    short :         bool
                    If True will only return the top hit for each neuron
                    (including a confidence score).
    coordinates :   "voxel" | "nm"
                    Units the neuron(s) are in.
    dataset :       str
                    The dataset to use. If not provided, uses the default dataset.

    Returns
    -------
    overlap_matrix :    pandas.DataFrame
                        DataFrame of root IDs (rows) and neuron IDs
                        (columns) with overlap in nodes as values.
    summary :           pandas.DataFrame
                        If ``short=True``: DataFrame of top hits only.

    Examples
    --------
    >>> from crantpy.utils.cave.segmentation import neuron_to_segments
    >>> import navis
    >>> # Assuming you have a neuron in the correct space
    >>> neuron = navis.TreeNeuron(...)
    >>> summary = neuron_to_segments(neuron, short=True)
    """
    if isinstance(x, navis.TreeNeuron):
        x = navis.NeuronList(x)

    assert isinstance(x, navis.NeuronList)

    # We must not perform this on x.nodes as this is a temporary property
    nodes = x.nodes

    # Get segmentation IDs
    nodes["root_id"] = locs_to_segments(
        nodes[["x", "y", "z"]].values, coordinates=coordinates, dataset=dataset
    )

    # Count segment IDs
    seg_counts = nodes.groupby(["neuron", "root_id"], as_index=False).node_id.count()
    seg_counts.columns = ["id", "root_id", "counts"]

    # Remove seg IDs 0
    seg_counts = seg_counts[seg_counts.root_id != 0]

    # Turn into matrix where columns are skeleton IDs, segment IDs are rows
    # and values are the overlap counts
    matrix = seg_counts.pivot(index="root_id", columns="id", values="counts")

    if not short:
        return matrix

    # Handle empty matrix (no overlaps found)
    if len(matrix) == 0:
        summary = pd.DataFrame([])
        summary["id"] = x.id if hasattr(x, "id") else list(range(len(x)))
        summary["match"] = [None] * len(summary)
        summary["confidence"] = [0.0] * len(summary)
        return summary

    # Extract top IDs and scores
    top_id = matrix.index[np.argmax(matrix.fillna(0).values, axis=0)]

    # Confidence is the difference between top and 2nd score
    top_score = matrix.max(axis=0).values

    # Handle case with only one row (can't compute second-best)
    if len(matrix) == 1:
        sec_score = np.zeros(len(matrix.columns))
    else:
        sec_score = np.sort(matrix.fillna(0).values, axis=0)[-2, :]

    conf = (top_score - sec_score) / matrix.sum(axis=0).values

    summary = pd.DataFrame([])
    summary["id"] = matrix.columns
    summary["match"] = top_id
    summary["confidence"] = conf

    return summary


@inject_dataset(allowed=CRANT_VALID_DATASETS)
def get_segmentation_cutout(
    bbox: np.ndarray,
    root_ids: bool = True,
    mip: int = 0,
    coordinates: str = "nm",
    *,
    dataset: Optional[str] = None,
):
    """Fetch cutout of segmentation.

    Parameters
    ----------
    bbox :          array-like
                    Bounding box for the cutout::
                        [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
    root_ids :      bool
                    If True, will return root IDs. If False, will return
                    supervoxel IDs.
    mip :           int
                    Scale at which to fetch the cutout.
    coordinates :   "voxel" | "nm"
                    Units in which your coordinates are in.
    dataset :       str
                    The dataset to use. If not provided, uses the default dataset.

    Returns
    -------
    cutout :        np.ndarray
                    (N, M, P) array of segmentation (root or supervoxel) IDs.
    resolution :    (3, ) numpy array
                    [x, y, z] resolution of voxel in cutout.
    nm_offset :     (3, ) numpy array
                    [x, y, z] offset in nanometers of the cutout with respect
                    to the absolute coordinates.

    Examples
    --------
    >>> from crantpy.utils.cave.segmentation import get_segmentation_cutout
    >>> bbox = [[100000, 100100], [50000, 50100], [3000, 3010]]
    >>> cutout, resolution, offset = get_segmentation_cutout(bbox)
    """
    assert coordinates in ["nm", "nanometer", "nanometers", "voxel", "voxels"]

    bbox = np.asarray(bbox)
    assert bbox.ndim == 2

    if bbox.shape == (2, 3):
        pass
    elif bbox.shape == (3, 2):
        bbox = bbox.T
    else:
        raise ValueError(f"`bbox` must have shape (2, 3) or (3, 2), got {bbox.shape}")

    vol = get_cloudvolume(dataset)
    vol.mip = mip

    # Get resolution info from client
    client = get_cave_client(dataset=dataset)
    base_res = np.array(
        [
            client.info.get_datastack_info()["viewer_resolution_x"],
            client.info.get_datastack_info()["viewer_resolution_y"],
            client.info.get_datastack_info()["viewer_resolution_z"],
        ]
    )

    # First convert to nanometers
    if coordinates in ("voxel", "voxels"):
        bbox = bbox * base_res

    # Now convert to voxel at the requested mip
    bbox_voxels = (bbox / vol.scale["resolution"]).round().astype(int)

    # Validate bounds are within CloudVolume
    vol_bounds = vol.bounds
    vol_bounds_mn = np.array(
        [vol_bounds.minpt[0], vol_bounds.minpt[1], vol_bounds.minpt[2]]
    )
    vol_bounds_mx = np.array(
        [vol_bounds.maxpt[0], vol_bounds.maxpt[1], vol_bounds.maxpt[2]]
    )

    for i, dim_name in enumerate(["X", "Y", "Z"]):
        if bbox_voxels[0, i] < vol_bounds_mn[i] or bbox_voxels[1, i] > vol_bounds_mx[i]:
            raise ValueError(
                f"Requested cutout bounds exceed CloudVolume coverage at mip={mip}.\n"
                f"  {dim_name}: requested [{bbox_voxels[0, i]}, {bbox_voxels[1, i]}] voxels "
                f"({bbox[0, i]:.0f}-{bbox[1, i]:.0f} nm), "
                f"available [{int(vol_bounds_mn[i])}, {int(vol_bounds_mx[i])}] voxels.\n"
                f"  Note: CloudVolume only covers a subset of the full CAVE dataset.\n"
                f"  The requested coordinates may be outside CloudVolume's spatial coverage."
            )

    offset_nm = bbox_voxels[0] * vol.scale["resolution"]

    # Get cutout
    cutout = vol[
        bbox_voxels[0][0] : bbox_voxels[1][0],
        bbox_voxels[0][1] : bbox_voxels[1][1],
        bbox_voxels[0][2] : bbox_voxels[1][2],
    ]

    if root_ids:
        svoxels = np.unique(cutout.flatten())
        svoxels = svoxels[svoxels != 0]  # Remove zeros

        if len(svoxels) > 0:
            roots = supervoxels_to_roots(svoxels, dataset=dataset)
            sv2r = dict(zip(svoxels, roots))

            for k, v in sv2r.items():
                cutout[cutout == k] = v

    return cutout[:, :, :, 0], np.asarray(vol.scale["resolution"]), offset_nm


@inject_dataset(allowed=CRANT_VALID_DATASETS)
def get_voxels(
    x: Union[int, np.int64],
    mip: int = 0,
    bounds: Optional[np.ndarray] = None,
    sv_map: bool = False,
    thin: bool = False,
    use_l2_chunks: bool = True,
    threads: int = 1,
    progress: bool = True,
    *,
    dataset: Optional[str] = None,
):
    """Fetch voxels making up a given root ID.

    This function has two modes:
    1. L2 chunk-based (default): Fetches voxels chunk by chunk using L2 IDs
    2. Cutout-based: Downloads entire bounding box and extracts voxels

    **IMPORTANT - CloudVolume Limitations:**

    CloudVolume has limited spatial coverage (~360 x 344 x 257 µm) and is
    missing CAVE's highest resolution layer. For most use cases, prefer:
    - `get_l2_skeleton()` for full neuron morphology
    - `get_mesh()` for 3D visualization
    - This function only for specific voxel-level analysis in small regions

    Parameters
    ----------
    x :             int
                    A single root ID.
    mip :           int
                    Scale at which to fetch voxels. For example, `mip=0` is
                    at highest resolution (16x16x42 nm/voxel for CloudVolume).
                    Every subsequent `mip` halves the resolution.
                    Use higher mip for faster queries: mip=1 is often sufficient.
    bounds :        (3, 2) or (2, 3) array, optional
                    Bounding box to return voxels in (in nanometers).
                    Format: [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
                    **REQUIRED in practice** - without bounds, will attempt to
                    fetch entire neuron which may exceed CloudVolume coverage.
                    Use small regions (< 10 µm per dimension) for best results.
    sv_map :        bool
                    If True, additionally return a map with the supervoxel ID
                    for each voxel. Useful for detailed connectivity analysis.
    thin :          bool
                    If True, will remove voxels at the interface of adjacent
                    supervoxels that are not supposed to be connected according
                    to the L2 graph. Useful for neurons that self-touch.
                    **WARNING**: This is computationally expensive!
    use_l2_chunks : bool
                    If True (default), fetch voxels chunk by chunk using L2 IDs.
                    Faster and more memory efficient for neurons with L2 metadata.
                    If False, download entire bounding box as single cutout.
    threads :       int
                    Number of parallel threads for CloudVolume operations.
                    More threads = faster but more memory usage.
    progress :      bool
                    Whether to show a progress bar or not.
    dataset :       str
                    The dataset to use. If not provided, uses the default dataset.

    Returns
    -------
    voxels :        (N, 3) np.ndarray
                    Voxel coordinates in voxel space according to `mip`.
                    Each row is [x, y, z] in voxel coordinates.
    sv_map :        (N, ) np.ndarray
                    Supervoxel ID for each voxel. Only if `sv_map=True`.

    Raises
    ------
    ValueError
        If bounds exceed CloudVolume coverage or if root ID is invalid.

    Examples
    --------
    >>> from crantpy.utils.cave.segmentation import get_voxels
    >>> # RECOMMENDED: Always use explicit bounds within CloudVolume coverage
    >>> bounds = [[100000, 105000], [50000, 55000], [3000, 3100]]
    >>> voxels = get_voxels(720575940621039145, bounds=bounds, mip=1)
    >>> print(f"Retrieved {len(voxels)} voxels")

    >>> # Get voxels with supervoxel mapping for detailed analysis
    >>> voxels, svids = get_voxels(
    ...     720575940621039145,
    ...     bounds=bounds,
    ...     mip=1,
    ...     sv_map=True
    ... )
    >>> print(f"Voxels from {len(np.unique(svids))} supervoxels")

    >>> # Use cutout method for small regions without L2 metadata
    >>> voxels = get_voxels(
    ...     720575940621039145,
    ...     bounds=bounds,
    ...     mip=1,
    ...     use_l2_chunks=False
    ... )

    Notes
    -----
    **Coordinate System:**
    - Input bounds are in nanometers (CAVE standard)
    - Output voxels are in voxel space at the specified MIP level
    - CloudVolume resolution at MIP 0: [16, 16, 42] nm/voxel
    - To convert voxels to nm: voxels * resolution

    **Performance Tips:**
    - Use `mip=1` (32x32x42 nm/voxel) for faster queries when precise
      resolution isn't critical
    - Keep bounds small (< 10 µm per dimension) to avoid timeouts
    - Set `use_l2_chunks=True` (default) for neurons with L2 metadata
    - Disable `sv_map` if you don't need supervoxel IDs (faster)
    - Only use `thin=True` when absolutely necessary (very slow)

    **Common Issues:**
    - "Bounds exceed CloudVolume coverage": Your bounds are outside the
      ~360 µm cube that CloudVolume contains. Try smaller bounds or check
      if your neuron is within CloudVolume's spatial coverage.
    - Slow performance: Reduce bounds size, increase mip level, or use
      `get_l2_skeleton()` instead for full neuron morphology.
    - Empty result: The neuron may not have voxels in the specified bounds,
      or bounds may need adjustment.

    See Also
    --------
    get_l2_skeleton : Get skeleton representation (better for full neurons)
    get_mesh : Get 3D mesh (better for visualization)
    get_segmentation_cutout : Get all segmentation in a region
    """
    from .helpers import is_valid_root

    x = np.int64(x)

    # Validate root ID
    is_valid_root(x, raise_exc=True, dataset=dataset)

    vol = get_cloudvolume(dataset)
    client = get_cave_client(dataset=dataset)

    # Store original settings
    old_mip = vol.mip
    old_parallel = vol.parallel

    try:
        vol.mip = mip
        vol.parallel = threads

        # Get the resolution at this mip
        resolution = np.array(vol.scale["resolution"])
        if resolution.ndim == 0:
            raise ValueError(
                f"Resolution at mip={mip} is scalar: {resolution}. "
                "This usually indicates an invalid MIP level or misconfigured volume. "
                "Troubleshooting steps: "
                "1) Verify that the requested MIP level is supported for this dataset. "
                "2) Check that the CloudVolume setup is correct and covers the desired region. "
                "3) Ensure the dataset name is correct. "
                "Refer to the documentation for valid MIP levels and volume configuration."
            )
        if resolution.shape[0] != 3:
            raise ValueError(f"Expected 3D resolution, got shape {resolution.shape}")

        # If bounds not provided, get L2 skeleton to determine bounding box
        if bounds is None:
            if progress:
                print(f"Fetching L2 skeleton to determine bounds...")

            from crantpy.viz.l2 import get_l2_skeleton

            l2_skel = get_l2_skeleton(int(x), progress=False, dataset=dataset)

            if (
                l2_skel is not None
                and hasattr(l2_skel, "nodes")
                and len(l2_skel.nodes) > 0
            ):
                nodes = l2_skel.nodes
                bbox_nm = np.array(
                    [
                        [nodes["x"].min(), nodes["x"].max()],
                        [nodes["y"].min(), nodes["y"].max()],
                        [nodes["z"].min(), nodes["z"].max()],
                    ]
                )

                # Add padding (10% on each side)
                bbox_size = bbox_nm[:, 1] - bbox_nm[:, 0]
                padding = bbox_size * 0.1
                bounds = np.column_stack(
                    [bbox_nm[:, 0] - padding, bbox_nm[:, 1] + padding]
                )

                if progress:
                    print(f"Determined bounds from L2 skeleton: {bounds.T}")
            else:
                raise ValueError(
                    f"Could not fetch L2 skeleton for root ID {x}. "
                    "Please check if the root ID is valid, or provide explicit bounds to proceed."
                )

        # Ensure bounds are in the right format
        bounds = np.asarray(bounds)
        if bounds.shape == (2, 3):
            bounds = bounds.T

        # Validate bounds are within volume
        bounds_voxels = (bounds / resolution[:, None]).round().astype(int)
        vol_bounds = vol.bounds
        # vol.bounds returns coordinates in voxel space at current MIP, NOT nanometers!
        vol_bounds_mn = np.array(
            [vol_bounds.minpt[0], vol_bounds.minpt[1], vol_bounds.minpt[2]]
        )
        vol_bounds_mx = np.array(
            [vol_bounds.maxpt[0], vol_bounds.maxpt[1], vol_bounds.maxpt[2]]
        )

        for i, dim_name in enumerate(["X", "Y", "Z"]):
            if (
                bounds_voxels[i, 0] < vol_bounds_mn[i]
                or bounds_voxels[i, 1] > vol_bounds_mx[i]
            ):
                raise ValueError(
                    f"Requested bounds exceed volume limits at mip={mip}.\n"
                    f"  {dim_name}: requested [{bounds_voxels[i, 0]}, {bounds_voxels[i, 1]}] voxels, "
                    f"available [{int(vol_bounds_mn[i])}, {int(vol_bounds_mx[i])}] voxels.\n"
                    f"  Try using a higher mip level (e.g., mip={mip+1}) or smaller bounds."
                )

        if use_l2_chunks and hasattr(vol, "mesh") and hasattr(vol.mesh, "meta"):
            # L2 chunk-based method (adapted from FlyWire)
            voxels, svids = _get_voxels_l2_chunks(
                x, vol, client, bounds, resolution, sv_map or thin, progress, dataset
            )
        else:
            # Cutout-based method (original implementation)
            voxels, svids = _get_voxels_cutout(
                x, bounds, resolution, mip, sv_map or thin, progress, dataset
            )

        # Thinning: remove interface voxels between non-connected L2 chunks
        if thin and len(voxels) > 0:
            if progress:
                print("Thinning voxels at supervoxel interfaces...")
            voxels, svids = _thin_voxels(x, voxels, svids, vol, progress, dataset)

        if not sv_map:
            return voxels
        else:
            return voxels, svids

    finally:
        # Restore original settings
        vol.mip = old_mip
        vol.parallel = old_parallel


def _get_voxels_l2_chunks(
    x, vol, client, bounds, resolution, return_svids, progress, dataset
):
    """Fetch voxels using L2 chunk-based method."""
    from .l2 import chunks_to_nm

    # Get L2 chunks and supervoxels for this neuron
    l2_ids = client.chunkedgraph.get_leaves(x, stop_layer=2)
    sv = roots_to_supervoxels(x, dataset=dataset, progress=False)[int(x)]

    # Turn l2_ids into chunk indices
    l2_ix = [np.array(vol.mesh.meta.meta.decode_chunk_position(l)) for l in l2_ids]
    l2_ix = np.unique(l2_ix, axis=0)

    # Convert to nm then to voxel space
    l2_nm = np.asarray(chunks_to_nm(l2_ix, vol=vol))
    l2_vxl = (l2_nm / resolution).astype(int)

    # Filter chunks to those within bounds
    bounds_vxl = (bounds / resolution[:, None]).astype(int)
    in_bounds = np.all(l2_vxl >= bounds_vxl[:, 0], axis=1) & np.all(
        l2_vxl <= bounds_vxl[:, 1], axis=1
    )
    l2_vxl = l2_vxl[in_bounds]

    if len(l2_vxl) == 0:
        empty_voxels = np.array([], dtype=np.uint32).reshape(0, 3)
        empty_svids = np.array([], dtype=np.uint64) if return_svids else None
        return empty_voxels, empty_svids

    # Get chunk size
    ch_size = np.array(vol.mesh.meta.meta.graph_chunk_size)
    ch_size = (ch_size / resolution).astype(int)

    voxels_list = []
    svids_list = []

    for ch in navis.config.tqdm(
        l2_vxl, disable=not progress, leave=False, desc="Fetching voxels"
    ):
        try:
            # Fetch chunk
            cutout = vol[
                ch[0] : ch[0] + ch_size[0],
                ch[1] : ch[1] + ch_size[1],
                ch[2] : ch[2] + ch_size[2],
            ][:, :, :, 0]

            # Find voxels belonging to our supervoxels
            is_root = np.isin(cutout, sv)
            this_vxl = np.column_stack(np.where(is_root))
            this_vxl = this_vxl + ch

            if len(this_vxl) > 0:
                voxels_list.append(this_vxl.astype(np.uint32))

                if return_svids:
                    svids_list.append(cutout[is_root])
        except Exception:
            # Skip chunks that fail to download
            continue

    if len(voxels_list) == 0:
        voxels = np.array([], dtype=np.uint32).reshape(0, 3)
        svids = np.array([], dtype=np.uint64) if return_svids else None
    else:
        voxels = np.vstack(voxels_list).astype(np.uint32)
        svids = np.concatenate(svids_list) if return_svids else None

    return voxels, svids


def _get_voxels_cutout(x, bounds, resolution, mip, return_svids, progress, dataset):
    """Fetch voxels using single cutout method.

    This method downloads the entire bounding box and extracts voxels belonging
    to the target root ID. It's simpler than the L2 chunk method but can be
    memory intensive for large regions.
    """
    # Download the segmentation cutout
    # Strategy: Get supervoxels if we need them, otherwise get root IDs directly
    cutout, cutout_resolution, offset = get_segmentation_cutout(
        bounds, root_ids=not return_svids, mip=mip, coordinates="nm", dataset=dataset
    )

    if return_svids:
        # We fetched supervoxel IDs - need to find which belong to our root
        # Get unique supervoxels (excluding zeros)
        sv_unique = np.unique(cutout[cutout != 0])

        if len(sv_unique) == 0:
            # Empty cutout
            voxels = np.array([], dtype=np.uint32).reshape(0, 3)
            svids = np.array([], dtype=np.uint64)
            return voxels, svids

        # Convert supervoxels to roots to find matches
        roots_unique = supervoxels_to_roots(sv_unique, dataset=dataset, progress=False)

        # Create mask for supervoxels belonging to our root
        sv_to_keep = sv_unique[roots_unique == x]
        is_match = np.isin(cutout, sv_to_keep)
        matches = np.where(is_match)

        # Extract supervoxel IDs at matched positions
        svids = cutout[matches]
    else:
        # We fetched root IDs directly - simple matching
        matches = np.where(cutout == x)
        svids = None

    if len(matches[0]) == 0:
        # No voxels found for this root ID
        voxels = np.array([], dtype=np.uint32).reshape(0, 3)
        svids = np.array([], dtype=np.uint64) if return_svids else None
        return voxels, svids

    # Convert array indices to voxel coordinates
    voxels = np.column_stack(matches).astype(np.uint32)

    # Add the offset to get absolute voxel coordinates
    offset_voxels = (offset / resolution).astype(np.uint32)
    voxels = voxels + offset_voxels

    return voxels, svids


def _thin_voxels(x, voxels, svids, vol, progress, dataset):
    """Remove interface voxels between non-connected L2 chunks."""
    from .l2 import get_l2_graph

    try:
        from pykdtree.kdtree import KDTree
    except ImportError:
        from scipy.spatial import cKDTree as KDTree

    # Get the L2 ID for each supervoxel
    l2_ids = vol.get_roots(svids, stop_layer=2)
    l2_dict = dict(zip(svids, l2_ids))

    # Get the L2 graph
    G = get_l2_graph(x, dataset=dataset)

    # Create KD tree for all voxels
    tree = KDTree(voxels.astype(float))

    # Create a mask for invalidated voxels
    invalid = np.zeros(len(voxels), dtype=bool)

    # Process each supervoxel
    for sv in navis.config.tqdm(
        np.unique(svids), disable=not progress, desc="Thinning", leave=False
    ):
        is_this_sv = svids == sv

        if not np.any(is_this_sv):
            continue

        # Get connected L2 chunks
        is_this_l2 = l2_ids == l2_dict[sv]
        neighbors = list(G.neighbors(l2_dict[sv])) if l2_dict[sv] in G else []
        is_connected_l2 = np.isin(l2_ids, neighbors)

        # Mask to exclude connected voxels
        mask = is_this_l2 | is_connected_l2 | invalid

        # Find touching voxels
        dist, ix = tree.query(
            voxels[is_this_sv].astype(float),
            k=10,  # Check up to 10 nearest neighbors
            distance_upper_bound=1.75,
        )

        # Mark interface voxels as invalid
        is_touching = (dist < np.inf).any(axis=1)
        if np.any(is_touching):
            invalid[np.where(is_this_sv)[0][is_touching]] = True

    return voxels[~invalid], svids[~invalid]


@inject_dataset(allowed=CRANT_VALID_DATASETS)
def find_common_time(
    root_ids: IDs,
    progress: bool = True,
    *,
    dataset: Optional[str] = None,
):
    """Find a time at which given root IDs co-existed.

    Parameters
    ----------
    root_ids :      list | np.ndarray
                    Root IDs to check.
    progress :      bool
                    If True, shows progress bar.
    dataset :       str
                    The dataset to use. If not provided, uses the default dataset.

    Returns
    -------
    datetime.datetime
                    A timestamp when all root IDs existed simultaneously.

    Examples
    --------
    >>> from crantpy.utils.cave.segmentation import find_common_time
    >>> common_time = find_common_time([123456789, 987654321])
    """
    from datetime import timezone

    root_ids = np.asarray(root_ids, dtype=np.int64)

    client = get_cave_client(dataset=dataset)

    # Get timestamps when roots were created
    creations = client.chunkedgraph.get_root_timestamps(root_ids)

    # Find out which IDs are still current
    is_latest = client.chunkedgraph.is_latest_roots(root_ids)

    # Prepare array with death times
    deaths = np.array([datetime.now(tz=timezone.utc) for r in root_ids])

    # Get lineage graph for outdated root IDs
    if not is_latest.all():
        G = client.chunkedgraph.get_lineage_graph(
            root_ids[~is_latest], timestamp_past=min(creations), as_nx_graph=True
        )

        # Get the immediate successors
        succ = np.array([next(G.successors(r)) for r in root_ids[~is_latest]])

        # Add time of death
        deaths[~is_latest] = client.chunkedgraph.get_root_timestamps(succ)

    # Find the latest creation
    latest_birth = max(creations)

    # Find the earliest death
    earliest_death = min(deaths)

    if latest_birth > earliest_death:
        raise ValueError("Given root IDs never existed at the same time.")

    return latest_birth + (earliest_death - latest_birth) / 2


@inject_dataset(allowed=CRANT_VALID_DATASETS)
def snap_to_id(
    locs: np.ndarray,
    id: Union[int, np.int64],
    snap_zero: bool = False,
    search_radius: int = 160,
    coordinates: str = "nm",
    verbose: bool = True,
    *,
    dataset: Optional[str] = None,
):
    """Snap locations to the correct segmentation ID.

    This function is useful for correcting imprecise coordinate annotations
    (e.g., from manual annotation, image registration, or synapse detection)
    to ensure they map to the expected neuron/segment.

    **How it works:**
     1. Check segmentation ID at each location
     2. For locations with wrong ID: search within radius for correct ID
     3. Snap to closest voxel with correct ID

    **IMPORTANT - CloudVolume Coverage:**
    This function requires CloudVolume segmentation data at the target locations.
    Locations outside CloudVolume's spatial coverage (~360 µm cube) cannot be
    snapped and will be returned as [0, 0, 0].

    Parameters
    ----------
    locs :          (N, 3) array
                    Array of x/y/z coordinates to snap.
    id :            int
                    Expected/target segmentation ID at each location.
                    Typically a root ID of the neuron of interest.
    snap_zero :     bool
                    If False (default), we will not snap locations that map to
                    segment ID 0 (i.e., no segmentation / background).
                    Set to True to attempt snapping even for background locations.
    search_radius : int
                    Radius [nm] around a location to search for a voxel with
                    the correct ID. Larger radius = more likely to find match
                    but slower. Default 160 nm is usually sufficient for small
                    annotation errors. Increase to 500-1000 nm for larger errors.
    coordinates :   "voxel" | "nm"
                    Coordinate system of `locs`. Default "nm" (nanometers).
    verbose :       bool
                    If True, will print summary of snapping results and any
                    errors encountered.
    dataset :       str
                    The dataset to use. If not provided, uses the default dataset.

    Returns
    -------
    (N, 3) array
                    Snapped x/y/z locations guaranteed to map to the correct ID
                    (or [0, 0, 0] for locations that couldn't be snapped).

    Raises
    ------
    ValueError
        If search region exceeds CloudVolume coverage for any location.

    Examples
    --------
    >>> from crantpy.utils.cave.segmentation import snap_to_id
    >>> import numpy as np
    >>>
    >>> # Example: Fix slightly misaligned synapse annotations
    >>> synapse_locs = np.array([
    ...     [100050, 50025, 3005],  # Slightly off target
    ...     [100150, 50125, 3015],
    ... ])
    >>> target_neuron_id = 720575940621039145
    >>>
    >>> # Snap to nearest voxel on target neuron
    >>> corrected_locs = snap_to_id(
    ...     synapse_locs,
    ...     id=target_neuron_id,
    ...     search_radius=200,  # Search within 200nm
    ...     verbose=True
    ... )
    >>> # Output: 2 of 2 locations needed to be snapped.
    >>> #         Of these 0 locations could not be snapped...

    >>> # Example: Quality control for traced neuron nodes
    >>> import navis
    >>> neuron = navis.TreeNeuron(...)  # Your neuron reconstruction
    >>> expected_root_id = 720575940621039145
    >>>
    >>> # Snap all nodes to ensure they're on the correct segment
    >>> corrected_nodes = snap_to_id(
    ...     neuron.nodes[['x', 'y', 'z']].values,
    ...     id=expected_root_id,
    ...     search_radius=500,
    ...     coordinates="nm"
    ... )
    >>>
    >>> # Update neuron with corrected coordinates
    >>> neuron.nodes[['x', 'y', 'z']] = corrected_nodes

    >>> # Example: Handle locations in background (ID 0)
    >>> locs_with_background = np.array([
    ...     [100000, 50000, 3000],  # On neuron
    ...     [999999, 999999, 9999],  # In background (ID 0)
    ... ])
    >>>
    >>> # By default, won't try to snap background locations
    >>> snapped = snap_to_id(locs_with_background, id=target_neuron_id)
    >>> # Background location will remain unchanged
    >>>
    >>> # Force snapping even for background (use with caution!)
    >>> snapped = snap_to_id(
    ...     locs_with_background,
    ...     id=target_neuron_id,
    ...     snap_zero=True,  # Try to snap background too
    ...     search_radius=1000  # Larger search needed
    ... )

    Notes
    -----
    **When to use this function:**
    - Synapse annotation QC: Ensure synapses are on correct pre/postsynaptic neurons
    - Image registration errors: Fix coordinate misalignment after registration
    - Manual annotation cleanup: Correct imprecise manual annotations
    - Traced neuron validation: Ensure skeleton nodes are on correct segment

    **Performance considerations:**
    - Each location requiring snapping fetches a small segmentation cutout
    - Larger `search_radius` = slower (more data to fetch and search)
    - Locations already on correct ID are very fast (no cutout needed)
    - For many locations, consider parallelizing or batching

    **Common issues:**
    - **"No voxels found in search region"**: The target ID doesn't exist
      within `search_radius`. Try increasing search_radius or verify the
      expected ID is correct.
    - **"Bounds exceed CloudVolume coverage"**: Location is outside the
      ~360 µm region covered by CloudVolume. These locations cannot be snapped.
    - **Many failures**: Check if your locations and target ID are in the
      same coordinate space and if the neuron actually exists at those locations.

    See Also
    --------
    locs_to_segments : Check which segment IDs are at given locations
    get_segmentation_cutout : Get segmentation in a region
    """
    from scipy import ndimage

    assert coordinates in ["nm", "nanometer", "nanometers", "voxel", "voxels"]

    if isinstance(locs, navis.TreeNeuron):
        locs = locs.nodes[["x", "y", "z"]].values

    # This also makes sure we work on a copy
    locs = np.array(locs, copy=True)
    assert locs.ndim == 2 and locs.shape[1] == 3

    # Get resolution info
    client = get_cave_client(dataset=dataset)
    res = np.array(
        [
            client.info.get_datastack_info()["viewer_resolution_x"],
            client.info.get_datastack_info()["viewer_resolution_y"],
            client.info.get_datastack_info()["viewer_resolution_z"],
        ]
    )

    # From hereon out we are working with nanometers
    if coordinates in ("voxel", "voxels"):
        locs *= res

    root_ids = locs_to_segments(locs, dataset=dataset, coordinates="nm")

    id_wrong = root_ids != id
    not_zero = root_ids != 0

    to_fix = id_wrong

    if not snap_zero:
        to_fix = to_fix & not_zero

    # If nothing needs fixing, return early
    if not np.any(to_fix):
        if verbose:
            print("All locations already on the correct segment!")
        return np.asarray(locs, dtype=float)

    # Process each location that needs fixing
    new_locs = []
    for loc in locs[to_fix]:
        loc = np.round(np.asarray(loc, dtype=float))

        # Generating bounding box around this location
        mn = loc - search_radius
        mx = loc + search_radius
        # Make sure it's a multiple of the resolution
        mn = mn - mn % res
        mx = mx - mx % res

        # Generate bounding box
        bbox = np.vstack((mn, mx))

        try:
            # Get the cutout
            cutout, _, offset_nm = get_segmentation_cutout(
                bbox, dataset=dataset, root_ids=True, coordinates="nm"
            )

            # Generate a mask
            mask = (cutout == id).astype(int, copy=False)

            # Erode so we move our point slightly more inside the segmentation
            mask = ndimage.binary_erosion(mask).astype(mask.dtype)

            # Find positions with the ID we are looking for
            our_id = np.vstack(np.where(mask)).T

            # Return [0, 0, 0] if unable to snap
            if not our_id.size:
                if verbose:
                    print(
                        f"    No voxels with ID {id} found in search region around {loc}"
                    )
                new_locs.append(np.array([0, 0, 0]))
                continue

            # Get the closest one to the center of the cutout
            center = np.divide(cutout.shape, 2).round()
            dist = np.abs(our_id - center).sum(axis=1)
            closest = our_id[np.argmin(dist)]

            # Convert the cutout offset to absolute coordinates
            resolution = get_cloudvolume(dataset).scale["resolution"]
            snapped = closest * resolution + offset_nm

            new_locs.append(snapped)
        except Exception as e:
            # If anything fails, return [0, 0, 0]
            if verbose:
                print(f"    Failed to snap location {loc}: {e}")
            new_locs.append(np.array([0, 0, 0]))

    # Stack locations
    new_locs = np.vstack(new_locs)

    # If no new location found, array will be [0, 0, 0]
    not_snapped = new_locs.max(axis=1) == 0

    # Update location
    to_update = np.where(to_fix)[0][~not_snapped]
    locs[to_update, :] = new_locs[~not_snapped]

    if verbose:
        import textwrap

        msg = f"""\
        {to_fix.sum()} of {to_fix.shape[0]} locations needed to be snapped.
        Of these {not_snapped.sum()} locations could not be snapped - consider
        increasing `search_radius`.
        """
        print(textwrap.dedent(msg))

    return np.asarray(locs, dtype=float)
