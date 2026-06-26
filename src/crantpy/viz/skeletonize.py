# -*- coding: utf-8 -*-
"""Neuron skeletonization tools.

This module offers optional suppression of urllib3 connection pool warnings
that can appear when downloading mesh data during skeletonization. These specific
messages are cosmetic and do not affect functionality, but can be noisy. Suppression
is opt-in via a function call or the environment variable
"CRANTPY_SUPPRESS_URLLIB3_WARNINGS".
"""

import os
import warnings
import urllib3
import numpy as np

import logging
import pandas as pd
import navis
import pcg_skel
import skeletor as sk
import trimesh
import multiprocessing as mp
import time
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from numpy.typing import NDArray
from tqdm import tqdm
from scipy.spatial import cKDTree

from caveclient import CAVEclient
from ..utils.decorators import parse_neuroncriteria, inject_dataset
from ..utils.cave import get_cave_client as create_client
import matplotlib.pyplot as plt

# Configure matplotlib for proper PDF export
plt.rcParams["pdf.fonttype"] = 42
import matplotlib.colors as mcolors
import requests

# Module logger
logger = logging.getLogger(__name__)


# --- Optional urllib3 warning suppression utilities ---
def configure_urllib3_warning_suppression(enable: Optional[bool] = None) -> bool:
    """Enable suppression of known cosmetic urllib3 warnings.

    Trade-offs: Hiding warnings can make it harder to notice real connectivity
    issues. When enabled, only the specific connection pool message is filtered;
    other warnings remain visible. Does not call urllib3.disable_warnings().

    Control via `enable` or environment variable `CRANTPY_SUPPRESS_URLLIB3_WARNINGS`.

    Returns True if suppression is enabled, False otherwise.
    """
    if enable is None:
        env = os.getenv("CRANTPY_SUPPRESS_URLLIB3_WARNINGS", "").strip().lower()
        enable = env in {"1", "true", "yes", "on"}

    # Clear any prior filters for this exact message to avoid duplicates
    if enable:
        try:
            warnings.filterwarnings(
                "ignore",
                message=r".*Connection pool is full, discarding connection.*",
                module=r"urllib3[.]connectionpool",
            )
            # Keep logging in default state; do not blanket silence urllib3
            logger.debug("Enabled urllib3 connection pool warning suppression")
        except Exception:
            # Non-fatal: log with traceback for debugging but do not raise
            logger.exception("Failed to configure urllib3 warning suppression")
            return False
        return True
    return False


from contextlib import contextmanager


@contextmanager
def suppress_urllib3_connectionpool_warnings():
    """Context manager to temporarily suppress urllib3 connectionpool messages.

    Only filters the specific "Connection pool is full, discarding connection"
    warnings while inside the context.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*Connection pool is full, discarding connection.*",
            module=r"urllib3[.]connectionpool",
        )
        yield


# Apply suppression if requested via environment variable
configure_urllib3_warning_suppression()

SKELETON_INFO = {
    "@type": "neuroglancer_skeletons",
    "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    "vertex_attributes": [
        {"id": "radius", "data_type": "float32", "num_components": 1}
    ],
}  # from navis

__all__ = [
    "skeletonize_neuron",
    "skeletonize_neurons_parallel",
    "get_skeletons",
    "chunks_to_nm",
    "detect_soma_skeleton",
    "detect_soma_mesh",
    "get_soma_from_annotations",
    "configure_urllib3_warning_suppression",
    "suppress_urllib3_connectionpool_warnings",
    "_create_node_info_dict",
    "_swc_dict_to_dataframe",
    "_preprocess_mesh",
    "_shave_skeleton",
    "_worker_wrapper",
    "_remove_soma_hairball",
    "divide_local_neighbourhood",
]


@parse_neuroncriteria()
@inject_dataset()
def skeletonize_neuron(
    client: CAVEclient,
    root_id: Union[int, List[int], NDArray],
    shave_skeleton: bool = True,
    remove_soma_hairball: bool = False,
    assert_id_match: bool = False,
    threads: int = 2,
    save_to: Optional[str] = None,
    progress: bool = True,
    use_pcg_skel: bool = False,
    dataset: Optional[str] = None,
    **kwargs: Any,
) -> Union[navis.TreeNeuron, navis.NeuronList]:
    """Skeletonize a neuron the main function.

    Parameters
    ----------
    client : CAVEclient
        CAVE client for data access.
    root_id : int
        Root ID of the neuron to skeletonize.
    dataset : str, optional
        Dataset to fetch meshes from in the skeletor fallback. If not given,
        the ``@inject_dataset`` decorator fills in the default dataset. Should
        match the dataset ``client`` was built for.
    shave_skeleton : bool, default True
        Remove small protrusions and bristles from skeleton (from my understanding).
    remove_soma_hairball : bool, default False
        Remove the hairball mesh from the soma
    assert_id_match : bool, default False
        Verify skeleton nodes map to correct segment ID.
    threads : int, default 2
        Number of parallel threads for mesh processing.
    save_to : str, optional
        Save skeleton as SWC file to this path.
    progress : bool, default True
        Show progress bars during processing.
    use_pcg_skel : bool, default False
        Try pcg_skel first before skeletor (CAVE-client skeletonization).
    **kwargs
        Additional arguments for skeletonization algorithms.

    Returns
    -------
    navis.TreeNeuron
        The skeletonized neuron.
    # TODOs from fafbseg:
    # - Use synapse locations as constraints
    # - Mesh preprocessing options
    # - Chunked skeletonization for large meshes
    # - Use soma annotations from external sources
    # - Better error handling/logging
    # - Allow user-supplied soma location/radius
    # - Option to return intermediate results
    # - Support more skeletonization algorithms
    # - Merge disconnected skeletons
    # - Custom node/edge attributes
    """
    if navis.utils.is_iterable(root_id):
        root_id_array = np.asarray(root_id).astype(np.int64)

        # For multiple neurons, ``save_to`` is treated as an output directory and
        # each neuron is written to "<root_id>.swc" so they don't overwrite one
        # another (the single-neuron branch below treats it as a file path).
        if save_to is not None:
            save_dir = os.path.abspath(save_to)
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = None

        return navis.NeuronList(
            [
                skeletonize_neuron(
                    client,
                    int(rid),
                    progress=False,
                    shave_skeleton=shave_skeleton,
                    remove_soma_hairball=remove_soma_hairball,
                    assert_id_match=assert_id_match,
                    threads=threads,
                    save_to=(
                        os.path.join(save_dir, f"{int(rid)}.swc") if save_dir else None
                    ),
                    use_pcg_skel=use_pcg_skel,
                    dataset=dataset,
                    **kwargs,
                )
                for rid in tqdm(
                    root_id_array,
                    desc="Skeletonizing",
                    disable=not progress,
                    leave=False,
                )
            ]
        )

    if save_to is not None:
        save_to = os.path.abspath(save_to)
        os.makedirs(os.path.dirname(save_to), exist_ok=True)

    assert isinstance(
        root_id, (int, np.integer)
    ), "root_id must be an integer for single neuron"
    root_id_int = int(root_id)

    if use_pcg_skel:  # try the CAVE client's pcg_skel first
        try:
            skel = pcg_skel.pcg_skeleton(root_id=root_id_int, client=client)
            try:
                if hasattr(skel, "vertices") and hasattr(skel, "edges"):
                    vertices = skel.vertices  # type: ignore
                    edges = skel.edges  # type: ignore
                else:
                    if isinstance(skel, tuple) and len(skel) > 0:
                        skel = skel[0]  # type: ignore
                    vertices = skel.vertices  # type: ignore
                    edges = skel.edges  # type: ignore
            except (AttributeError, IndexError, KeyError, TypeError) as e:
                raise ValueError(f"Failed to extract skeleton data from pcg_skel: {e}")

            node_info = _create_node_info_dict(vertices, edges)
            df = _swc_dict_to_dataframe(node_info)

            tn = navis.TreeNeuron(
                df, id=root_id_int, units="1 nm"
            )  # navis treeneuron object

            if shave_skeleton:
                _shave_skeleton(tn)

            _apply_soma_processing(tn, root_id_int, client, remove_soma_hairball)

            if assert_id_match:
                _assert_id_match(tn, root_id_int, client)

            if save_to:
                tn.to_swc(save_to)

            return tn

        except (
            requests.HTTPError,
            requests.ConnectionError,
            requests.Timeout,
            ValueError,
        ) as e:
            logger.warning(
                "pcg_skel failed for %s: %s. Falling back to skeletor.",
                root_id_int,
                e,
            )
        except Exception as e:
            # Unexpected failure: log full traceback, then fall back
            logger.exception(
                "Unexpected error in pcg_skel for %s. Falling back to skeletor.",
                root_id_int,
            )

    try:
        from ..utils.cave import get_cloudvolume

        # Use the same dataset the client was built for so the mesh is fetched
        # from the matching segmentation source (dataset is injected by the
        # @inject_dataset decorator when not supplied).
        vol = get_cloudvolume(dataset=dataset)
        mesh_dict = vol.mesh.get(root_id_int) if hasattr(vol, "mesh") else vol.get_mesh(root_id_int)  # type: ignore

        if isinstance(mesh_dict, dict) and root_id_int in mesh_dict:
            mesh = mesh_dict[root_id_int]
        else:
            mesh = mesh_dict

        if not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)  # type: ignore

        mesh = _preprocess_mesh(mesh, **kwargs)

        defaults = {"waves": 1, "step_size": 1}

        skeletor_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            not in (
                "dataset",
                "lod",
                "assert_id_match",
                "shave_skeleton",
                "remove_soma_hairball",
                "save_to",
                "use_pcg_skel",
                "_soma_prefetched",
                "threads",
            )
        }
        defaults.update(skeletor_kwargs)
        s = sk.skeletonize.by_wavefront(mesh, progress=progress, **defaults)  # type: ignore

        if hasattr(s, "swc"):
            swc_data = s.swc  # type: ignore
        else:
            swc_data = s[0] if isinstance(s, tuple) else s  # type: ignore

        swc_data["node_id"] += 1  # type: ignore
        swc_data.loc[swc_data.parent_id >= 0, "parent_id"] += 1  # type: ignore

        swc_data["radius"] = swc_data.radius.round().astype(int)  # type: ignore

        tn = navis.TreeNeuron(swc_data, units="1 nm", id=root_id_int, soma=None)

    except (ValueError, RuntimeError, requests.RequestException) as e:
        # Known/expected error types: raise with concise message
        raise ValueError(f"Failed to skeletonize neuron {root_id_int}: {e}")
    except Exception:
        # Unexpected: log full traceback to aid debugging, then raise
        logger.exception("Unexpected error while skeletonizing neuron %s", root_id_int)
        raise

    if shave_skeleton:
        _shave_skeleton(tn)

    _apply_soma_processing(tn, root_id_int, client, remove_soma_hairball)

    if assert_id_match:
        _assert_id_match(tn, root_id_int, client)

    if save_to:
        tn.to_swc(save_to)

    return tn


@parse_neuroncriteria()
@inject_dataset()
def skeletonize_neurons_parallel(
    client: CAVEclient,
    root_ids: Union[List[int], NDArray],
    n_cores: Optional[int] = None,
    progress: bool = True,
    color_map: Optional[str] = None,
    dataset: Optional[str] = None,
    **kwargs: Any,
) -> Union[navis.NeuronList, Tuple[navis.NeuronList, List[str]]]:
    """Skeletonize multiple neurons in parallel.

    Parameters
    ----------
    client : CAVEclient
        CAVE client for data access.
    root_ids : list of int or np.ndarray
        Root IDs of neurons to skeletonize.
    n_cores : int, optional
        Number of cores to use. If None, uses half of available cores.
    progress : bool, default True
        Show progress bars during processing.
    color_map : str, optional
        Generate colors for each neuron using this colormap.
        Returns tuple of (neurons, colors) instead of just neurons.
    **kwargs
        Additional arguments passed to skeletonize_neuron.

    Returns
    -------
    navis.NeuronList or tuple
        NeuronList of skeletonized neurons, or tuple of (NeuronList, colors)
        if color_map is specified.
    """
    if n_cores is not None:
        if n_cores < 1:
            raise ValueError("n_cores must be at least 1")
        if n_cores > mp.cpu_count():
            raise ValueError(
                f"n_cores cannot exceed {mp.cpu_count()} (available cores)"
            )
    else:
        n_cores = max(1, mp.cpu_count() // 2)

    root_ids = np.asarray(root_ids, dtype=np.int64)

    import inspect

    sig = inspect.signature(skeletonize_neuron)
    for k in kwargs:
        if k not in sig.parameters and k not in ("lod",):
            raise ValueError(f"unexpected keyword argument for skeletonize_neuron: {k}")

    kwargs["progress"] = False
    kwargs["threads"] = 1
    kwargs["dataset"] = dataset
    kwargs["_soma_prefetched"] = False

    funcs = [skeletonize_neuron] * len(root_ids)
    args_list = [[client, root_id] for root_id in root_ids]
    kwargs_list = [dict(kwargs) for _ in root_ids]
    combinations = list(zip(funcs, args_list, kwargs_list))

    results = []
    with mp.Pool(n_cores) as pool:
        chunksize = 1

        iterator = pool.imap(_worker_wrapper, combinations, chunksize=chunksize)

        for result in tqdm(
            iterator,
            total=len(combinations),
            desc="Skeletonizing",
            disable=not progress,
            leave=True,
        ):
            results.append(result)

    # Separate successes and failures (structured error dicts)
    error_entries: List[Dict[str, Any]] = [
        r for r in results if isinstance(r, dict) and r.get("status") == "error"
    ]
    if error_entries:
        failed_ids = [str(e.get("root_id")) for e in error_entries]
        error_types: Dict[str, int] = {}
        for e in error_entries:
            et = str(e.get("error_type", "unknown"))
            error_types[et] = error_types.get(et, 0) + 1
        warnings.warn(
            f"{len(error_entries)} neurons failed to skeletonize: "
            + ", ".join(failed_ids[:10])
            + ("..." if len(error_entries) > 10 else "")
        )
        warnings.warn(f"Error breakdown: {error_types}")

    neurons = navis.NeuronList([r for r in results if isinstance(r, navis.TreeNeuron)])

    if color_map is not None:
        try:
            # Generate one color per surviving neuron (failures are dropped from
            # `neurons`) so colors and neurons stay aligned by position.
            n = len(neurons)
            cmap = plt.get_cmap(color_map, max(n, 1))
            colors = [mcolors.to_hex(cmap(i)) for i in range(n)]
            return neurons, colors
        except Exception as e:
            warnings.warn(f"Failed to generate colors: {e}")
            return neurons
    else:
        return neurons


def _assert_id_match(tn: navis.TreeNeuron, root_id: int, client: CAVEclient) -> None:
    """Verify that skeleton nodes map to the correct segment ID.

    TODO: ID matching is not implemented yet. This function currently acts as
    a placeholder and will be implemented once the CAVEclient API usage is
    finalized. Calls to this function log a warning to make it clear that the
    `assert_id_match` option is currently a no-op.
    """
    if root_id == 0:
        raise ValueError("Segmentation ID must not be 0")

    warnings.warn(
        "assert_id_match is not implemented yet and currently has no effect.",
        category=UserWarning,
    )


def _worker_wrapper(
    x: Tuple[Callable, List[Any], Dict[str, Any]],
) -> Union[navis.TreeNeuron, str]:
    """worker wrapper (from fafbseg) with error handling and retry logic.

    Parameters
    ----------
    x : tuple
        Tuple of (function, arguments, kwargs) for worker execution.

    Returns
    -------
    navis.TreeNeuron or dict
        Successfully skeletonized neuron or a structured error dict:
        {"status": "error", "root_id": int, "error_type": str, "message": str}
    """
    f, args, kwargs = x
    root_id = args[1] if len(args) > 1 else "unknown"

    try:
        result = f(*args, **kwargs)
        if result is None:
            return {
                "status": "error",
                "root_id": root_id,
                "error_type": "skeletonization_failed",
                "message": "Function returned None",
            }
        return result
    except KeyboardInterrupt:
        raise
    except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
        try:
            # Brief backoff before the single retry: these errors are usually
            # transient (rate limiting / overloaded server), so an immediate
            # retry tends to just fail the same way.
            time.sleep(2)
            result = f(*args, **kwargs)
            if result is None:
                return {
                    "status": "error",
                    "root_id": root_id,
                    "error_type": "skeletonization_failed_after_retry",
                    "message": "Function returned None after retry",
                }
            return result
        except KeyboardInterrupt:
            raise
        except Exception as retry_error:
            logger.warning(
                "Network error for neuron %s: %s, retry failed: %s",
                root_id,
                e,
                retry_error,
            )
            return {
                "status": "error",
                "root_id": root_id,
                "error_type": type(e).__name__,
                "message": f"Network error: {e}. Retry failed: {retry_error}",
            }
    except ValueError as e:
        warnings.warn(f"Validation error for neuron {root_id}: {e}")
        return {
            "status": "error",
            "root_id": root_id,
            "error_type": "validation_error",
            "message": str(e),
        }
    except Exception as e:
        logger.exception(
            "Failed to skeletonize neuron %s: %s: %s", root_id, type(e).__name__, e
        )
        return {
            "status": "error",
            "root_id": root_id,
            "error_type": type(e).__name__,
            "message": str(e),
        }


def _create_node_info_dict(
    vertices: NDArray, edges: NDArray
) -> Dict[int, Dict[str, Any]]:
    """Create node info dictionary for SWC format.

    Edge orientation from the various skeleton sources is not guaranteed to be
    ``[parent, child]`` -- meshparty (the pcg_skel path) emits ``[child, parent]``
    and precomputed CAVE skeleton edges may be unordered. Parent pointers are
    therefore derived from a breadth-first traversal of the *undirected*
    connectivity rather than trusting the edge direction. This guarantees
    exactly one parent per node and one root per connected component, and avoids
    silently inverting roots or dropping the children of branch points.

    Absent any soma information the root of each component is the lowest vertex
    index (deterministic but arbitrary); callers that need a somatic root should
    reroot afterwards (see ``_apply_soma_processing``).
    """
    node_info: Dict[int, Dict[str, Any]] = {}

    for i, coord in enumerate(vertices):
        node_info[i] = {
            "PointNo": i + 1,
            "Type": 3,
            "X": float(coord[0]),
            "Y": float(coord[1]),
            "Z": float(coord[2]),
            "Radius": 1.0,
            "Parent": -1,
        }

    # Build undirected adjacency from the edge list.
    adjacency: Dict[int, set] = collections.defaultdict(set)
    for edge in edges:
        a, b = int(edge[0]), int(edge[1])
        adjacency[a].add(b)
        adjacency[b].add(a)

    # Derive a rooted forest via BFS: every node gets a single parent and each
    # connected component gets exactly one root (lowest index, deterministic).
    parent_map: Dict[int, int] = {}
    seen: set = set()
    for start in node_info:
        if start in seen:
            continue
        seen.add(start)
        queue = collections.deque([start])
        while queue:
            cur = queue.popleft()
            for neighbor in sorted(adjacency[cur]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    parent_map[neighbor] = cur
                    queue.append(neighbor)

    for child, parent in parent_map.items():
        node_info[child]["Parent"] = parent + 1  # SWC node ids are 1-indexed

    # Classify node types from undirected degree: roots (1), leaves/endpoints (6),
    # everything else intermediate/branch (3).
    root_nodes = set(node_info.keys()) - set(parent_map.keys())
    for node_idx, info in node_info.items():
        if node_idx in root_nodes:
            info["Type"] = 1
            info["Parent"] = -1
        elif len(adjacency[node_idx]) <= 1:
            info["Type"] = 6
        else:
            info["Type"] = 3

    return node_info


def _swc_dict_to_dataframe(node_info: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    """Convert node info dictionary to SWC DataFrame."""
    df = pd.DataFrame.from_dict(node_info, orient="index")
    df = df[["PointNo", "Type", "X", "Y", "Z", "Radius", "Parent"]]
    df = df.sort_values("PointNo")
    for col in ["X", "Y", "Z", "Radius"]:
        df[col] = df[col].astype(float)
    return df


def detect_soma_skeleton(
    s: navis.TreeNeuron, min_rad: int = 800, N: int = 3
) -> Optional[int]:
    """Try detecting the soma based on radii.

    Looks for consecutive nodes with large radii to identify soma. Includes
    additional checks to ensure the skeleton is valid.

    Parameters
    ----------
    s : navis.TreeNeuron
        The skeleton to analyze for soma detection.
    min_rad : int, default 800
        Minimum radius for a node to be considered a soma candidate (in nm).
    N : int, default 3
        Number of consecutive nodes with radius > `min_rad` needed
        to consider them soma candidates.

    Returns
    -------
    int or None
        Node ID of the detected soma, or None if no soma found.
    """
    assert isinstance(s, navis.TreeNeuron), "Input must be a navis.TreeNeuron"

    if not hasattr(s, "nodes") or s.nodes is None or len(s.nodes) == 0:
        warnings.warn("Skeleton has no nodes for soma detection")
        return None

    if "radius" not in s.nodes.columns:
        warnings.warn("Skeleton nodes missing radius column for soma detection")
        return None

    try:
        segments = s.segments
        if not segments or len(segments) == 0:
            warnings.warn("Skeleton has no segments for soma detection")
            return None
    except Exception as e:
        warnings.warn(f"Failed to get skeleton segments: {e}")
        return None

    try:
        radii = s.nodes.set_index("node_id").radius.to_dict()
    except Exception as e:
        warnings.warn(f"Failed to extract radii from skeleton: {e}")
        return None

    candidates = []

    for seg in s.segments:
        rad = np.array([radii[node_id] for node_id in seg])
        is_big = np.where(rad > min_rad)[0]

        # `is_big` holds the positions of large-radius nodes; an empty array
        # means none qualify. (Using ``any(is_big)`` here would wrongly skip a
        # segment whose only large node sits at position 0.)
        if is_big.size == 0:
            continue

        for stretch in np.split(is_big, np.where(np.diff(is_big) != 1)[0] + 1):
            if len(stretch) < N:
                continue
            candidates += [seg[i] for i in stretch]

    if not candidates:
        return None

    return sorted(candidates, key=lambda x: radii[x])[-1]


def detect_soma_mesh(mesh: trimesh.Trimesh) -> NDArray:
    """Try detecting the soma based on vertex clusters.

    Identifies dense vertex clusters that likely represent the soma.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Coordinates in nanometers. Mesh must not be downsampled for
        accurate detection.

    Returns
    -------
    np.ndarray
        Array of vertex indices that belong to the detected soma region.
        Returns empty array if no soma is detected.
    """
    if mesh is None:
        warnings.warn("Cannot detect soma on None mesh")
        return np.array([])

    if not hasattr(mesh, "vertices") or mesh.vertices is None:
        warnings.warn("Mesh has no vertices for soma detection")
        return np.array([])

    if len(mesh.vertices) == 0:
        warnings.warn("Mesh has no vertices for soma detection")
        return np.array([])

    if len(mesh.vertices) < 100:
        warnings.warn(
            f"Mesh has too few vertices ({len(mesh.vertices)}) for reliable soma detection"
        )
        return np.array([])

    try:
        tree = cKDTree(mesh.vertices)
    except Exception as e:
        warnings.warn(f"Failed to build KDTree for soma detection: {e}")
        return np.array([])

    # Count neighbours within 4 µm of each vertex in a single vectorized call;
    # dense clusters mark the soma. A per-vertex ``tree.query(v, k=n)`` here
    # would be O(n^2) and pathological on real meshes (tens of thousands of
    # vertices).
    n_neighbors = tree.query_ball_point(mesh.vertices, r=4000, return_length=True)

    seed = np.argmax(n_neighbors)

    res = np.mean(mesh.area_faces)
    if n_neighbors.max() < (20e4 / res):
        return np.array([])

    dist, ix = tree.query(
        mesh.vertices[[seed]],
        k=mesh.vertices.shape[0],
        distance_upper_bound=10000,
    )
    soma_verts = ix[dist < float("inf")]

    return soma_verts


def get_soma_from_annotations(
    root_id: int, client: CAVEclient, dataset: Optional[str] = None
) -> Optional[Tuple[float, float, float]]:
    """Try to get soma location from nucleus annotations.

    Parameters
    ----------
    root_id : int
        Root ID of the neuron to get soma information for.
    client : CAVEclient
        CAVE client for data access.
    dataset : str, optional
        Dataset identifier (handled by decorators if not provided).

    Returns
    -------
    tuple or None
        (x, y, z) coordinates of the soma in nanometers, or None if not found.
    """
    try:
        return None  # TODO: implement when annotation source is available

    except Exception as e:
        warnings.warn(f"Failed to fetch soma annotations for {root_id}: {e}")
        return None


def get_skeletons(
    root_ids: Union[int, List[int], NDArray],
    dataset: str = "latest",
    progress: bool = True,
    omit_failures: Optional[bool] = None,
    max_threads: int = 6,
    **kwargs: Any,
) -> navis.NeuronList:
    """Fetch skeletons for multiple neurons.

    Tries to get precomputed skeletons first, then falls back to on-demand
    skeletonization if needed. Multiple root IDs are fetched concurrently using
    a thread pool (see ``max_threads``).

    Parameters
    ----------
    root_ids : list of int or np.ndarray
        Root IDs of neurons to fetch skeletons for.
    dataset : str, default 'latest'
        Dataset to query against.
    progress : bool, default True
        Show progress during fetching.
    omit_failures : bool, optional
        None: raise exception on failures
        True: skip failed neurons
        False: return empty TreeNeuron for failed cases
    max_threads : int, default 6
        Number of parallel threads for fetching skeletons.
    **kwargs
        Additional arguments passed to skeletonization if needed.

    Returns
    -------
    navis.NeuronList
        List of successfully fetched/generated skeletons.
    """
    if omit_failures not in (None, True, False):
        raise ValueError(
            "`omit_failures` must be either None, True or False. "
            f'Got "{omit_failures}".'
        )

    # Normalize root_ids to a Python list of ints for downstream handling
    if isinstance(root_ids, (int, np.integer)):
        root_ids_list = [int(root_ids)]
    elif isinstance(root_ids, np.ndarray):
        root_ids_list = [int(x) for x in root_ids.tolist()]
    else:
        root_ids_list = [int(x) for x in root_ids]

    # Drop duplicates (preserving first-seen order) so we don't fetch the same
    # neuron twice and so the final id-based reindexing stays unambiguous.
    root_ids = pd.unique(np.asarray(root_ids_list, dtype=np.int64))

    client = create_client(dataset=dataset)

    skeletons = []

    def fetch_single_skeleton(root_id: int) -> Optional[navis.TreeNeuron]:
        """Fetch single skeleton with fallback strategies."""
        try:
            try:
                skel = client.skeleton.get_skeleton(root_id, output_format="dict")
                if skel is not None:
                    vertices = np.array(skel["vertices"], dtype=float)
                    edges = np.array(skel["edges"], dtype=int)

                    node_info = _create_node_info_dict(vertices, edges)
                    df = _swc_dict_to_dataframe(node_info)

                    tn = navis.TreeNeuron(df, id=root_id, units="1 nm")
                    return tn
            except Exception as e:
                logger.debug(
                    "Precomputed skeleton fetch/parse failed for %s (%s: %s); "
                    "falling back to on-demand skeletonization",
                    root_id,
                    type(e).__name__,
                    e,
                )

            tn = skeletonize_neuron(
                client, root_id, progress=False, dataset=dataset, **kwargs
            )
            return tn

        except Exception as e:
            if omit_failures is None:
                raise ValueError(f"Failed to fetch skeleton for {root_id}: {e}")
            elif omit_failures:
                return None
            else:
                try:
                    df = pd.DataFrame(
                        {
                            "node_id": [1],
                            "parent_id": [-1],
                            "x": [0.0],
                            "y": [0.0],
                            "z": [0.0],
                            "radius": [1.0],
                        }
                    )
                    return navis.TreeNeuron(df, id=root_id, units="1 nm")
                except Exception:
                    return None

    if max_threads > 1 and len(root_ids) > 1:
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            future_to_rid = {
                executor.submit(fetch_single_skeleton, rid): rid for rid in root_ids
            }

            results = []
            for future in tqdm(
                as_completed(future_to_rid),
                desc="Fetching skeletons",
                total=len(root_ids),
                disable=not progress or len(root_ids) == 1,
                leave=False,
            ):
                rid = future_to_rid[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    if omit_failures is None:
                        raise
                    warnings.warn(f"Failed to fetch skeleton for {rid}: {e}")

            skeletons = results
    else:
        for root_id in tqdm(
            root_ids,
            desc="Fetching skeletons",
            disable=not progress or len(root_ids) == 1,
            leave=False,
        ):
            result = fetch_single_skeleton(root_id)
            if result is not None:
                skeletons.append(result)

    nl = navis.NeuronList(skeletons)

    if len(nl) > 0:
        available_ids = root_ids[np.isin(root_ids, nl.id)]
        if len(available_ids) > 0:
            nl = nl.idx[available_ids]

    return nl


def chunks_to_nm(xyz_ch, vol, voxel_resolution=[4, 4, 40]):
    """Map a chunk location to Euclidean space. CV workaround Implemented here after Giacomo's suggestion

    Parameters
    ----------
    xyz_ch : array-like
        (N, 3) array of chunk indices.
    vol : cloudvolume.CloudVolume
        CloudVolume object associated with the chunked space.
    voxel_resolution : list, optional
        Voxel resolution.

    Returns
    -------
    np.array
        (N, 3) array of spatial points.
    """
    mip_scaling = vol.mip_resolution(0) // np.array(voxel_resolution, dtype=int)

    x_vox = np.atleast_2d(xyz_ch) * vol.mesh.meta.meta.graph_chunk_size
    return (
        (x_vox + np.array(vol.mesh.meta.meta.voxel_offset(0)))
        * voxel_resolution
        * mip_scaling
    )


def _preprocess_mesh(mesh: trimesh.Trimesh, **kwargs) -> trimesh.Trimesh:
    """Apply mesh preprocessing pipeline.

    Prepares meshes for high-quality skeletonization by validating structure,
    removing small disconnected components, and fixing common mesh issues.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh to preprocess.
    **kwargs
        Additional parameters (e.g., lod for level of detail).

    Returns
    -------
    trimesh.Trimesh
        Preprocessed mesh ready for skeletonization.

    Raises
    ------
    ValueError
        If mesh is empty or has insufficient vertices/faces for skeletonization.
    """
    if mesh is None:
        raise ValueError("Cannot preprocess None mesh")

    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        raise ValueError("Mesh must have vertices and faces attributes")

    if len(mesh.vertices) == 0:
        raise ValueError("Cannot skeletonize mesh with no vertices")

    if len(mesh.faces) == 0:
        raise ValueError("Cannot skeletonize mesh with no faces")

    if len(mesh.vertices) < 4:
        raise ValueError(
            f"Mesh has too few vertices ({len(mesh.vertices)}) for skeletonization (minimum 4)"
        )

    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, validate=True)

    min_component_size = int(0.0001 * mesh.vertices.shape[0])
    remove_disconnected = min_component_size > 0

    try:
        fixed_mesh = sk.pre.fix_mesh(
            mesh, inplace=True, remove_disconnected=remove_disconnected
        )
        if fixed_mesh is not None:
            mesh = fixed_mesh if isinstance(fixed_mesh, trimesh.Trimesh) else mesh
    except Exception as e:
        raise ValueError(f"Mesh preprocessing failed: {e}")

    if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        raise ValueError("Mesh preprocessing resulted in empty mesh")

    return mesh


def _shave_skeleton(tn: navis.TreeNeuron) -> None:
    """Apply  skeleton cleanup.


    Parameters
    ----------
    tn : navis.TreeNeuron
        Skeleton to clean up (modified in place).
    """
    if tn is None:
        raise ValueError("Cannot shave None skeleton")

    if not hasattr(tn, "nodes") or tn.nodes is None:
        raise ValueError("Skeleton has no nodes to shave")

    if len(tn.nodes) == 0:
        warnings.warn("Skeleton is empty - skipping shaving")
        return

    if len(tn.nodes) < 2:
        warnings.warn(f"Skeleton has too few nodes ({len(tn.nodes)}) for shaving")
        return

    try:
        d = navis.morpho.mmetrics.parent_dist(tn, root_dist=0)
    except Exception as e:
        warnings.warn(f"Failed to compute parent distances: {e} - skipping shaving")
        return

    long = tn.nodes[d >= 1000].node_id.values

    while True:
        if len(tn.nodes) < 2:
            warnings.warn("Skeleton reduced to single node during shaving - stopping")
            break

        try:
            leaf_segs = [
                seg for seg in tn.small_segments if seg[0] in tn.leafs.node_id.values
            ]
        except Exception as e:
            warnings.warn(f"Failed to find leaf segments: {e} - stopping shaving")
            break

        to_remove = [
            seg for seg in leaf_segs if any(np.isin(seg, long)) or (len(seg) <= 2)
        ]

        to_remove = [seg for seg in to_remove if len(seg) < 10]

        to_remove = [n for l in to_remove for n in l[:-1]]

        if not len(to_remove):
            break

        if len(to_remove) >= len(tn.nodes):
            warnings.warn("Attempting to remove all nodes during shaving - stopping")
            break

        try:
            navis.subset_neuron(tn, ~tn.nodes.node_id.isin(to_remove), inplace=True)
        except Exception as e:
            warnings.warn(f"Failed to remove nodes during shaving: {e} - stopping")
            break

    bp = tn.nodes.loc[tn.nodes.type == "branch", "node_id"].values

    is_end = tn.nodes.type == "end"
    parent_is_bp = tn.nodes.parent_id.isin(bp)
    twigs = tn.nodes.loc[is_end & parent_is_bp, "node_id"].values

    # Drop the short terminal twigs via the public navis API rather than poking
    # at tn._nodes / tn._clear_temp_attr() directly.
    if len(twigs):
        navis.subset_neuron(tn, ~tn.nodes.node_id.isin(twigs), inplace=True)


def _apply_soma_processing(
    tn: navis.TreeNeuron, root_id: int, client: CAVEclient, remove_soma_hairball: bool
) -> None:
    """Apply soma detection and processing.

    Tries to get soma from annotation system, falls back to radius-based
    detection, reroots skeleton to soma, and optionally removes soma hairball.

    Parameters
    ----------
    tn : navis.TreeNeuron
        Skeleton to process (modified in place).
    root_id : int
        Root ID for annotation lookup.
    client : CAVEclient
        Client for data access.
    remove_soma_hairball : bool
        Whether to remove dense branching in soma region.
    """
    soma = None

    soma_loc = get_soma_from_annotations(root_id, client)
    if soma_loc is not None:
        soma = tn.snap(soma_loc)[0]

    if soma is None:
        soma = detect_soma_skeleton(tn, min_rad=800, N=3)

    if soma:
        tn.soma = soma

        tn.reroot(tn.soma, inplace=True)

        if remove_soma_hairball:
            _remove_soma_hairball(tn, soma)


def _remove_soma_hairball(tn: navis.TreeNeuron, soma: int) -> None:
    """Remove hairball structure inside soma.

    Finds all nodes within 2x soma radius (min 4μm), identifies segments
    containing these nodes, keeps only the longest segment, and removes
    all other segments in the region.

    Parameters
    ----------
    tn : navis.TreeNeuron
        Skeleton to process (modified in place).
    soma : int
        Node ID of the soma.
    """
    soma_info = tn.nodes.set_index("node_id").loc[soma]
    soma_loc = soma_info[["x", "y", "z"]].values

    tree = navis.neuron2KDTree(tn)
    search_radius = max(4000, soma_info.radius * 2)
    ix = tree.query_ball_point(soma_loc, search_radius)

    ids = tn.nodes.iloc[ix].node_id.values

    segs = [s for s in tn.segments if any(np.isin(ids, s))]

    segs = sorted(segs, key=lambda x: len(x))

    to_drop = np.array([n for s in segs[:-1] for n in s])
    to_drop = to_drop[~np.isin(to_drop, segs[-1] + [soma])]

    navis.remove_nodes(tn, to_drop, inplace=True)


def divide_local_neighbourhood(mesh: trimesh.Trimesh, radius: float):
    """Divide the mesh into locally connected patches of a given size (overlapping).

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to divide.
    radius : float
        The radius (in mesh units) for local neighborhoods.

    Returns
    -------
    list of sets
        Each set contains vertex indices belonging to a local patch.
    """
    import numbers
    import networkx as nx

    assert isinstance(mesh, trimesh.Trimesh), "mesh must be a trimesh.Trimesh object"
    assert isinstance(radius, numbers.Number), "radius must be a number"

    # Generate a graph for mesh
    G = mesh.vertex_adjacency_graph
    # Use Euclidean distance for edge weights
    edges = np.array(G.edges)
    e1 = mesh.vertices[edges[:, 0]]
    e2 = mesh.vertices[edges[:, 1]]
    dist = np.sqrt(np.sum((e1 - e2) ** 2, axis=1))
    nx.set_edge_attributes(G, dict(zip(map(tuple, edges), dist)), name="weight")

    not_seen = set(G.nodes)
    patches = []
    while not_seen:
        center = not_seen.pop()
        sg = nx.ego_graph(G, center, radius=radius, distance="weight")
        nodes = set(sg.nodes)
        patches.append(nodes)
        not_seen -= nodes
    return patches
