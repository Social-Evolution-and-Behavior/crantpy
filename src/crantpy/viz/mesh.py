# -*- coding: utf-8 -*-
"""Mesh module for CRANTBpy."""

import functools
import logging
from typing import (
    Dict,
    List,
    Optional,
    Union,
)

import numpy as np
import navis
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import skeletor as sk
from crantpy.utils.cave import get_cave_client, get_cloudvolume
import trimesh as tm
from tqdm import tqdm
from cloudvolume import CloudVolume

from crantpy.utils.config import (
    CRANT_VALID_DATASETS,
    SCALE_X,
    SCALE_Y,
    SCALE_Z,
    WHOLE_BRAIN_TISSUE_MESH_URL,
    NEUROPIL_MESH_URL,
    NEUROPIL_MESH_DICT,
    NEUROPIL_MESH_ALIASES,
)
from crantpy.utils.decorators import inject_dataset, parse_neuroncriteria
from crantpy.queries.neurons import NeuronCriteria
from crantpy.utils.helpers import parse_root_ids, retry

from neuroglancer_scripts.mesh import read_precomputed_mesh
import requests
import io
import pyvista as pv
import seaborn as sns
from cloudvolume.cacheservice import CacheService
from cloudvolume.datasource.precomputed import SharedConfiguration
from cloudvolume.datasource.precomputed.metadata import PrecomputedMetadata
from cloudvolume.datasource.precomputed.mesh import PrecomputedMeshSource


@inject_dataset(allowed=CRANT_VALID_DATASETS)
@parse_neuroncriteria()
def get_mesh_neuron(
    neurons: Union[int, str, List[Union[int, str]], "NeuronCriteria"],
    dataset: Optional[str] = None,
    omit_failures: Optional[bool] = None,
    threads: int = 5,
    progress: bool = True,
) -> Union["navis.MeshNeuron", "navis.NeuronList"]:
    """
    Fetch one or more CRANTB neurons as navis.MeshNeuron objects.

    This function retrieves mesh representations for the specified neuron(s) from the selected dataset.
    It supports batch queries, parallel fetching, and flexible error handling.

    Parameters
    ----------
    neurons : int, str, list of int/str, or NeuronCriteria
        Neuron root ID(s) or a NeuronCriteria instance specifying which neurons to fetch.
        Accepts a single ID, a list/array of IDs, or a NeuronCriteria object.
    dataset : str, optional
        Dataset to fetch info from. If None, uses the default dataset.
    omit_failures : bool, optional
        Behavior when mesh download fails:
            - None (default): raise an exception
            - True: skip the offending neuron (may result in empty NeuronList)
            - False: return an empty MeshNeuron for failed fetches
    threads : int, optional
        Number of parallel threads to use for batch queries. Default is 5.
    progress : bool, optional
        Whether to show a progress bar during batch fetching. Default is True.

    Returns
    -------
    navis.MeshNeuron or navis.NeuronList
        MeshNeuron if a single neuron is requested, or NeuronList for multiple neurons.

    """
    if omit_failures not in (None, True, False):
        raise ValueError(
            "`omit_failures` must be either None, True or False. "
            f'Got "{omit_failures}".'
        )

    # Normalize input
    root_ids = parse_root_ids(neurons)

    # Convert to list of ints
    root_ids = [int(rid) for rid in root_ids]

    # Batch mode: multiple root IDs
    if len(root_ids) > 1:
        get_mesh = partial(
            get_mesh_neuron,
            dataset=dataset,
            omit_failures=omit_failures,
            threads=None,
            progress=False,
        )
        results = []
        with ThreadPoolExecutor(max_workers=threads) as pool:
            futures = pool.map(get_mesh, root_ids)
            results = [
                f
                for f in navis.config.tqdm(
                    futures,
                    desc="Fetching meshes",
                    total=len(root_ids),
                    disable=not progress or len(root_ids) == 1,
                    leave=False,
                )
            ]
        return navis.NeuronList(results)

    # Single root ID
    root_id = root_ids[0]

    # Get/Initialize the CAVE client
    _ = get_cave_client(dataset=dataset)
    vol = get_cloudvolume()

    # Set logging level for urllib3 to suppress warnings
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

    try:
        vol.parallel = threads if threads else 1
        mesh = None
        if vol.path.startswith("graphene"):
            mesh = vol.mesh.get(root_id, deduplicate_chunk_boundaries=False)[root_id]
        elif vol.path.startswith("precomputed"):
            try:
                mesh = vol.mesh.get(root_id)[root_id]
            except Exception as e:
                raise ValueError(f"No mesh for id {root_id} found") from e
        if mesh is None:
            raise ValueError(f"No mesh for id {root_id} found")
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logging.error(f"Failed to fetch mesh for root_id {root_id}: {e}")
        if omit_failures is None:
            raise
        elif omit_failures:
            return navis.NeuronList([])
        else:
            return navis.MeshNeuron(None, id=root_id, units="nm", dataset=dataset)

    n = navis.MeshNeuron(mesh, id=root_id, units="nm", dataset=dataset)

    return n


@inject_dataset(allowed=CRANT_VALID_DATASETS)
def detect_soma(
    x: Union[
        int,
        str,
        tm.Trimesh,
        navis.MeshNeuron,
        navis.NeuronList,
        "NeuronCriteria",
        List[Union[int, str, tm.Trimesh, navis.MeshNeuron, "NeuronCriteria"]],
    ],
    dataset=None,
    min_rad=800,
    N=3,
    progress=True,
) -> np.ndarray:
    """
    Detect the soma (cell body) location of a neuron based on mesh radius.

    This function attempts to identify the soma by finding regions of the mesh with a sufficiently large radius.
    It supports both single neurons and batch input.

    Parameters
    ----------
    x : int, str, trimesh.Trimesh, navis.MeshNeuron, or list/array-like
        Neuron ID, mesh, or list thereof. Meshes must not be downsampled. If a list/array is provided, returns coordinates for each.
    dataset : str, optional
        Dataset to query. If None, falls back to the default dataset.
    min_rad : float, optional
        Minimum radius for a node to be considered a soma candidate. Default is 800.
    N : int, optional
        Number of consecutive vertices with radius > `min_rad` to consider as soma candidates. Default is 3.
    progress : bool, optional
        Whether to show a progress bar for batch input. Default is True.

    Returns
    -------
    np.ndarray
        If input is a single neuron, returns a (3,) array of x, y, z coordinates of the detected soma.
        If input is a list/array, returns (N, 3) array of coordinates for each neuron.
        If no soma is found, returns [None, None, None] for that neuron.

    """
    # Normalize input: handle batch mode
    # Accept navis.NeuronList, list, or np.ndarray, but not single neuron objects
    if isinstance(x, (list, np.ndarray, navis.NeuronList)) and not isinstance(
        x, (tm.Trimesh, navis.MeshNeuron, int, str, NeuronCriteria)
    ):
        return np.vstack(
            [
                detect_soma(n, min_rad=min_rad, N=N, progress=False, dataset=dataset)
                for n in tqdm(
                    x, desc="Detecting soma", disable=not progress, leave=False
                )
            ]
        )

    # Single mesh or neuron
    mesh = None
    if isinstance(x, tm.Trimesh):
        mesh = x
    elif isinstance(x, navis.MeshNeuron):
        mesh = x.trimesh
    else:
        try:
            mesh = get_mesh_neuron(x, dataset=dataset).trimesh
        except Exception as e:
            logging.error(f"Failed to fetch mesh for soma detection: {e}")
            return np.array([None, None, None])

    try:
        centers, radii, G = sk.skeletonize.wave._cast_waves(
            mesh, waves=3, step_size=1, progress=True
        )
    except Exception as e:
        logging.error(f"Failed to compute skeleton/waves for soma detection: {e}")
        return np.array([None, None, None])

    is_big = np.where(radii >= min_rad)[0]
    if not any(is_big):
        return np.array([None, None, None])

    # Find stretches of consecutive above-threshold radii
    candidates = []
    for stretch in np.split(is_big, np.where(np.diff(is_big) != 1)[0] + 1):
        if len(stretch) < N:
            continue
        candidates += [i for i in stretch]

    if not candidates:
        return np.array([None, None, None])

    # Find the largest candidate
    candidates = sorted(candidates, key=lambda idx: radii[idx])

    # Convert to integer coordinates
    center = centers[candidates[-1]]
    center = [center[0] / SCALE_X, center[1] / SCALE_Y, center[2] / SCALE_Z]
    center = np.array([int(round(coord)) for coord in center], dtype=int)

    return center


@retry
def load_whole_brain_mesh() -> tm.Trimesh:
    """
    Download and decode a whole-brain tissue mesh.

    Returns
    -------
    trimesh.Trimesh
        The whole-brain tissue mesh.

    """
    response = requests.get(WHOLE_BRAIN_TISSUE_MESH_URL)
    response.raise_for_status()
    vertices, faces = read_precomputed_mesh(io.BytesIO(response.content))
    # Ensure canonical dtypes
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.uint32)
    # Convert to trimesh
    brain_trimesh = tm.Trimesh(vertices=vertices, faces=faces, process=False)
    return brain_trimesh


def get_supported_neuropil_mesh_labels() -> List[str]:
    """Return all supported neuropil mesh labels, including compatibility aliases."""
    return sorted([*NEUROPIL_MESH_DICT.values(), *NEUROPIL_MESH_ALIASES.keys()])


def _resolve_neuropil_mesh_label_ids(
    neuropil_label: str,
    label_map: Dict[int, str],
    alias_map: Optional[Dict[str, List[str]]] = None,
) -> List[int]:
    """Resolve a neuropil label or alias to one or more source mesh IDs."""
    alias_map = alias_map or {}
    label_to_id = {name: label_id for label_id, name in label_map.items()}

    if neuropil_label in label_to_id:
        return [label_to_id[neuropil_label]]

    if neuropil_label in alias_map:
        alias_labels = alias_map[neuropil_label]
        missing_labels = [label for label in alias_labels if label not in label_to_id]
        if missing_labels:
            raise ValueError(
                f"Invalid neuropil alias configuration for {neuropil_label}: "
                f"missing source labels {missing_labels}"
            )
        return [label_to_id[label] for label in alias_labels]

    raise ValueError(
        f"Invalid neuropil label: {neuropil_label}. "
        f"Available labels are: {sorted([*label_to_id.keys(), *alias_map.keys()])}"
    )


def resolve_neuropil_mesh_label_ids(neuropil_label: str) -> List[int]:
    """Resolve a neuropil label or alias to one or more source mesh IDs."""
    return _resolve_neuropil_mesh_label_ids(
        neuropil_label,
        NEUROPIL_MESH_DICT,
        NEUROPIL_MESH_ALIASES,
    )


def _normalize_precomputed_mesh_url(mesh_url: str) -> str:
    """Normalize a precomputed mesh URL so cache keys and fetches are stable."""
    return mesh_url.rstrip("/")


def _is_http_mesh_source(mesh_url: str) -> bool:
    """Return True for mesh-only HTTP(S) sources that need direct mesh metadata loading."""
    stripped = _normalize_precomputed_mesh_url(mesh_url).removeprefix("precomputed://")
    return stripped.startswith(("http://", "https://"))


@functools.lru_cache(maxsize=None)
def _get_cloudvolume_mesh_volume(mesh_url: str):
    """Create a CloudVolume handle for standard precomputed sources."""
    mesh_cloudpath = _normalize_precomputed_mesh_url(mesh_url)
    return CloudVolume(
        mesh_cloudpath,
        mip=0,
        fill_missing=False,
        use_https=True,
        progress=False,
    )


@functools.lru_cache(maxsize=None)
def _get_precomputed_mesh_source(mesh_url: str):
    """Create a mesh-layer source for a mesh-only neuropil dataset."""
    mesh_cloudpath = _normalize_precomputed_mesh_url(mesh_url)
    mesh_info_url = mesh_cloudpath.removeprefix("precomputed://") + "/info"
    mesh_info = requests.get(mesh_info_url)
    mesh_info.raise_for_status()
    mesh_info = mesh_info.json()

    config = SharedConfiguration(
        cdn_cache=False,
        compress=True,
        compress_level=None,
        green=False,
        mip=0,
        parallel=1,
        progress=False,
        secrets=None,
        spatial_index_db=None,
        cache_locking=False,
        codec_threads=1,
    )
    cache = CacheService(
        cloudpath=mesh_cloudpath,
        enabled=False,
        config=config,
        compress=True,
    )
    meta = PrecomputedMetadata(
        mesh_cloudpath,
        config,
        cache,
        info={"mesh": ""},
        provenance={
            "sources": [],
            "owners": [],
            "processing": [],
            "description": "",
        },
    )
    return PrecomputedMeshSource(meta, cache, config, info=mesh_info)


def _cv_mesh_to_trimesh(mesh) -> tm.Trimesh:
    """Convert a CloudVolume mesh object to a trimesh.Trimesh."""
    return tm.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces),
        process=False,
    )


def _trimesh_to_polydata(mesh: tm.Trimesh) -> "pv.PolyData":
    """Convert a trimesh.Trimesh to a PyVista PolyData object."""
    return pv.PolyData(
        mesh.vertices,
        np.hstack((np.full((len(mesh.faces), 1), 3), mesh.faces)),
    )


def _load_neuropil_mesh_from_source(
    neuropil_label: str,
    mesh_url: str,
    label_map: Dict[int, str],
    alias_map: Optional[Dict[str, List[str]]] = None,
) -> tm.Trimesh:
    """Load a neuropil mesh from an arbitrary source URL plus label mapping.

    This is a private utility for internal comparisons and notebook audits.
    Production code should call ``load_neuropil_mesh()`` so the current configured
    source stays isolated in one place.
    """
    label_ids = _resolve_neuropil_mesh_label_ids(neuropil_label, label_map, alias_map)
    meshes = []

    if _is_http_mesh_source(mesh_url):
        mesh_source = _get_precomputed_mesh_source(mesh_url)
        for label_id in label_ids:
            # CloudVolume multi-LOD meshes use lod=0 for the highest-detail surface.
            mesh_data = mesh_source.get(label_id, lod=0)
            if isinstance(mesh_data, dict):
                mesh = mesh_data.get(label_id)
                if mesh is None:
                    mesh = next(iter(mesh_data.values()))
            else:
                mesh = mesh_data
            meshes.append(_cv_mesh_to_trimesh(mesh))
    else:
        vol = _get_cloudvolume_mesh_volume(mesh_url)
        for label_id in label_ids:
            mesh_dict = vol.mesh.get(label_id)
            mesh = mesh_dict[label_id]
            meshes.append(_cv_mesh_to_trimesh(mesh))

    tri = meshes[0] if len(meshes) == 1 else tm.util.concatenate(meshes)
    tri.fix_normals()
    return tri


def _load_current_neuropil_mesh(neuropil_label: str) -> tm.Trimesh:
    """Load a neuropil mesh from the current configured production source."""
    return _load_neuropil_mesh_from_source(
        neuropil_label,
        NEUROPIL_MESH_URL,
        NEUROPIL_MESH_DICT,
        NEUROPIL_MESH_ALIASES,
    )


@retry
def load_neuropil_mesh(neuropil_label: str) -> tm.Trimesh:
    """
    Download and decode a neuropil mesh.
    Parameters
    ----------
    neuropil_label : str
        The label of the neuropil to load.

    Returns
    -------
    trimesh.Trimesh
        The neuropil mesh.

    """
    return _load_current_neuropil_mesh(neuropil_label)


@inject_dataset(allowed=CRANT_VALID_DATASETS)
@parse_neuroncriteria()
def get_brain_mesh_scene(
    neurons: Union[
        int,
        str,
        tm.Trimesh,
        navis.MeshNeuron,
        navis.NeuronList,
        "NeuronCriteria",
        List[Union[int, str, tm.Trimesh, navis.MeshNeuron, "NeuronCriteria"]],
        None,
    ] = None,
    dataset: Optional[str] = None,
    omit_failures: Optional[bool] = None,
    threads: int = 5,
    progress: bool = True,
    brain_mesh_color: Union[str, tuple, list] = "grey",
    brain_mesh_alpha: float = 0.1,
    neuron_mesh_alpha: float = 1,
    neuron_mesh_colors: Union[List[Union[str, tuple, list]], None] = None,
    neuropil_meshes: Union[str, tm.Trimesh, List[Union[str, tm.Trimesh]], None] = None,
    neuropil_mesh_alphas: Union[float, List[float], None] = None,
    neuropil_mesh_colors: Union[
        str, tuple, list, List[Union[str, tuple, list]], None
    ] = None,
    backend: str = "static",
) -> pv.Plotter:
    """
    Create a 3D scene of the brain mesh with the specified neurons in random colors.

    Parameters
    ----------
    neurons : Union[int, str, tm.Trimesh, navis.MeshNeuron, navis.NeuronList, 'NeuronCriteria', List[Union[int, str, tm.Trimesh, navis.MeshNeuron, 'NeuronCriteria']], None], optional
        The neurons to highlight in the scene. If None, only brain and neuropil meshes will be shown, by default None
    dataset : Optional[str], optional
        The dataset to use for fetching neuron meshes, by default None
    omit_failures : Optional[bool], optional
        Whether to omit neurons that fail to load, by default None
    threads : int, optional
        The number of threads to use for loading meshes, by default 5
    progress : bool, optional
        Whether to show a progress bar, by default True
    brain_mesh_color : Union[str, tuple, list], optional
        The color of the brain mesh. Can be a color name string (e.g., "grey") or an RGB/RGBA
        tuple/list (e.g., (0.5, 0.5, 0.5) or [128, 128, 128, 255]), by default "grey"
    brain_mesh_alpha : Optional[float], optional
        The transparency of the brain mesh, by default 0.1
    neuron_mesh_alpha : Optional[float], optional
        The transparency of the neuron meshes, by default 1
    neuron_mesh_colors : Union[List[Union[str, tuple, list]], None], optional
        List of colors for neuron meshes. Each color can be a string or RGB/RGBA tuple/list.
        If None, random colors are generated, by default None
    neuropil_meshes : Union[str, tm.Trimesh, List[Union[str, tm.Trimesh]], None], optional
        Neuropil meshes to add to the scene. Can be a single neuropil label string, a trimesh object,
        or a list of neuropil label strings and/or trimesh objects. If strings are provided, the
        function will attempt to load them using load_neuropil_mesh(), by default None
    neuropil_mesh_alphas : Union[float, List[float], None], optional
        Transparency value(s) for neuropil meshes. Can be a single float applied to all neuropil
        meshes or a list of floats (one per neuropil mesh). If None, defaults to 0.3, by default None
    neuropil_mesh_colors : Union[str, tuple, list, List[Union[str, tuple, list]], None], optional
        Color(s) for neuropil meshes. Can be a single color (string or RGB/RGBA tuple/list) applied
        to all neuropil meshes or a list of colors (one per neuropil mesh). If None, random colors
        are generated, by default None
    backend : Optional[str], optional
        The pyvista backend to use ('static', 'trame', 'client'), by default 'static'

    Returns
    -------
    pv.Plotter
        The 3D scene containing the brain mesh and highlighted neurons.
    """
    logging.info("Loading whole brain mesh...")
    brain_trimesh = load_whole_brain_mesh()
    logging.info(
        f"Whole brain mesh loaded: {len(brain_trimesh.vertices)} vertices, {len(brain_trimesh.faces)} faces"
    )

    # Load neuron meshes if provided
    neuron_meshes = []
    if neurons is not None:
        logging.info("Loading neuron meshes...")
        result = get_mesh_neuron(
            neurons,
            dataset=dataset,
            omit_failures=omit_failures,
            threads=threads,
            progress=progress,
        )
        # Normalize single MeshNeuron to list for uniform handling
        neuron_list = (
            result
            if isinstance(result, navis.NeuronList)
            else navis.NeuronList([result])
        )
        neuron_meshes = [
            tm.Trimesh(vertices=n.vertices, faces=n.faces, process=False)
            for n in neuron_list
        ]
        logging.info(f"Loaded {len(neuron_meshes)} neuron mesh(es)")

    # Process neuropil meshes if provided
    neuropil_trimeshes = []
    neuropil_colors = []
    neuropil_alphas = []
    if neuropil_meshes is not None:
        if not isinstance(neuropil_meshes, list):
            neuropil_meshes = [neuropil_meshes]

        logging.info(f"Loading {len(neuropil_meshes)} neuropil mesh(es)...")
        for neuropil in neuropil_meshes:
            if isinstance(neuropil, str):
                neuropil_trimeshes.append(load_neuropil_mesh(neuropil))
            elif isinstance(neuropil, tm.Trimesh):
                neuropil_trimeshes.append(neuropil)
            else:
                raise ValueError(
                    f"Neuropil mesh must be a string label or trimesh.Trimesh object, got {type(neuropil)}"
                )
        logging.info(f"Loaded {len(neuropil_trimeshes)} neuropil mesh(es)")

        # Process neuropil mesh alphas
        if neuropil_mesh_alphas is None:
            neuropil_alphas = [0.3] * len(neuropil_trimeshes)
        elif isinstance(neuropil_mesh_alphas, (int, float)):
            neuropil_alphas = [float(neuropil_mesh_alphas)] * len(neuropil_trimeshes)
        elif isinstance(neuropil_mesh_alphas, list):
            if len(neuropil_mesh_alphas) != len(neuropil_trimeshes):
                raise ValueError(
                    f"Number of neuropil_mesh_alphas ({len(neuropil_mesh_alphas)}) "
                    f"does not match number of neuropil meshes ({len(neuropil_trimeshes)})"
                )
            neuropil_alphas = neuropil_mesh_alphas
        else:
            raise ValueError("neuropil_mesh_alphas must be a float or list of floats")

        # Process neuropil mesh colors
        if neuropil_mesh_colors is None:
            neuropil_colors = sns.color_palette("husl", len(neuropil_trimeshes))
        elif isinstance(neuropil_mesh_colors, (str, tuple)):
            # Single color (string or RGB tuple) applied to all meshes
            neuropil_colors = [neuropil_mesh_colors] * len(neuropil_trimeshes)
        elif isinstance(neuropil_mesh_colors, list):
            # Check if it's a single RGB color (list of 3 or 4 numbers) or a list of colors
            if len(neuropil_mesh_colors) > 0 and isinstance(
                neuropil_mesh_colors[0], (int, float)
            ):
                # It's a single RGB/RGBA color as a list
                neuropil_colors = [neuropil_mesh_colors] * len(neuropil_trimeshes)
            else:
                # It's a list of colors
                if len(neuropil_mesh_colors) != len(neuropil_trimeshes):
                    raise ValueError(
                        f"Number of neuropil_mesh_colors ({len(neuropil_mesh_colors)}) "
                        f"does not match number of neuropil meshes ({len(neuropil_trimeshes)})"
                    )
                neuropil_colors = neuropil_mesh_colors
        else:
            raise ValueError(
                "neuropil_mesh_colors must be a string, tuple, list, or list of colors"
            )

    logging.info("Converting meshes to PyVista PolyData format...")
    brain_pv = _trimesh_to_polydata(brain_trimesh)
    neuron_meshes_pv = [_trimesh_to_polydata(n) for n in neuron_meshes]
    neuropil_meshes_pv = [_trimesh_to_polydata(n) for n in neuropil_trimeshes]

    # Set backend for pyvista (local or trame)
    pv.set_jupyter_backend(backend)
    logging.info(f"Using PyVista backend: {backend}")

    # Create a PyVista plotter
    logging.info("Creating PyVista scene...")
    plotter = pv.Plotter()

    # Add the brain mesh
    logging.info(
        f"Adding brain mesh (color={brain_mesh_color}, alpha={brain_mesh_alpha})"
    )
    plotter.add_mesh(brain_pv, color=brain_mesh_color, opacity=brain_mesh_alpha)

    # Add the neuron meshes if present
    if neuron_meshes_pv:
        # Generate random colors for neurons if not provided
        if neuron_mesh_colors is None or len(neuron_mesh_colors) != len(
            neuron_meshes_pv
        ):
            neuron_colors = sns.color_palette("bright", len(neuron_meshes_pv))
        else:
            neuron_colors = neuron_mesh_colors

        logging.info(
            f"Adding {len(neuron_meshes_pv)} neuron mesh(es) (alpha={neuron_mesh_alpha})"
        )
        for neuron_pv, color in zip(neuron_meshes_pv, neuron_colors):
            plotter.add_mesh(neuron_pv, color=color, opacity=neuron_mesh_alpha)

    # Add the neuropil meshes
    if neuropil_meshes_pv:
        logging.info(f"Adding {len(neuropil_meshes_pv)} neuropil mesh(es)")
        for neuropil_pv, color, alpha in zip(
            neuropil_meshes_pv, neuropil_colors, neuropil_alphas
        ):
            plotter.add_mesh(neuropil_pv, color=color, opacity=alpha)

    # Set camera position
    plotter.view_xy(negative=True)
    plotter.set_viewup([0, -1, 1])

    logging.info("Brain mesh scene created successfully")
    return plotter
