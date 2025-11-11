# -*- coding: utf-8 -*-
"""Mesh module for CRANTBpy."""

import functools
import logging
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import pandas as pd
import navis
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import networkx as nx
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
    NEUROPIL_MESH_DICT
)
from crantpy.utils.decorators import inject_dataset, parse_neuroncriteria
from crantpy.queries.neurons import NeuronCriteria
from crantpy.utils.helpers import parse_root_ids, retry

from neuroglancer_scripts.mesh import read_precomputed_mesh
import requests
import io
import pyvista as pv
import seaborn as sns


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


@retry
def load_neuropil_mesh(
        neuropil_label: str
) -> tm.Trimesh:
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

    # Check if the label exists in the dictionary
    label_id = None
    for key, value in NEUROPIL_MESH_DICT.items():
        if value == neuropil_label:
            label_id = key
            break

    if label_id is None:
        raise ValueError(f"Invalid neuropil label: {neuropil_label}. Available labels are: {list(NEUROPIL_MESH_DICT.values())}")

    vol = CloudVolume(NEUROPIL_MESH_URL, mip=0, fill_missing=False, use_https=True, progress=True)
    mesh_dict = vol.mesh.get(label_id)
    mesh = mesh_dict[label_id]
    tri = tm.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    return tri



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
    brain_mesh_color: str = "grey",
    brain_mesh_alpha: float = 0.1,
    neuron_mesh_alpha: float = 1,
    neuron_mesh_colors: list = None,
    neuropil_meshes: Union[
        str,
        tm.Trimesh,
        List[Union[str, tm.Trimesh]],
        None
    ] = None,
    neuropil_mesh_alphas: Union[float, List[float], None] = None,
    neuropil_mesh_colors: Union[str, List[str], None] = None,
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
    brain_mesh_color : Optional[str], optional
        The color of the brain mesh, by default "grey"
    brain_mesh_alpha : Optional[float], optional
        The transparency of the brain mesh, by default 0.1
    neuron_mesh_alpha : Optional[float], optional
        The transparency of the neuron meshes, by default 1
    neuron_mesh_colors : list, optional
        List of colors for neuron meshes. If None, random colors are generated, by default None
    neuropil_meshes : Union[str, tm.Trimesh, List[Union[str, tm.Trimesh]], None], optional
        Neuropil meshes to add to the scene. Can be a single neuropil label string, a trimesh object,
        or a list of neuropil label strings and/or trimesh objects. If strings are provided, the
        function will attempt to load them using load_neuropil_mesh(), by default None
    neuropil_mesh_alphas : Union[float, List[float], None], optional
        Transparency value(s) for neuropil meshes. Can be a single float applied to all neuropil
        meshes or a list of floats (one per neuropil mesh). If None, defaults to 0.3, by default None
    neuropil_mesh_colors : Union[str, List[str], None], optional
        Color(s) for neuropil meshes. Can be a single color applied to all neuropil meshes or a list
        of colors (one per neuropil mesh). If None, random colors are generated, by default None
    backend : Optional[str], optional
        The pyvista backend to use ('static', 'trame', 'client'), by default 'static'

    Returns
    -------
    pv.Plotter
        The 3D scene containing the brain mesh and highlighted neurons.
    """
    logging.info("Loading whole brain mesh...")
    # Load the whole brain mesh
    brain_trimesh = load_whole_brain_mesh()
    logging.info(f"Whole brain mesh loaded: {len(brain_trimesh.vertices)} vertices, {len(brain_trimesh.faces)} faces")

    # Load neuron meshes if provided
    neuron_meshes = []
    if neurons is not None:
        logging.info("Loading neuron meshes...")
        # Load neuron meshes
        neuron_meshes_result = get_mesh_neuron(
            neurons,
            dataset=dataset,
            omit_failures=omit_failures,
            threads=threads,
            progress=progress,
        )
        logging.info(f"Loaded {len(neuron_meshes_result) if isinstance(neuron_meshes_result, navis.NeuronList) else 1} neuron mesh(es)")

        # Convert to Trimesh objects
        if isinstance(neuron_meshes_result, navis.MeshNeuron):
            neuron_meshes = [
                tm.Trimesh(
                    vertices=neuron_meshes_result.vertices,
                    faces=neuron_meshes_result.faces,
                    process=False,
                )
            ]
        elif isinstance(neuron_meshes_result, navis.NeuronList):
            neuron_meshes = [
                tm.Trimesh(vertices=n.vertices, faces=n.faces, process=False)
                for n in neuron_meshes_result
            ]
        else:
            # Throw error if unexpected type
            raise ValueError("Unexpected type for neuron_meshes")
    else:
        logging.info("No neurons provided, skipping neuron mesh loading")

    # Process neuropil meshes if provided
    neuropil_trimeshes = []
    if neuropil_meshes is not None:
        logging.info("Processing neuropil meshes...")
        # Normalize to list
        if not isinstance(neuropil_meshes, list):
            neuropil_meshes = [neuropil_meshes]
        
        logging.info(f"Loading {len(neuropil_meshes)} neuropil mesh(es)...")
        # Load neuropil meshes
        for neuropil in neuropil_meshes:
            if isinstance(neuropil, str):
                logging.info(f"Loading neuropil mesh from label: {neuropil}")
                # Load from string label
                neuropil_trimeshes.append(load_neuropil_mesh(neuropil))
            elif isinstance(neuropil, tm.Trimesh):
                logging.info(f"Using provided trimesh object with {len(neuropil.vertices)} vertices")
                # Already a trimesh object
                neuropil_trimeshes.append(neuropil)
            else:
                raise ValueError(
                    f"Neuropil mesh must be a string label or trimesh.Trimesh object, got {type(neuropil)}"
                )
        
        logging.info(f"Successfully loaded {len(neuropil_trimeshes)} neuropil mesh(es)")
        
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
        elif isinstance(neuropil_mesh_colors, str):
            neuropil_colors = [neuropil_mesh_colors] * len(neuropil_trimeshes)
        elif isinstance(neuropil_mesh_colors, list):
            if len(neuropil_mesh_colors) != len(neuropil_trimeshes):
                raise ValueError(
                    f"Number of neuropil_mesh_colors ({len(neuropil_mesh_colors)}) "
                    f"does not match number of neuropil meshes ({len(neuropil_trimeshes)})"
                )
            neuropil_colors = neuropil_mesh_colors
        else:
            raise ValueError("neuropil_mesh_colors must be a string or list of strings")

    logging.info("Converting meshes to PyVista PolyData format...")
    # Convert to pv.PolyData
    brain_pv = pv.PolyData(
        brain_trimesh.vertices,
        np.hstack((np.full((len(brain_trimesh.faces), 1), 3), brain_trimesh.faces)),
    )
    
    neuron_meshes_pv = []
    if neuron_meshes:
        neuron_meshes_pv = [
            pv.PolyData(
                neuron.vertices,
                np.hstack((np.full((len(neuron.faces), 1), 3), neuron.faces)),
            )
            for neuron in neuron_meshes
        ]
    
    # Convert neuropil meshes to pv.PolyData
    neuropil_meshes_pv = []
    if neuropil_trimeshes:
        neuropil_meshes_pv = [
            pv.PolyData(
                neuropil.vertices,
                np.hstack((np.full((len(neuropil.faces), 1), 3), neuropil.faces)),
            )
            for neuropil in neuropil_trimeshes
        ]

    # Set backend for pyvista (local or trame)
    pv.set_jupyter_backend(backend)
    logging.info(f"Using PyVista backend: {backend}")

    # Create a PyVista plotter
    logging.info("Creating PyVista scene...")
    plotter = pv.Plotter()

    # Add the brain mesh
    logging.info(f"Adding brain mesh (color={brain_mesh_color}, alpha={brain_mesh_alpha})")
    plotter.add_mesh(brain_pv, color=brain_mesh_color, opacity=brain_mesh_alpha)

    # Add the neuron meshes if present
    if neuron_meshes_pv:
        # Generate random colors for neurons if not provided
        if neuron_mesh_colors is None or len(neuron_mesh_colors) != len(neuron_meshes_pv):
            neuron_colors = sns.color_palette("bright", len(neuron_meshes_pv))
        else:
            neuron_colors = neuron_mesh_colors

        logging.info(f"Adding {len(neuron_meshes_pv)} neuron mesh(es) (alpha={neuron_mesh_alpha})")
        for neuron_pv, color in zip(neuron_meshes_pv, neuron_colors):
            plotter.add_mesh(neuron_pv, color=color, opacity=neuron_mesh_alpha)

    # Add the neuropil meshes
    if neuropil_meshes_pv:
        logging.info(f"Adding {len(neuropil_meshes_pv)} neuropil mesh(es)")
        for neuropil_pv, color, alpha in zip(neuropil_meshes_pv, neuropil_colors, neuropil_alphas):
            plotter.add_mesh(neuropil_pv, color=color, opacity=alpha)

    # Set camera position
    plotter.view_xy(negative=True)
    plotter.set_viewup([0, -1, 1])

    logging.info("Brain mesh scene created successfully")
    return plotter
