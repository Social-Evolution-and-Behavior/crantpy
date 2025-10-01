# -*- coding: utf-8 -*-
"""Mesh module for CRANTBpy."""

import functools
import logging
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Optional,
                    Type, TypeVar, Union, cast)

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

from crantpy.utils.config import CRANT_VALID_DATASETS, SCALE_X, SCALE_Y, SCALE_Z, WHOLE_BRAIN_TISSUE_MESH_URL
from crantpy.utils.decorators import inject_dataset, parse_neuroncriteria
from crantpy.queries.neurons import NeuronCriteria 

from neuroglancer_scripts.mesh import read_precomputed_mesh
import requests
import io 
import pyvista as pv
import seaborn as sns   

@inject_dataset(allowed=CRANT_VALID_DATASETS)
@parse_neuroncriteria()
def get_mesh_neuron(
    neurons: Union[int, str, List[Union[int, str]], 'NeuronCriteria'],
    dataset: Optional[str] = None,
    omit_failures: Optional[bool] = None,
    threads: int = 5,
    progress: bool = True,
) -> Union['navis.MeshNeuron', 'navis.NeuronList']:
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
        raise ValueError("`omit_failures` must be either None, True or False. "
                         f'Got "{omit_failures}".')

    # Normalize input
    if hasattr(neurons, 'get_roots'):
        root_ids = neurons.get_roots()
    elif isinstance(neurons, (int, str)):
        root_ids = np.array([neurons])
    elif isinstance(neurons, (list, np.ndarray)):
        root_ids = np.array(neurons)
    else:
        logging.error(f"Invalid input type for 'neurons': {type(neurons)}. Must be int, str, list, np.ndarray, or NeuronCriteria.")
        raise ValueError("Invalid input type for neurons. Must be int, str, list, np.ndarray, or NeuronCriteria.")

    # Convert to list of ints
    root_ids = [int(rid) for rid in root_ids]

    # Batch mode: multiple root IDs
    if len(root_ids) > 1:
        get_mesh = partial(get_mesh_neuron, dataset=dataset, omit_failures=omit_failures, 
                           threads=None, progress=False)
        results = []
        with ThreadPoolExecutor(max_workers=threads) as pool:
            futures = pool.map(get_mesh, root_ids)
            results = [
                f for f in navis.config.tqdm(
                    futures,
                    desc="Fetching meshes",
                    total=len(root_ids),
                    disable=not progress or len(root_ids) == 1,
                    leave=False
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
    x: Union[int, str, tm.Trimesh, navis.MeshNeuron, navis.NeuronList, 'NeuronCriteria', List[Union[int, str, tm.Trimesh, navis.MeshNeuron, 'NeuronCriteria']]],
    dataset=None,
    min_rad=800,
    N=3,
    progress=True
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
    if (
        isinstance(x, (list, np.ndarray, navis.NeuronList))
        and not isinstance(x, (tm.Trimesh, navis.MeshNeuron, int, str, NeuronCriteria))
    ):
        return np.vstack([
            detect_soma(n, min_rad=min_rad, N=N, progress=False, dataset=dataset)
            for n in tqdm(
                x, desc="Detecting soma", disable=not progress, leave=False
            )
        ])

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
    center = [center[0]/SCALE_X, center[1]/SCALE_Y, center[2]/SCALE_Z]
    center = np.array([int(round(coord)) for coord in center], dtype=int)

    return center


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
    brain_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return brain_trimesh

@inject_dataset(allowed=CRANT_VALID_DATASETS)
@parse_neuroncriteria()
def get_brain_mesh_scene(
    neurons: Union[int, str, tm.Trimesh, navis.MeshNeuron, navis.NeuronList, 'NeuronCriteria', List[Union[int, str, tm.Trimesh, navis.MeshNeuron, 'NeuronCriteria']]], 
    dataset: Optional[str] = None,
    omit_failures: Optional[bool] = None,
    threads: int = 5,
    progress: bool = True,
    brain_mesh_color: Optional[str] = "grey", 
    brain_mesh_alpha: Optional[float] = 0.1, 
    neuron_mesh_alpha: Optional[float] = 1, 
    backend: Optional[str] = 'client'
) -> pv.Plotter:
    """
    Create a 3D scene of the brain mesh with the specified neurons in random colors. 

    Parameters
    ----------
    neurons : Union[int, str, tm.Trimesh, navis.MeshNeuron, navis.NeuronList, 'NeuronCriteria', List[Union[int, str, tm.Trimesh, navis.MeshNeuron, 'NeuronCriteria']]]
        The neurons to highlight in the scene.
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
    backend : Optional[str], optional
        The pyvista backend to use ('static', 'trame', 'client'), by default 'client'

    Returns
    -------
    pv.Plotter
        The 3D scene containing the brain mesh and highlighted neurons.
    """
    # Load the whole brain mesh 
    brain_trimesh = load_whole_brain_mesh()

    # Load neuron meshes
    neuron_meshes = get_mesh_neuron(
        neurons,
        dataset=dataset,
        omit_failures=omit_failures,
        threads=threads,
        progress=progress
    )

    # Convert to Trimesh objects
    if isinstance(neuron_meshes, navis.MeshNeuron):
        neuron_meshes = [trimesh.Trimesh(vertices=neuron_meshes.vertices, faces=neuron_meshes.faces, process=False)]
    elif isinstance(neuron_meshes, navis.NeuronList):
        neuron_meshes = [trimesh.Trimesh(vertices=n.vertices, faces=n.faces, process=False) for n in neuron_meshes]
    else:
        # Throw error if unexpected type
        raise ValueError("Unexpected type for neuron_meshes")

    # Convert to pv.PolyData 
    brain_pv = pv.PolyData(brain_trimesh.vertices, np.hstack((np.full((len(brain_trimesh.faces),1),3), brain_trimesh.faces)))
    neuron_meshes_pv = [pv.PolyData(neuron.vertices, np.hstack((np.full((len(neuron.faces),1),3), neuron.faces))) for neuron in neuron_meshes]

    # Set backend for pyvista (local or trame)
    pv.set_jupyter_backend(backend)

    # Create a PyVista plotter
    plotter = pv.Plotter()

    # Add the brain mesh
    plotter.add_mesh(brain_pv, color=brain_mesh_color, opacity=brain_mesh_alpha)

    # Generate random colors for neurons
    neuron_colors = sns.color_palette("bright", len(neuron_meshes_pv))


    # Add the neuron meshes
    for neuron_pv, color in zip(neuron_meshes_pv, neuron_colors):
        plotter.add_mesh(neuron_pv, color=color, opacity=neuron_mesh_alpha)

    # Set camera position
    plotter.view_xy(negative=True)
    plotter.set_viewup([0, -1, 1])

    return plotter
