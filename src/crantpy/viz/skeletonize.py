# -*- coding: utf-8 -*-
"""Neuron skeletonization tools."""

import os
import warnings
import numpy as np
import pandas as pd
import navis
import pcg_skel
import skeletor as sk
import trimesh
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from ..utils.decorators import parse_neuroncriteria, inject_dataset 
from ..utils.cave import get_cave_client as create_client
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from typing import Any, Dict, List, Optional, Tuple, Union
from numpy.typing import NDArray

# Skeleton metadata configuration
SKELETON_INFO = {
    "@type": "neuroglancer_skeletons",
    "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    "vertex_attributes": [{"id": "radius", "data_type": "float32", "num_components": 1}]
}

__all__ = ['skeletonize_neuron', 'skeletonize_neurons_parallel', 'get_skeletons']

@parse_neuroncriteria()
@inject_dataset()
def skeletonize_neuron(
    client: CAVEclient,
    root_id: int,
    shave_skeleton: bool = True,
    remove_soma_hairball: bool = False,
    assert_id_match: bool = False,
    threads: int = 2,
    save_to: Optional[str] = None,
    progress: bool = True,
    use_pcg_skel: bool = True,
    **kwargs: Any
) -> navis.TreeNeuron:
    """Skeletonize a neuron."""
    if save_to is not None:
        save_to = os.path.abspath(save_to)
        os.makedirs(os.path.dirname(save_to), exist_ok=True)

    if use_pcg_skel:
        try:
            skel = pcg_skel.pcg_skeleton(root_id=root_id, client=client)
            vertices = skel.vertices
            edges = skel.edges
            
            node_info = _create_node_info_dict(vertices, edges)
            df = _swc_dict_to_dataframe(node_info)
            
            tn = navis.TreeNeuron(df, id=root_id, units='1 nm')
            
        except Exception as e:
            warnings.warn(f"pcg_skel failed for {root_id}: {e}. Falling back to skeletor.")
            use_pcg_skel = False

    if not use_pcg_skel:
        try:
            mesh = client.mesh.get(root_id)
            if not isinstance(mesh, trimesh.Trimesh):
                mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
            mesh = sk.utilities.make_trimesh(mesh, validate=True)

            to_remove = int(0.0001 * mesh.vertices.shape[0])
            to_remove = None if to_remove == 0 else to_remove
            mesh = sk.pre.fix_mesh(mesh, inplace=True, remove_disconnected=to_remove)

            kwargs.pop('dataset', None)

            defaults = dict(waves=1, step_size=1)
            defaults.update(kwargs)
            s = sk.skeletonize.by_wavefront(mesh, progress=progress, **defaults)

            s.swc['node_id'] += 1
            s.swc.loc[s.swc.parent_id >= 0, 'parent_id'] += 1

            s.swc['radius'] = s.swc.radius.round().astype(int)

            tn = navis.TreeNeuron(s.swc, units='1 nm', id=root_id, soma=None)

        except Exception as e:
            raise ValueError(f"Failed to skeletonize neuron {root_id}: {e}")

    if shave_skeleton:
        _shave_skeleton(tn)

    if remove_soma_hairball:
        _remove_soma_hairball(tn)

    if assert_id_match:
        _assert_id_match(tn, root_id, client)

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
    **kwargs: Any
) -> Union[navis.NeuronList, Tuple[navis.NeuronList, List[str]]]:
    """Skeletonize multiple neurons in parallel."""
    if n_cores is not None:
        if n_cores < 1:
            raise ValueError("n_cores must be at least 1")
        if n_cores > mp.cpu_count():
            raise ValueError(f"n_cores cannot exceed {mp.cpu_count()}")
    else:
        n_cores = max(1, mp.cpu_count() // 2)

    kwargs['progress'] = False
    kwargs['threads'] = 1
    
    tasks = []
    for root_id in root_ids:
        task = (skeletonize_neuron, [client, root_id], dict(kwargs))
        tasks.append(task)

    results = []
    with mp.Pool(n_cores) as pool:
        for result in pool.imap(_worker_wrapper, tasks, chunksize=1):
            if isinstance(result, navis.TreeNeuron):
                results.append(result)
            else:
                warnings.warn(f'Failed to skeletonize neuron {result}')

    neurons = navis.NeuronList(results)

    if color_map is not None:
        cmap = cm.get_cmap(color_map, len(root_ids))
        colors = [mcolors.to_hex(cmap(i)) for i in range(len(root_ids))]
        return neurons, colors
    else:
        return neurons

def _shave_skeleton(tn: navis.TreeNeuron) -> None:
    """Remove small protrusions from the skeleton."""
    d = navis.morpho.mmetrics.parent_dist(tn, root_dist=0)
    long = tn.nodes[d >= 1000].node_id.values

    while True:
        leaf_segs = [seg for seg in tn.small_segments if seg[0] in tn.leafs.node_id.values]
        to_remove = [seg for seg in leaf_segs if any(np.isin(seg, long)) or (len(seg) <= 2)]
        to_remove = [seg for seg in to_remove if len(seg) < 10]
        to_remove = [n for l in to_remove for n in l[:-1]]

        if not len(to_remove):
            break

        navis.subset_neuron(tn, ~tn.nodes.node_id.isin(to_remove), inplace=True)

    bp = tn.nodes.loc[tn.nodes.type == 'branch', 'node_id'].values
    is_end = tn.nodes.type == 'end'
    parent_is_bp = tn.nodes.parent_id.isin(bp)
    twigs = tn.nodes.loc[is_end & parent_is_bp, 'node_id'].values
    tn._nodes = tn.nodes.loc[~tn.nodes.node_id.isin(twigs)].copy()
    tn._clear_temp_attr()

def _remove_soma_hairball(tn: navis.TreeNeuron) -> None:
    """Remove hairball structure inside the soma."""
    if not tn.soma:
        return

    soma = tn.nodes.set_index('node_id').loc[tn.soma]
    soma_loc = soma[['x', 'y', 'z']].values

    tree = navis.neuron2KDTree(tn)
    ix = tree.query_ball_point(soma_loc, max(4000, soma.radius * 2))

    ids = tn.nodes.iloc[ix].node_id.values

    segs = [s for s in tn.segments if any(np.isin(ids, s))]
    segs = sorted(segs, key=lambda x: len(x))

    to_drop = np.array([n for s in segs[:-1] for n in s])
    to_drop = to_drop[~np.isin(to_drop, segs[-1] + [soma.name])]

    navis.remove_nodes(tn, to_drop, inplace=True)

def _assert_id_match(tn: navis.TreeNeuron, root_id: int, client: CAVEclient) -> None:
    """Verify that skeleton nodes map to the correct segment ID."""
    if root_id == 0:
        raise ValueError('Segmentation ID must not be 0')

    coords = tn.nodes[['x', 'y', 'z']].values

    try:
        new_ids = client.chunkedgraph.get_roots(coords)
        if not np.all(new_ids == root_id):
            raise ValueError(f'Skeleton nodes do not map to correct segment ID {root_id}')
    except Exception as e:
        warnings.warn(f'Failed to verify segment IDs: {e}')

def _worker_wrapper(x: Tuple[Callable, List[Any], Dict[str, Any]]) -> Union[navis.TreeNeuron, int]:
    """Helper function for parallel processing."""
    f, args, kwargs = x
    if len(args) >= 2 and 'client' in kwargs:
        del kwargs['client']
    try:
        result = f(*args, **kwargs)
        if result is None:
            return args[1]
        return result
    except KeyboardInterrupt:
        raise
    except Exception as e:
        warnings.warn(f'Failed to process neuron {args[1]}: {e}')
        return args[1]

def _create_node_info_dict(vertices: NDArray, edges: NDArray) -> Dict[int, Dict[str, Any]]:
    """Create node info dictionary for SWC format."""
    node_info = {}
    parent_map = {}

    for i, coord in enumerate(vertices):
        node_info[i] = {
            'PointNo': i + 1,
            'Type': 0,
            'X': float(coord[0]),
            'Y': float(coord[1]),
            'Z': float(coord[2]),
            'Radius': 1.0,
            'Parent': -1
        }

    child_nodes = set()
    parent_nodes_in_edges = set()
    for edge in edges:
        parent, child = edge
        parent_map[child] = parent
        child_nodes.add(child)
        parent_nodes_in_edges.add(parent)

        if node_info[parent]['Type'] == 0:
            node_info[parent]['Type'] = 3
        if node_info[child]['Type'] == 0:
            node_info[child]['Type'] = 3

    for child, parent in parent_map.items():
        node_info[child]['Parent'] = parent + 1

    all_nodes = set(node_info.keys())
    root_nodes = all_nodes - child_nodes
    for root in root_nodes:
        node_info[root]['Type'] = 1
        node_info[root]['Parent'] = -1

    for node_idx, info in node_info.items():
        if node_idx in child_nodes and node_idx not in parent_nodes_in_edges:
            node_info[node_idx]['Type'] = 6

    return node_info

def _swc_dict_to_dataframe(node_info: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    """Convert node info dictionary to SWC DataFrame."""
    df = pd.DataFrame.from_dict(node_info, orient='index')
    df = df[['PointNo', 'Type', 'X', 'Y', 'Z', 'Radius', 'Parent']]
    df = df.sort_values('PointNo')
    for col in ['X', 'Y', 'Z', 'Radius']:
        df[col] = df[col].astype(float)
    return df

def get_skeletons(
    root_ids: Union[List[int], NDArray],
    dataset: str = 'kronauer_ant',
    progress: bool = True
) -> navis.NeuronList:
    """Get skeletons for multiple neurons from the dataset."""
    client = create_client(datastack_name=dataset)
    
    skeletons = []
    for root_id in navis.config.tqdm(root_ids, desc='Fetching skeletons', disable=not progress):
        try:
            try:
                skel = client.skeleton.get(root_id)
                if skel is not None:
                    vertices = np.array(skel.vertices, dtype=float)
                    edges = np.array(skel.edges, dtype=int)
                    
                    node_info = _create_node_info_dict(vertices, edges)
                    df = _swc_dict_to_dataframe(node_info)
                    
                    tn = navis.TreeNeuron(df, id=root_id, units='1 nm')
                    skeletons.append(tn)
                    continue
            except:
                pass
            
            skel = skeletonize_neuron(client, root_id, progress=False)
            if skel is not None:
                skeletons.append(skel)
            
        except Exception as e:
            warnings.warn(f"Failed to fetch skeleton for {root_id}: {e}")
            continue
            
    return navis.NeuronList(skeletons)