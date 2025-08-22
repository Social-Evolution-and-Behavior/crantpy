# -*- coding: utf-8 -*-
"""Visualization module for CRANTBpy."""



import functools
import logging
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Optional,
                    Type, TypeVar, Union, cast)

import numpy as np
import pandas as pd
import navis
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import networkx as nx
import fastremap
import skeletor as sk
from crantpy.utils.cave import get_cave_client, get_cloudvolume
import trimesh as tm
from tqdm import tqdm

from crantpy.utils.config import CRANT_VALID_DATASETS, SCALE_X, SCALE_Y, SCALE_Z
from crantpy.utils.decorators import inject_dataset, parse_neuroncriteria
from crantpy.queries.neurons import NeuronCriteria 


@inject_dataset(allowed=CRANT_VALID_DATASETS)
@parse_neuroncriteria()
def get_l2_info(
    neurons: Union[int, str, List[Union[int, str]], 'NeuronCriteria'],
    dataset: Optional[str] = None,
    progress: bool = True,
    max_threads=4
    ) -> pd.DataFrame:
    """Fetch basic info for given neuron(s) using the L2 cache.

    Parameters
    ----------
    neurons : int, str, list or NeuronCriteria
        Neurons to fetch info for. Can be a single root ID, a list of root IDs,
        or an instance of NeuronCriteria.
    dataset : str, optional
        Dataset to fetch info from.
    progress : bool
        Whether to show a progress bar.
    max_threads : int
        Number of parallel requests to make.

    Returns
    -------
    pandas.DataFrame
        DataFrame with basic L2 information for the given neuron(s). 
        - `length_um` is the underestimated sum of the max diameter across all L2 chunks 
        - `bounds_nm` is the rough bounding box based on the representative coordinates of the L2 chunks 
        - `chunks_missing` is the number of L2 chunks not present in the L2 cache.
    """
    # Convert neurons to root_ids (np.ndarray) using NeuronCriteria if needed
    if hasattr(neurons, 'get_roots'):
        root_ids = neurons.get_roots()
    elif isinstance(neurons, (int, str)):
        root_ids = np.array([neurons])
    elif isinstance(neurons, (list, np.ndarray)):
        root_ids = np.array(neurons)
    else:
        logging.error("Invalid input type for 'neurons' in get_l2_info. Must be int, str, list, np.ndarray, or NeuronCriteria.")
        raise ValueError("Invalid input type for neurons. Must be int, str, list, np.ndarray, or NeuronCriteria.")

    # Ensure all root IDs are strings for consistency
    root_ids = np.array([str(rid) for rid in root_ids])

    # If multiple root_ids, parallelize
    if len(root_ids) > 1:
        root_ids = np.unique(root_ids)
        info = []
        with ThreadPoolExecutor(max_workers=max_threads) as pool:
            func = partial(get_l2_info, dataset=dataset)
            futures = {pool.submit(func, rid): rid for rid in root_ids}
            info = [
                future.result()
                for future in tqdm(
                    as_completed(futures),
                    desc="Fetching L2 info",
                    total=len(root_ids),
                    disable=not progress or len(root_ids) == 1,
                    leave=False,
                )
            ]
        return pd.concat(info, axis=0).reset_index(drop=True)

    # Get/Initialize the CAVE client
    client = get_cave_client(dataset=dataset)

    # Directly call client methods (no retry logic)
    l2_ids = client.chunkedgraph.get_leaves(root_ids[0], stop_layer=2)

    attributes = ["area_nm2", "size_nm3", "max_dt_nm", "rep_coord_nm"]
    info = client.l2cache.get_l2data(l2_ids.tolist(), attributes=attributes)
    # n_miss: number of L2 chunks for which no data was returned from the L2 cache
    n_miss = len([v for v in info.values() if not v])

    row = [root_ids[0], len(l2_ids), n_miss]
    info_df = pd.DataFrame([row], columns=["root_id", "l2_chunks", "chunks_missing"])

    # Convert attributes to micrometers and sum them up
    for at in attributes:
        if at in ("rep_coord_nm",):
            continue
        summed = sum([v.get(at, 0) for v in info.values()])
        if at.endswith("3"):
            summed /= 1000**3
        elif at.endswith("2"):
            summed /= 1000**2
        else:
            summed /= 1000
        info_df[at.replace("_nm", "_um")] = [summed]

    # Calculate the bounding box in nanometers 
    pts = np.array([v["rep_coord_nm"] for v in info.values() if v])
    if len(pts) > 1:
        bounds = [v for l in zip(pts.min(axis=0), pts.max(axis=0)) for v in l]
    elif len(pts) == 1:
        pt = pts[0]
        rad = [v["max_dt_nm"] for v in info.values() if v][0] / 2
        bounds = [
            pt[0] - rad,
            pt[0] + rad,
            pt[1] - rad,
            pt[1] + rad,
            pt[2] - rad,
            pt[2] + rad,
        ]
        bounds = [int(co) for co in bounds]
    else:
        bounds = None
    info_df["bounds_nm"] = [bounds]

    info_df.rename({"max_dt_um": "length_um"}, axis=1, inplace=True)

    return info_df



@parse_neuroncriteria()
@inject_dataset(allowed=CRANT_VALID_DATASETS)
def get_l2_chunk_info(
    neurons: Union[int, str, List[Union[int, str]], 'NeuronCriteria'],
    dataset: Optional[str] = None,
    progress: bool = True,
    chunk_size: int = 2000
    ) -> pd.DataFrame:
    """Fetch info for L2 chunks associated with given neuron(s).

    Parameters
    ----------
    neurons : int, str, list, np.ndarray, or NeuronCriteria
        Neurons to fetch L2 chunk info for. Can be a single root ID, a list of root IDs,
        or an instance of NeuronCriteria.
    dataset : str, optional
        Dataset to fetch info from.
    progress : bool
        Whether to show a progress bar.
    chunk_size : int
        Number of L2 IDs per query.

    Returns
    -------
    pandas.DataFrame
        DataFrame with L2 chunk info (coordinates, vectors, size).
    """
    # Convert neurons to a flat np.ndarray of root IDs (as strings)
    if hasattr(neurons, 'get_roots'):
        root_ids = neurons.get_roots()
    elif isinstance(neurons, (int, str)):
        root_ids = np.array([neurons])
    elif isinstance(neurons, (list, np.ndarray)):
        root_ids = np.array(neurons)
    else:
        logging.error("Invalid input type for 'neurons' in get_l2_chunk_info. Must be int, str, list, np.ndarray, or NeuronCriteria.")
        raise ValueError("Invalid input type for neurons. Must be int, str, list, np.ndarray, or NeuronCriteria.")
    root_ids = np.array([str(rid) for rid in root_ids])

    # Get/Initialize the CAVE client
    client = get_cave_client(dataset=dataset)

    # Use DataFrames from the start: each row is a single L2 chunk, with root_id column
    attributes = ['rep_coord_nm', 'pca', 'size_nm3']
    dfs = []
    for rid in root_ids:
        l2_ids = client.chunkedgraph.get_leaves(str(rid), stop_layer=2)
        l2_ids = [str(l2id) for l2id in l2_ids]
        if not l2_ids:
            continue
        l2_info = {}
        with navis.config.tqdm(desc=f'Fetching L2 chunk info for root {rid}',
                               disable=not progress,
                               total=len(l2_ids),
                               leave=False) as pbar:
            for chunk_ix in np.arange(0, len(l2_ids), chunk_size):
                chunk = l2_ids[chunk_ix: chunk_ix + chunk_size]
                l2_info.update(client.l2cache.get_l2data(chunk, attributes=attributes))
                pbar.update(len(chunk))
        l2_info = {k: v for k, v in l2_info.items() if v}
        if not l2_info:
            continue
        # Build DataFrame for this root
        l2_keys = list(l2_info.keys())
        pts = np.vstack([i['rep_coord_nm'] for i in l2_info.values()])
        vec = np.vstack([i.get('pca', [[None, None, None]])[0] for i in l2_info.values()])
        sizes = np.array([i['size_nm3'] for i in l2_info.values()])
        df = pd.DataFrame({
            'root_id': rid,
            'l2_id': l2_keys,
            'x': (pts[:, 0] / 4).astype(int),
            'y': (pts[:, 1] / 4).astype(int),
            'z': (pts[:, 2] / 40).astype(int),
            'vec_x': vec[:, 0],
            'vec_y': vec[:, 1],
            'vec_z': vec[:, 2],
            'size_nm3': sizes
        })
        dfs.append(df)
    if dfs:
        info_df = pd.concat(dfs, axis=0, ignore_index=True)
    else:
        info_df = pd.DataFrame(columns=['root_id', 'l2_id', 'x', 'y', 'z', 'vec_x', 'vec_y', 'vec_z', 'size_nm3'])
    return info_df



@parse_neuroncriteria()
@inject_dataset(allowed=CRANT_VALID_DATASETS)
def find_anchor_loc(
    neurons: Union[int, str, List[Union[int, str]], 'NeuronCriteria'],
    dataset: Optional[str] = None,
    max_threads: int = 4,
    progress: bool = True
    ) -> pd.DataFrame:
    """Find a representative coordinate for neuron(s) using the L2 cache.

    This works by querying the L2 cache and using the representative coordinate
    for the largest L2 chunk for each neuron.

    Parameters
    ----------
    neurons : int, str, list, np.ndarray, or NeuronCriteria
        Root ID(s) to get coordinate for. Can be a single root ID, a list of root IDs,
        or an instance of NeuronCriteria.
    dataset : str, optional
        Dataset to query. If None, will use the default dataset.
    max_threads : int
        Number of parallel threads to use for batch queries.
    progress : bool
        Whether to show a progress bar.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: root_id, x, y, z.
    """
    # Convert neurons to root_ids (np.ndarray) using NeuronCriteria if needed
    if hasattr(neurons, 'get_roots'):
        root_ids = neurons.get_roots()
    elif isinstance(neurons, (int, str)):
        root_ids = np.array([neurons])
    elif isinstance(neurons, (list, np.ndarray)):
        root_ids = np.array(neurons)
    else:
        logging.error("Invalid input type for 'neurons' in find_anchor_loc. Must be int, str, list, np.ndarray, or NeuronCriteria.")
        raise ValueError("Invalid input type for neurons. Must be int, str, list, np.ndarray, or NeuronCriteria.")
    root_ids = np.array([str(rid) for rid in root_ids])

    # Batch mode: multiple root IDs
    if len(root_ids) > 1:
        root_ids_unique = np.unique(root_ids)
        info = []
        with ThreadPoolExecutor(max_workers=max_threads) as pool:
            func = partial(find_anchor_loc, dataset=dataset, progress=False)
            futures = pool.map(func, root_ids_unique)
            info = [f for f in navis.config.tqdm(futures,
                                                 desc='Fetching locations',
                                                 total=len(root_ids_unique),
                                                 disable=not progress or len(root_ids_unique) == 1,
                                                 leave=False)]
        df = pd.concat(info, axis=0, ignore_index=True)
        # Retain original order
        df = df.set_index('root_id').loc[root_ids].reset_index(drop=False)
        return df

    # Single root ID
    root_id = root_ids[0]
    info = get_l2_chunk_info(root_id, dataset=dataset, progress=progress)
    if info.empty:
        loc = [None, None, None]
    else:
        info.sort_values('size_nm3', ascending=False, inplace=True)
        loc = info[['x', 'y', 'z']].values[0].tolist()

    df = pd.DataFrame([[root_id] + loc], columns=['root_id', 'x', 'y', 'z'])
    return df



@parse_neuroncriteria()
@inject_dataset(allowed=CRANT_VALID_DATASETS)
def get_l2_graph(
    root_ids: Union[int, str, List[Union[int, str]], 'NeuronCriteria'],
    dataset: Optional[str] = None,
    progress: bool = True,
    max_threads: int = 4
    ) -> Union['nx.Graph', List['nx.Graph']]:
    """Fetch L2 graph(s) for given neuron(s).

    Parameters
    ----------
    root_ids : int, str, list, np.ndarray, or NeuronCriteria
        FlyWire root ID(s) for which to fetch the L2 graphs.
    dataset : str, optional
        Against which FlyWire dataset to query. If None, will use the default dataset.
    progress : bool
        Whether to show a progress bar.
    max_threads : int
        Number of parallel threads to use for batch queries.

    Returns
    -------
    networkx.Graph or list of networkx.Graph
        The L2 graph or list thereof.
    """
    # Convert root_ids to a flat np.ndarray of root IDs (as strings)
    if hasattr(root_ids, 'get_roots'):
        ids = root_ids.get_roots()
    elif isinstance(root_ids, (int, str)):
        ids = np.array([root_ids])
    elif isinstance(root_ids, (list, np.ndarray)):
        ids = np.array(root_ids)
    else:
        logging.error("Invalid input type for 'root_ids' in get_l2_graph. Must be int, str, list, np.ndarray, or NeuronCriteria.")
        raise ValueError("Invalid input type for root_ids. Must be int, str, list, np.ndarray, or NeuronCriteria.")
    ids = np.array([str(rid) for rid in ids])

    # Batch mode: multiple root IDs
    if len(ids) > 1:
        ids_unique = np.unique(ids)
        graphs = []
        with ThreadPoolExecutor(max_workers=max_threads) as pool:
            func = partial(get_l2_graph, dataset=dataset, progress=False)
            futures = pool.map(func, ids_unique)
            graphs = [f for f in navis.config.tqdm(futures,
                                                   desc='Fetching L2 graphs',
                                                   total=len(ids_unique),
                                                   disable=not progress or len(ids_unique) == 1,
                                                   leave=False)]
        return graphs

    # Single root ID
    root_id = ids[0]
    client = get_cave_client(dataset=dataset)

    # Load the L2 graph for given root ID
    try:
        l2_eg = np.array(client.chunkedgraph.level2_chunk_graph(root_id))
    except Exception as e:
        logging.error(f"Failed to get L2 graph for root_id {root_id}: {e}")
        return None

    # Generate graph
    G = nx.Graph()

    if not len(l2_eg):
        # If no edges, this neuron consists of a single chunk
        # Get the single chunk's ID
        try:
            chunks = client.chunkedgraph.get_leaves(root_id, stop_layer=2)
            G.add_nodes_from(chunks)
        except Exception as e:
            logging.error(f"Failed to get L2 leaves for root_id {root_id}: {e}")
            return None
    else:
        # Drop duplicate edges
        l2_eg = np.unique(np.sort(l2_eg, axis=1), axis=0)
        G.add_edges_from(l2_eg)

    return G



@parse_neuroncriteria()
@inject_dataset(allowed=CRANT_VALID_DATASETS)
def get_l2_skeleton(
    root_ids: Union[int, str, List[Union[int, str]], 'NeuronCriteria'],
    refine: bool = True,
    drop_missing: bool = True,
    l2_node_ids: bool = False,
    omit_failures: Optional[bool] = None,
    progress: bool = True,
    max_threads: int = 4,
    dataset: Optional[str] = None,
    **kwargs
    ) -> Union['navis.TreeNeuron', 'navis.NeuronList']:
    """Generate skeleton(s) from L2 graph(s) for given neuron(s).

    Parameters
    ----------
    root_ids : int, str, list, np.ndarray, or NeuronCriteria
        Root ID(s) of the FlyWire neuron(s) to skeletonize.
    refine : bool
        If True, refine skeleton nodes by moving them to the center of their corresponding chunk meshes using the L2 cache.
    drop_missing : bool
        Only relevant if ``refine=True``: If True, drop chunks that don't exist in the L2 cache.
    l2_node_ids : bool
        If True, use the L2 IDs as node IDs.
    omit_failures : bool, optional
        Behaviour when skeleton generation fails. None (default) raises, True skips, False returns empty TreeNeuron.
    progress : bool
        Whether to show a progress bar.
    max_threads : int
        Number of parallel requests to make when fetching the L2 skeletons.
    dataset : str, optional
        Against which FlyWire dataset to query. If None, will use the default dataset.
    **kwargs
        Additional keyword arguments passed to TreeNeuron initialization.

    Returns
    -------
    navis.TreeNeuron or navis.NeuronList
        The extracted L2 skeleton(s).
    """
    if omit_failures not in (None, True, False):
        raise ValueError('`omit_failures` must be either None, True or False. '
                         f'Got "{omit_failures}".')

    # Convert root_ids to a flat np.ndarray of root IDs (as strings)
    if hasattr(root_ids, 'get_roots'):
        ids = root_ids.get_roots()
    elif isinstance(root_ids, (int, str)):
        ids = np.array([root_ids])
    elif isinstance(root_ids, (list, np.ndarray)):
        ids = np.array(root_ids)
    else:
        logging.error("Invalid input type for 'root_ids' in get_l2_skeleton. Must be int, str, list, np.ndarray, or NeuronCriteria.")
        raise ValueError("Invalid input type for root_ids. Must be int, str, list, np.ndarray, or NeuronCriteria.")
    ids = np.array([str(rid) for rid in ids])

    

    # Batch mode: multiple root IDs
    if len(ids) > 1:
        logging.info(f'Fetching L2 skeletons for {len(ids)} root IDs.')
        ids_unique = np.unique(ids)
        with ThreadPoolExecutor(max_workers=max_threads) as pool:
            futures = pool.map(lambda rid: get_l2_skeleton(rid, refine=refine, drop_missing=drop_missing,
                                                  l2_node_ids=l2_node_ids, omit_failures=omit_failures,
                                                  dataset=dataset, progress=False, **kwargs),
                       ids_unique.tolist())
            results = [f for f in navis.config.tqdm(futures,
                                                    desc='Fetching L2 skeletons',
                                                    total=len(ids_unique),
                                                    disable=not progress or len(ids_unique) == 1,
                                                    leave=False)]
        # Combine into a NeuronList
        nl = navis.NeuronList(results)
        return nl

    # Single root ID
    try:
        root_id = np.int64(ids[0])
    except (ValueError, TypeError) as e:
        logging.error(f"Invalid root ID '{ids[0]}': cannot convert to integer. Error: {e}")
        raise ValueError(f"Invalid root ID '{ids[0]}': must be an integer or string representing an integer.") from e
    logging.info(f"Fetching L2 skeleton for root ID: {root_id}")

    # Get/Initialize the CAVE client
    client = get_cave_client(dataset=dataset)

    # Get the cloudvolume
    vol = get_cloudvolume()

    # Load the L2 graph for given root ID (this is a (N, 2) array of edges)
    try:
        l2_eg = client.chunkedgraph.level2_chunk_graph(root_id)
    except Exception as e:
        logging.error(f"Failed to get L2 edges for root_id {root_id}: {e}")
        msg = (f'Unable to create L2 skeleton: root ID {root_id} failed to fetch edges.')
        if omit_failures is None:
            raise ValueError(msg)
        elif omit_failures:
            return navis.NeuronList([])
        else:
            return navis.TreeNeuron(None, id=root_id, units='1 nm', **kwargs)

    # If no edges, we can't create a skeleton
    if not len(l2_eg):
        msg = (f'Unable to create L2 skeleton: root ID {root_id} '
               'consists of only a single L2 chunk.')
        if omit_failures is None:
            raise ValueError(msg)
        elif omit_failures:
            return navis.NeuronList([])
        else:
            return navis.TreeNeuron(None, id=root_id, units='1 nm', **kwargs)

    # Drop duplicate edges
    l2_eg = np.unique(np.sort(l2_eg, axis=1), axis=0)

    # Unique L2 IDs
    l2_ids = np.unique(l2_eg)

    # ID to index
    l2dict = {l2: ii for ii, l2 in enumerate(l2_ids)}

    # Remap edge graph to indices
    eg_arr_rm = fastremap.remap(l2_eg, l2dict)

    coords = [np.array(vol.mesh.meta.meta.decode_chunk_position(l2)) for l2 in l2_ids]
    coords = np.vstack(coords)

    # This turns the graph into a hierarchal tree by removing cycles and
    # ensuring all edges point towards a root
    if hasattr(sk, "skeletonizers"):
        G = sk.skeletonizers.edges_to_graph(eg_arr_rm)
        swc = sk.skeletonizers.make_swc(G, coords=coords)
    else:
        G = sk.skeletonize.utils.edges_to_graph(eg_arr_rm)
        swc = sk.skeletonize.utils.make_swc(G, coords=coords, reindex=False)

    # Set radius to 0
    swc['radius'] = 0

    # Convert to Euclidian space
    # Dimension of a single chunk
    ch_dims = chunks_to_nm([1, 1, 1], vol) - chunks_to_nm([0, 0, 0], vol)
    ch_dims = np.squeeze(ch_dims)

    xyz = swc[['x', 'y', 'z']].values
    swc[['x', 'y', 'z']] = chunks_to_nm(xyz, vol) + ch_dims / 2

    if refine:
        # Get the L2 representative coordinates
        l2_info = client.l2cache.get_l2data(l2_ids.tolist(), attributes=['rep_coord_nm', 'max_dt_nm'])
        # Missing L2 chunks will be {'id': {}}
        new_co = {l2dict[np.int64(k)]: v['rep_coord_nm'] for k, v in l2_info.items() if v}
        new_r = {l2dict[np.int64(k)]: v.get('max_dt_nm', 0) for k, v in l2_info.items() if v}

        # Map refined coordinates onto the SWC
        has_new = swc.node_id.isin(new_co)

        # Only apply if we actually have new coordinates - otherwise there
        # the datatype is changed to object for some reason...
        if any(has_new):
            swc.loc[has_new, 'x'] = swc.loc[has_new, 'node_id'].map(lambda x: new_co[x][0])
            swc.loc[has_new, 'y'] = swc.loc[has_new, 'node_id'].map(lambda x: new_co[x][1])
            swc.loc[has_new, 'z'] = swc.loc[has_new, 'node_id'].map(lambda x: new_co[x][2])

        swc['radius'] = swc.node_id.map(new_r)

        # Turn into a proper neuron
        tn = navis.TreeNeuron(swc, id=root_id, units='1 nm', **kwargs)

        # Drop nodes that are still at their unrefined chunk position
        if drop_missing:
            frac_refined = has_new.sum() / len(has_new)
            if not any(has_new):
                msg = (f'Unable to refine: no L2 info for root ID {root_id} '
                       'available. Set `drop_missing=False` to use unrefined '
                       'positions.')
                if omit_failures is None:
                    raise ValueError(msg)
                elif omit_failures:
                    return navis.NeuronList([])
                else:
                    return navis.TreeNeuron(None, id=root_id, units='1 nm', **kwargs)
            elif frac_refined < .5:
                msg = (f'Root ID {root_id} has only {frac_refined:.1%} of their '
                       'L2 IDs in the cache. Set `drop_missing=False` to use '
                       'unrefined positions.')
                navis.config.logger.warning(msg)

            tn = navis.remove_nodes(tn, swc.loc[~has_new, 'node_id'].values)
            tn._l2_chunks_missing = (~has_new).sum()
    else:
        tn = navis.TreeNeuron(swc, id=root_id, units='1 nm', **kwargs)

    if l2_node_ids:
        ixdict = {ii: l2 for ii, l2 in enumerate(l2_ids)}
        tn.nodes['node_id'] = tn.nodes.node_id.map(ixdict)
        tn.nodes['parent_id'] = tn.nodes.parent_id.map(lambda x: ixdict.get(x, -1))

    return tn


@parse_neuroncriteria()
@inject_dataset()
def get_l2_dotprops(
    root_ids: Union[int, str, List[Union[int, str]], 'NeuronCriteria'],
    min_size: Optional[int] = None,
    sample: Optional[float] = False,
    omit_failures: Optional[bool] = None,
    progress: bool = True,
    max_threads: int = 10,
    dataset: Optional[str] = None,
    **kwargs
    ) -> 'navis.NeuronList':
    """Generate dotprops from L2 chunks for given neuron(s).

    Parameters
    ----------
    root_ids : int, str, list, np.ndarray, or NeuronCriteria
        Root ID(s) of the FlyWire neuron(s) to generate dotprops for.
    min_size : int, optional
        Minimum size (in nm^3) for the L2 chunks. Smaller chunks will be ignored.
    sample : float [0 > 1], optional
        If float, will create Dotprops based on a fractional sample of the L2 chunks.
    omit_failures : bool, optional
        Behaviour when dotprops generation fails. None (default) raises, True skips, False returns empty Dotprops.
    progress : bool
        Whether to show a progress bar.
    max_threads : int
        Number of parallel requests to make when fetching the L2 IDs.
    dataset : str, optional
        Against which FlyWire dataset to query. If None, will use the default dataset.
    **kwargs
        Additional keyword arguments passed to Dotprops initialization.

    Returns
    -------
    navis.NeuronList
        List of Dotprops.
    """
    if omit_failures not in (None, True, False):
        raise ValueError('`omit_failures` must be either None, True or False. '
                         f'Got "{omit_failures}".')

    # Convert root_ids to a flat np.ndarray of root IDs (as strings)
    if hasattr(root_ids, 'get_roots'):
        ids = root_ids.get_roots()
    elif isinstance(root_ids, (int, str)):
        ids = np.array([root_ids])
    elif isinstance(root_ids, (list, np.ndarray)):
        ids = np.array(root_ids)
    else:
        logging.error("Invalid input type for 'root_ids' in get_l2_dotprops. Must be int, str, list, np.ndarray, or NeuronCriteria.")
        raise ValueError("Invalid input type for root_ids. Must be int, str, list, np.ndarray, or NeuronCriteria.")
    ids = np.array([str(rid) for rid in ids])

    # Get/Initialize the CAVE client
    client = get_cave_client(dataset=dataset)

    # Load the L2 IDs
    with ThreadPoolExecutor(max_workers=max_threads) as pool:
        get_l2_ids = partial(client.chunkedgraph.get_leaves, stop_layer=2)
        futures = pool.map(get_l2_ids, ids)
        l2_ids = [f for f in navis.config.tqdm(futures,
                                               desc='Fetching L2 IDs',
                                               total=len(ids),
                                               disable=not progress or len(ids) == 1,
                                               leave=False)]

    # Turn IDs into strings
    l2_ids = [np.array(i).astype(str) for i in l2_ids]

    if sample:
        if sample <= 0 or sample >= 1:
            raise ValueError(f'`sample` must be between 0 and 1, got {sample}')
        for i in range(len(l2_ids)):
            np.random.seed(42)
            l2_ids[i] = np.random.choice(l2_ids[i],
                                         size=max(1, int(len(l2_ids[i]) * sample)),
                                         replace=False)

    # Flatten into a list of all L2 IDs
    l2_ids_all = np.unique([i for l in l2_ids for i in l])

    # Get the L2 representative coordinates, vectors and (if required) volume
    chunk_size = 2000
    attributes = ['rep_coord_nm', 'pca']
    if min_size:
        attributes.append('size_nm3')

    l2_info = {}
    with navis.config.tqdm(desc='Fetching L2 vectors',
                           disable=not progress,
                           total=len(l2_ids_all),
                           leave=False) as pbar:
        for chunk_ix in np.arange(0, len(l2_ids_all), chunk_size):
            chunk = l2_ids_all[chunk_ix: chunk_ix + chunk_size]
            l2_info.update(client.l2cache.get_l2data(chunk.tolist(), attributes=attributes))
            pbar.update(len(chunk))

    # L2 chunks without info will show as empty dictionaries
    l2_info = {k: v for k, v in l2_info.items() if 'pca' in v}

    # Generate dotprops
    dps = []
    for root, ids_ in navis.config.tqdm(zip(ids, l2_ids),
                                       desc='Creating dotprops',
                                       total=len(ids),
                                       disable=not progress or len(ids) <= 1,
                                       leave=False):
        this_info = [l2_info[i] for i in ids_ if i in l2_info]
        if not len(this_info):
            msg = ('Unable to create L2 dotprops: none of the L2 chunks for '
                   f'root ID {root} are present in the L2 cache.')
            if omit_failures is None:
                raise ValueError(msg)
            if not omit_failures:
                dps.append(navis.Dotprops(None, k=None, id=root,
                                          units='1 nm', **kwargs))
                dps[-1]._l2_chunks_missing = len(ids_)
            continue
        pts = np.vstack([i['rep_coord_nm'] for i in this_info])
        vec = np.vstack([i['pca'][0] for i in this_info])
        if min_size:
            sizes = np.array([i['size_nm3'] for i in this_info])
            pts = pts[sizes >= min_size]
            vec = vec[sizes >= min_size]
        dps.append(navis.Dotprops(points=pts, vect=vec, id=root, k=None,
                                  units='1 nm', **kwargs))
        dps[-1]._l2_chunks_missing = len(ids_) - len(this_info)
    return navis.NeuronList(dps)


@inject_dataset(allowed=CRANT_VALID_DATASETS)
def get_l2_meshes(
    x: Union[int, str, 'NeuronCriteria'],
    threads: int = 10,
    progress: bool = True,
    dataset: Optional[str] = None
    ) -> 'navis.NeuronList':
    """Fetch L2 meshes for a single neuron or NeuronCriteria in CRANTb.

    Parameters
    ----------
    x : int | str | NeuronCriteria
        Root ID or NeuronCriteria. Must not be a list.
    threads : int
    progress : bool
    dataset : str, optional
        Against which CRANTb dataset to query. If None, will use the default dataset.

    Returns
    -------
    navis.NeuronList
    """
    # Accept NeuronCriteria, int, or str, but not a list/array
    if hasattr(x, 'get_roots'):
        root_ids = x.get_roots()
        if isinstance(root_ids, (list, np.ndarray)) and len(root_ids) != 1:
            raise ValueError("get_l2_meshes only accepts a single NeuronCriteria or root ID, not a list.")
        x = root_ids[0] if isinstance(root_ids, (list, np.ndarray)) else root_ids
    try:
        x = np.int64(x)
    except ValueError:
        raise ValueError(f'Unable to convert root ID {x} to integer')
    client = get_cave_client(dataset=dataset)
    vol = get_cloudvolume()
    l2_ids = client.chunkedgraph.get_leaves(x, stop_layer=2)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(vol.mesh.get, i,
                               allow_missing=True,
                               deduplicate_chunk_boundaries=False) for i in l2_ids]
        res = [f.result() for f in navis.config.tqdm(futures,
                                                     disable=not progress,
                                                     leave=False,
                                                     desc='Loading meshes')]
    meshes = {k: v for d in res for k, v in d.items()}
    return navis.NeuronList([navis.MeshNeuron(v, id=k) for k, v in meshes.items()])


def _get_l2_centroids(
    l2_ids: List[str],
    vol,
    threads: int = 10,
    progress: bool = True
    ) -> Dict[str, np.ndarray]:
    """Fetch L2 meshes and compute centroid for each mesh.

    Parameters
    ----------
    l2_ids : list of str
        List of L2 chunk IDs.
    vol : cloudvolume.CloudVolume
        CloudVolume object associated with the chunked space.
    threads : int, optional
        Number of parallel threads to use.
    progress : bool, optional
        Whether to show a progress bar.

    Returns
    -------
    dict
        Dictionary mapping L2 IDs to centroid coordinates (np.ndarray).
    """
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(vol.mesh.get, i,
                               allow_missing=True,
                               deduplicate_chunk_boundaries=False) for i in l2_ids]
        res = [f.result() for f in navis.config.tqdm(futures,
                                                     disable=not progress,
                                                     leave=False,
                                                     desc='Loading meshes')]
    meshes = {k: v for d in res for k, v in d.items()}
    centroids = {}
    for k, m in meshes.items():
        m = tm.Trimesh(m.vertices, m.faces)
        centroids[k] = m.centroid
    return centroids


def chunks_to_nm(xyz_ch, vol, voxel_resolution=None):
    """Map a chunk location to Euclidean space.

    Parameters
    ----------
    xyz_ch : array-like
        (N, 3) array of chunk indices.
    vol : cloudvolume.CloudVolume
        CloudVolume object associated with the chunked space.
    voxel_resolution : list, optional
        Voxel resolution. If None, use SCALE_X, SCALE_Y, SCALE_Z.

    Returns
    -------
    np.array
        (N, 3) array of spatial points.
    """
    voxel_resolution = [SCALE_X, SCALE_Y, SCALE_Z] if voxel_resolution is None else voxel_resolution
    mip_scaling = vol.mip_resolution(0) // np.array(voxel_resolution, dtype=int)

    x_vox = np.atleast_2d(xyz_ch) * vol.mesh.meta.meta.graph_chunk_size
    return (
        (x_vox + np.array(vol.mesh.meta.meta.voxel_offset(0)))
        * voxel_resolution
        * mip_scaling
    )

