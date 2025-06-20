# -*- coding: utf-8 -*-
"""Visualization module for CRANTBpy."""


import functools
import logging
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Optional,
                    Type, TypeVar, Union, cast)

import numpy as np
import pandas as pd
import navis
from concurrent.futures import ThreadPoolExecutor
from functools import partial

#import fastremap
#import skeletor as sk
#import networkx as nx
from crantpy.utils.cave import get_cave_client

from crantpy.utils.config import CRANT_VALID_DATASETS
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
            futures = pool.map(func, root_ids)
            info = [
                f
                for f in navis.config.tqdm(
                    futures,
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

    # Gather all L2 chunk IDs for the given root IDs
    all_l2_ids = []
    for rid in root_ids:
        l2 = client.chunkedgraph.get_leaves(str(rid), stop_layer=2)
        all_l2_ids.extend(l2)
    l2_ids = np.array([str(l2id) for l2id in all_l2_ids])

    # Get the L2 representative coordinates, vectors and (if required) volume
    attributes = ['rep_coord_nm', 'pca', 'size_nm3']
    l2_info = {}
    with navis.config.tqdm(desc='Fetching L2 chunk info',
                           disable=not progress,
                           total=len(l2_ids),
                           leave=False) as pbar:
        for chunk_ix in np.arange(0, len(l2_ids), chunk_size):
            chunk = l2_ids[chunk_ix: chunk_ix + chunk_size]
            l2_info.update(client.l2cache.get_l2data(chunk.tolist(), attributes=attributes))
            pbar.update(len(chunk))

    # L2 chunks without info will show as empty dictionaries
    # Let's drop them to make our life easier (speeds up indexing too)
    l2_info = {k: v for k, v in l2_info.items() if v}

    if l2_info:
        pts = np.vstack([i['rep_coord_nm'] for i in l2_info.values()])
        vec = np.vstack([i.get('pca', [[None, None, None]])[0] for i in l2_info.values()])
        sizes = np.array([i['size_nm3'] for i in l2_info.values()])

        info_df = pd.DataFrame()
        info_df['id'] = list(l2_info.keys())
        info_df['x'] = (pts[:, 0] / 4).astype(int)
        info_df['y'] = (pts[:, 1] / 4).astype(int)
        info_df['z'] = (pts[:, 2] / 40).astype(int)
        info_df['vec_x'] = vec[:, 0]
        info_df['vec_y'] = vec[:, 1]
        info_df['vec_z'] = vec[:, 2]
        info_df['size_nm3'] = sizes
    else:
        info_df = pd.DataFrame([], columns=['id',
                                            'x', 'y', 'z',
                                            'vec_x', 'vec_y', 'vec_z',
                                            'size_nm3'])

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
    client = get_cave_client(dataset=dataset)
    try:
        l2_ids = client.chunkedgraph.get_leaves(root_id, stop_layer=2)
    except Exception as e:
        logging.error(f"Failed to get L2 leaves for root_id {root_id}: {e}")
        return pd.DataFrame([[root_id, None, None, None]], columns=['root_id', 'x', 'y', 'z'])

    info = get_l2_chunk_info(l2_ids, dataset=dataset, progress=progress)
    if info.empty:
        loc = [None, None, None]
    else:
        info.sort_values('size_nm3', ascending=False, inplace=True)
        loc = info[['x', 'y', 'z']].values[0].tolist()

    df = pd.DataFrame([[root_id] + loc], columns=['root_id', 'x', 'y', 'z'])
    return df