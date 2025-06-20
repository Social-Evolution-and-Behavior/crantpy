# -*- coding: utf-8 -*-
"""Visualization module for CRANTBpy."""


import functools
import logging
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Optional,
                    Type, TypeVar, Union, cast)

import numpy as np
import pandas as pd
from numpy.typing import NDArray

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


# Function to get L2 info 
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

    # Sum the attributes across all L2 chunks and convert units to micrometers
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