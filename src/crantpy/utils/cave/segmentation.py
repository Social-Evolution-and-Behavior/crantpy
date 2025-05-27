# -*- coding: utf-8 -*-
"""
This module provides utility functions for cave-related segmentation operations.
"""
import os
import time
import numpy as np
from diskcache import Cache
from typing import Optional, Union

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
from crantpy.utils.cave.helpers import parse_root_ids
from crantpy.utils.helpers import retry, make_iterable, parse_timestamp
from crantpy.utils.types import IDs, Neurons, Timestamp

@inject_dataset(allowed=CRANT_VALID_DATASETS)
def roots_to_supervoxels(
    neurons: Neurons,
    clear_cache: bool = False,
    progress: bool = True,
    *,
    dataset: Optional[str] = None
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
    else:
        timestamp = parse_timestamp(timestamp)

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
