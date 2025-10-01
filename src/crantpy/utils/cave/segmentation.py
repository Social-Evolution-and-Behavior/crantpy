# -*- coding: utf-8 -*-
"""
This module provides utility functions for cave-related segmentation operations.
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
        # Check length match
        if len(raw_supervoxels) != len(root_ids):
            logging.warning(
                f"Number of supervoxels ({len(raw_supervoxels)}) does not match root IDs ({len(root_ids)}). Ignoring supervoxels."
            )
            raw_supervoxels = None
        else:
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
