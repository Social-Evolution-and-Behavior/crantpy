# -*- coding: utf-8 -*-
"""
This module provides utility functions for cave-related helpers.
"""
import numpy as np
import requests
from typing import Iterable, Optional, Union
import navis
from datetime import datetime

from crantpy.utils.decorators import parse_neuroncriteria, inject_dataset
from crantpy.utils.helpers import make_iterable, parse_timestamp, retry
from crantpy.utils.types import Neurons, IDs, Timestamp
from crantpy.utils.config import CRANT_VALID_DATASETS
from crantpy.utils.cave import get_cave_client, get_current_cave_token, get_cloudvolume


@parse_neuroncriteria()
def parse_root_ids(x: Neurons) -> np.ndarray:
    """
    Parse root IDs from various input formats to a list of np.int64.

    Parameters
    ----------
    x : Neurons = str | int | np.int64 | navis.BaseNeuron | Iterables of previous types | navis.NeuronList | NeuronCriteria
        The input to parse. Can be a single ID, a list of IDs, or a navis neuron object.

    Returns
    -------
    np.ndarray
        A numpy array of root IDs as np.int64.
    """
    # process NeuronList
    if isinstance(x, navis.NeuronList):
        x = x.id

    # process neuron objects (single)
    if isinstance(x, navis.BaseNeuron):
        x = x.id
    elif isinstance(x, Iterable):
        # process neuron objects (iterable)
        x = [i.id if isinstance(i, navis.BaseNeuron) else i for i in x]

    # make iterable
    x = make_iterable(x, force_type=np.int64)

    return x.astype(np.int64) if len(x) > 0 else np.array([], dtype=np.int64)


@inject_dataset(allowed=CRANT_VALID_DATASETS)
def is_valid_root(
    x: IDs, dataset: Optional[str] = None, raise_exc: bool = False
) -> np.ndarray:
    """Check if ID is (potentially) valid root ID.

    Parameters
    ----------
    x : IDs = str | int | np.int64
        The root IDs to check.
    dataset : str, optional
        The dataset to use.
    raise_exc : bool, default False
        Whether to raise an exception if invalid IDs are found.

    Returns
    -------
    np.ndarray
        A boolean array indicating whether each root ID is valid.

    Raises
    ------
    ValueError
        If raise_exc is True and invalid IDs are found.
    """
    client = get_cave_client(dataset=dataset)
    vol = get_cloudvolume(dataset=dataset)

    def _is_valid(x, raise_exc):
        try:
            is_valid = vol.get_chunk_layer(x) == vol.info["graph"]["n_layers"]
        except ValueError:
            is_valid = False

        if raise_exc and not is_valid:
            raise ValueError(f"{x} is not a valid root ID")

        return is_valid

    if navis.utils.is_iterable(x):
        x = make_iterable(x, force_type=np.int64)
        is_valid = np.array([_is_valid(r, raise_exc=False) for r in x])
        if raise_exc and not all(is_valid):
            invalid = set(np.asarray(x)[~is_valid].tolist())
            raise ValueError(f"Invalid root IDs found: {invalid}")
        return is_valid
    else:
        return _is_valid(x, raise_exc=raise_exc)


@inject_dataset(allowed=CRANT_VALID_DATASETS)
def is_latest_roots(
    x: IDs,
    timestamp: Optional[Union[str, Timestamp]] = None,
    dataset: Optional[str] = None,
    progress: bool = True,
    batch_size: int = 100_000,
    validate_ids: bool = True,
    use_http_session: bool = True,
) -> np.ndarray:
    """
    Check if the given root IDs are the latest based on the timestamp.

    Parameters
    ----------
    x : IDs = str | int | np.int64
        The root IDs to check.
    timestamp : Timestamp = str | int | np.int64 | datetime | np.datetime64 | pd.Timestamp
        The timestamp to compare against. Can also be "mat" for the latest
        materialization timestamp.
    dataset : str, optional
        The dataset to use.
    progress : bool, default True
        Whether to show progress bar for large batches.
    batch_size : int, default 100_000
        Batch size for processing large numbers of IDs.
    validate_ids : bool, default True
        Whether to validate root IDs before processing.
    use_http_session : bool, default True
        Whether to use direct HTTP session for better performance.

    Returns
    -------
    np.ndarray
        A boolean array indicating whether each root ID is the latest.

    Examples
    --------
    >>> from crantpy.utils.cave.helpers import is_latest_roots
    >>> is_latest_roots([123456789, 987654321])
    array([ True, False])

    >>> # Check against latest materialization
    >>> is_latest_roots([123456789], timestamp="mat")
    array([ True])
    """
    # Ensure x is iterable and convert to proper format
    x = make_iterable(x, force_type=np.int64)

    # Handle empty input
    if len(x) == 0:
        return np.array([], dtype=bool)

    # The server doesn't like being asked for zeros - filter them out
    not_zero = x != 0

    # Initialize result array
    is_latest = np.ones(len(x), dtype=bool)

    # If no valid IDs (all zeros), return early
    if not not_zero.any():
        return np.zeros(len(x), dtype=bool)

    # Validate IDs if requested
    if validate_ids:
        is_valid_root(x[not_zero], dataset=dataset, raise_exc=True)

    # Get cave client
    client = get_cave_client(dataset=dataset)

    # Handle materialization timestamps
    if isinstance(timestamp, str) and timestamp.startswith("mat"):
        if timestamp == "mat" or timestamp == "mat_latest":
            timestamp = client.materialize.get_timestamp()
        else:
            # Split e.g. 'mat_432' to extract version and query timestamp
            try:
                version = int(timestamp.split("_")[1])
                timestamp = client.materialize.get_timestamp(version)
            except (IndexError, ValueError):
                raise ValueError(
                    f"Invalid materialization timestamp format: {timestamp}"
                )

    # Parse timestamp using the existing helper function
    if timestamp is not None:
        timestamp = int(parse_timestamp(timestamp))

    # Process in batches with progress bar
    valid_x = x[not_zero]

    if use_http_session:
        # Use direct HTTP session for better performance (similar to FlyWire)
        session = requests.Session()
        token = get_current_cave_token()
        session.headers["Authorization"] = f"Bearer {token}"
        url = client.chunkedgraph._endpoints["is_latest_roots"].format_map(
            client.chunkedgraph.default_url_mapping
        )

        # Prepare timestamp parameters for HTTP request
        params = None
        if timestamp is not None:
            # Convert Unix timestamp to datetime object for HTTP request
            if isinstance(timestamp, (int, np.int64)):
                timestamp_dt = datetime.fromtimestamp(timestamp)
            else:
                timestamp_dt = timestamp
            params = {"timestamp": timestamp_dt.isoformat()}

        with navis.config.tqdm(
            desc="Checking latest roots",
            total=len(valid_x),
            disable=(len(valid_x) <= batch_size) or not progress,
            leave=False,
        ) as pbar:
            for i in range(0, len(valid_x), batch_size):
                batch = valid_x[i : i + batch_size]

                # Update progress bar
                pbar.update(len(batch))

                # Prepare POST request
                post_data = {"node_ids": batch.tolist()}

                # Make HTTP request with retry logic
                @retry
                def make_request():
                    r = session.post(url, json=post_data, params=params)
                    r.raise_for_status()
                    return r.json()["is_latest"]

                batch_result = make_request()
                batch_result = np.array(batch_result, dtype=bool)

                # Update the result array for valid (non-zero) indices
                valid_indices = np.where(not_zero)[0][i : i + batch_size]
                is_latest[valid_indices] = batch_result
    else:
        # Use standard CAVEclient approach
        with navis.config.tqdm(
            desc="Checking latest roots",
            total=len(valid_x),
            disable=(len(valid_x) <= batch_size) or not progress,
            leave=False,
        ) as pbar:
            for i in range(0, len(valid_x), batch_size):
                batch = valid_x[i : i + batch_size]

                # Update progress bar
                pbar.update(len(batch))

                # Check if the root IDs are the latest for this batch
                is_latest_roots_func = retry(client.chunkedgraph.is_latest_roots)
                batch_result = is_latest_roots_func(batch, timestamp=timestamp)

                if not isinstance(batch_result, np.ndarray):
                    batch_result = np.array(batch_result, dtype=bool)

                # Update the result array for valid (non-zero) indices
                valid_indices = np.where(not_zero)[0][i : i + batch_size]
                is_latest[valid_indices] = batch_result

    return is_latest


@inject_dataset(allowed=CRANT_VALID_DATASETS)
def is_valid_supervoxel(x: IDs, dataset: Optional[str] = None, raise_exc: bool = False):
    """Check if ID is (potentially) valid supervoxel ID.

    Parameters
    ----------
    x : IDs = str | int | np.int64
        The supervoxel IDs to check.
    dataset : str, optional
        The dataset to use.
    raise_exc : bool, default False
        Whether to raise an exception if invalid IDs are found.

    Returns
    -------
    bool or np.ndarray
        If x is a single ID, returns bool. If x is iterable, returns array.

    Raises
    ------
    ValueError
        If raise_exc is True and invalid IDs are found.

    See Also
    --------
    is_valid_root : Use this function to check if a root ID is valid.
    """
    client = get_cave_client(dataset=dataset)
    vol = get_cloudvolume(dataset=dataset)

    def _is_valid(x, raise_exc):
        try:
            is_valid = vol.get_chunk_layer(x) == 1
        except ValueError:
            is_valid = False

        if raise_exc and not is_valid:
            raise ValueError(f"{x} is not a valid supervoxel ID")

        return is_valid

    if navis.utils.is_iterable(x):
        x = make_iterable(x, force_type=np.int64)
        is_valid = np.array([_is_valid(r, raise_exc=False) for r in x])
        if raise_exc and not all(is_valid):
            invalid = set(np.asarray(x)[~is_valid].tolist())
            raise ValueError(f"Invalid supervoxel IDs found: {invalid}")
        return is_valid
    else:
        return _is_valid(x, raise_exc=raise_exc)
