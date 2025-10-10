# -*- coding: utf-8 -*-
"""
This module contains helper functions for crantpy.
"""

import logging
import os
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
    cast,
    Set,
    TYPE_CHECKING,
)
import functools
import time
import warnings

import navis
import pandas as pd
import requests
import numpy as np
from collections.abc import Iterable

from crantpy.utils.types import T, F, Neurons, IDs, Timestamp
from crantpy.utils.decorators import parse_neuroncriteria

if TYPE_CHECKING:
    from crantpy.queries.neurons import NeuronCriteria

from crantpy.utils.config import ALIGNED_EM_URL
import cloudvolume as cv

# set up logging and options to change logging level
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def set_logging_level(level: str) -> None:
    """
    Sets the logging level for the logger.

    Parameters
    ----------
    level : str
        The logging level to set. Options are 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
    """
    logging_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    if level in logging_levels:
        logging.getLogger().setLevel(logging_levels[level])
    else:
        raise ValueError(
            f"Invalid logging level: {level}. Choose from {list(logging_levels.keys())}."
        )


# Custom functions
def create_sql_query(
    table_name: str,
    fields: List[str],
    condition: Optional[str] = None,
    limit: Optional[int] = None,
    start: Optional[int] = None,
) -> str:
    """
    Creates a SQL query to get the specified fields from the specified table.

    Parameters
    ----------
    table_name : str
        The name of the table to query.
    fields : List[str]
        The list of field names to include in the query.
    condition : str, optional
        The WHERE clause of the query.
    limit : int, optional
        The maximum number of rows to return.
    start : int, optional
        The number of rows to skip (OFFSET).

    Returns
    -------
    str
        The constructed SQL query string.
    """
    # Create the SQL query
    sql_query = f"SELECT {', '.join(fields)} FROM {table_name}"
    if condition:
        sql_query += f" WHERE {condition}"
    if limit is not None:
        sql_query += f" LIMIT {limit}"
    if start is not None:
        sql_query += f" OFFSET {start}"
    return sql_query


def match_dtype(value: Any, dtype: Union[str, type]) -> Any:
    """
    Match the dtype of a value to a given dtype.

    Parameters
    ----------
    value : Any
        The value to convert.
    dtype : str or type
        The target dtype to convert to.

    Returns
    -------
    Any
        The converted value.

    Raises
    ------
    ValueError
        If the dtype is not supported.
    """
    if pd.api.types.is_integer_dtype(dtype):
        return int(value)
    elif pd.api.types.is_float_dtype(dtype):
        return float(value)
    elif pd.api.types.is_bool_dtype(dtype):
        return bool(value)
    elif pd.api.types.is_string_dtype(dtype):
        return str(value)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def _check_list_membership(
    cell_value: Any, filter_value: Any, is_filter_list: bool
) -> bool:
    """Helper function to check membership within a list cell.

    Parameters
    ----------
    cell_value : Any
        The value of the cell to check.
    filter_value : Any
        The value or values to filter by.
    is_filter_list : bool
        Whether filter_value is a list.

    Returns
    -------
    bool
        True if the filter value(s) are found in the cell value, False otherwise.
    """
    if not isinstance(cell_value, list):
        return False  # Cell itself is not a list
    if not is_filter_list:
        # Filter value is a single item, check if it's in the cell's list
        return filter_value in cell_value
    else:
        # Filter value is a list, check if *any* item from filter_value is in the cell's list
        return any(item in cell_value for item in filter_value)


def filter_df(
    df: pd.DataFrame,
    column: str,
    value: Any,
    regex: bool = False,
    case: bool = False,
    match_all: bool = False,
    exact: bool = True,
) -> pd.DataFrame:
    """
    This function filters a df based on a column and a value.
    It can handle string, numeric, and list-containing columns.

    Parameters
    ----------
        df (pd.DataFrame): The df to filter.
        column (str): The column to filter on.
        value (Any): The value(s) to filter by (can be single item or list).
        regex (bool): Whether to use regex for string columns (not applicable for list columns).
        case (bool): Whether to ignore case for string columns (not applicable for list columns).
        match_all (bool): For list-containing columns and list filter values,
                          if True, requires all filter values to be present in the cell's list.
                          If False, requires at least one filter value to be present.
                          Defaults to False.
        exact (bool): For string columns, if False, use substring (contains) matching instead of exact match.

    Returns
    -------
        pd.DataFrame: The filtered df.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")

    # Drop rows where the target column is NaN to avoid errors in comparisons/iterations
    df_filtered = df.dropna(subset=[column]).copy()
    if df_filtered.empty:
        return df_filtered  # Return empty if all were NaN

    # Determine if the column likely contains lists (check first non-NA value)
    first_val = df_filtered[column].iloc[0]
    col_contains_lists = isinstance(first_val, list)

    if col_contains_lists:
        if regex:
            logging.warning(
                f"Regex filtering is not supported for list-containing column '{column}'. Ignoring regex flag."
            )
        is_filter_list = navis.utils.is_iterable(value) and not isinstance(
            value, str
        )  # Check if filter value is a list/iterable

        if is_filter_list and match_all:
            # Apply "AND" logic: check if *all* items in the filter list `value` are present in the cell's list
            mask = df_filtered[column].apply(
                lambda cell_list: isinstance(cell_list, list)
                and all(item in cell_list for item in value)
            )
        else:
            # Apply "OR" logic (or single item check) using the helper function
            mask = df_filtered[column].apply(
                _check_list_membership, args=(value, is_filter_list)
            )

        return df_filtered[mask]
    else:
        # --- Original logic for non-list columns ---
        dt = df_filtered[column].dtype
        is_str_col = (
            pd.api.types.is_string_dtype(dt) or dt == "object"
        )  # Broader check for string-like

        # Ensure filter value(s) match column type if not regex on string
        if not (regex and is_str_col):
            try:
                if navis.utils.is_iterable(value) and not isinstance(value, str):
                    # Convert each item in the list value
                    value = [match_dtype(v, dt) for v in value]
                else:
                    # Convert single value
                    value = match_dtype(value, dt)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f'Unable to convert filter value for column "{column}" to type {dt}: {e}'
                )

        # Apply filtering
        is_filter_list = navis.utils.is_iterable(value) and not isinstance(value, str)
        if is_filter_list:
            if regex and is_str_col:
                # Combine patterns for regex matching against list of values
                pattern = "|".join(map(str, value))
                df_filtered = df_filtered[
                    df_filtered[column]
                    .astype(str)
                    .str.contains(pattern, na=False, case=case, regex=True)
                ]
            elif is_str_col and not case:
                # Case-insensitive isin for string columns
                lower_value = {str(v).lower() for v in value}
                df_filtered = df_filtered[
                    df_filtered[column].astype(str).str.lower().isin(lower_value)
                ]
            elif is_str_col and not exact:
                # Substring (contains) match for each value in list
                mask = (
                    df_filtered[column]
                    .astype(str)
                    .apply(lambda x: any(str(v).lower() in x.lower() for v in value))
                )
                df_filtered = df_filtered[mask]
            else:
                # Standard isin for exact matches or non-string types
                df_filtered = df_filtered[df_filtered[column].isin(value)]
        else:  # Single filter value
            if regex and is_str_col:
                df_filtered = df_filtered[
                    df_filtered[column]
                    .astype(str)
                    .str.contains(str(value), na=False, case=case, regex=True)
                ]
            elif is_str_col and not case:
                # Case-insensitive comparison for single string value
                if not exact:
                    # Substring (contains) match, case-insensitive
                    mask = (
                        df_filtered[column]
                        .astype(str)
                        .str.lower()
                        .str.contains(str(value).lower(), na=False)
                    )
                    df_filtered = df_filtered[mask]
                else:
                    df_filtered = df_filtered[
                        df_filtered[column].astype(str).str.lower()
                        == str(value).lower()
                    ]
            elif is_str_col and not exact:
                # Substring (contains) match, case-sensitive
                mask = (
                    df_filtered[column].astype(str).str.contains(str(value), na=False)
                )
                df_filtered = df_filtered[mask]
            else:
                # Standard equality check
                df_filtered = df_filtered[df_filtered[column] == value]
        return df_filtered


def make_iterable(x: Any, force_type: Optional[type] = None) -> np.ndarray:
    """
    Convert input to an numpy array.

    Parameters
    ----------
    x : Any
        The input to convert.

    force_type : Optional[type]
        If specified, the input will be cast to this type.

    Returns
    -------
    np.ndarray
        The converted numpy array.
    """
    if not isinstance(x, Iterable):
        x = [x]
    if isinstance(x, (set, dict, pd.Series)):
        x = list(x)

    if force_type is not None:
        try:
            arr = np.array(x, dtype=force_type)
        except Exception as e:
            raise ValueError(f"Cannot convert {x} of type {type(x)} to {force_type}.")
        return arr
    else:
        return np.array(x)


def parse_timestamp(x: Timestamp) -> str:
    """
    Parse a timestamp string to Unix timestamp.

    Parameters
    ----------
    x : Timestamp
        The timestamp string to parse. Int must be unix timestamp. String must be ISO 8601 - e.g. '2021-11-15'. datetime, np.datetime64, pd.Timestamp are also accepted.

    Returns
    -------
    str
        The Unix timestamp.
    """
    if isinstance(x, str):
        try:
            # Convert to datetime and then to Unix timestamp
            dt = pd.to_datetime(x)
            return str(int(dt.timestamp()))
        except ValueError:
            raise ValueError(
                f"Invalid timestamp format: {x}. String must be ISO 8601 format."
            )
    elif isinstance(x, (int, np.int64)):
        return str(int(x))
    else:
        try:
            dt = pd.to_datetime(x)
            return str(int(dt.timestamp()))
        except ValueError:
            raise ValueError(
                f"Invalid timestamp format: {x}. Input must be a valid datetime-like object."
            )


def retry(func, retries=5, cooldown=2):
    """
    Retry function on HTTPError.

    This also suppresses UserWarnings (commonly raised by l2 cache requests)

    Parameters
    ----------
    cooldown :  int | float
                Cooldown period in seconds between attempts.
    retries :   int
                Number of retries before we give up. Every subsequent retry
                will delay by an additional `retry`.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for i in range(1, retries + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    return func(*args, **kwargs)
                except KeyboardInterrupt:
                    raise
                except requests.RequestException:
                    if i >= retries:
                        raise
                except BaseException:
                    raise
                time.sleep(cooldown * i)

    return wrapper


def parse_root_ids(
    neurons: Union[int, str, List[Union[int, str]], "NeuronCriteria"],
) -> List[str]:
    """
    Parse various neuron input types to a list of root ID strings.
    Parameters
    ----------
    neurons : Union[int, str, List[Union[int, str]], NeuronCriteria]
        The neuron(s) to parse. Can be a single root ID (int or str),
        a list of root IDs, or a NeuronCriteria object.

    Returns
    -------
    List[str]
        A list of root ID strings.
    """

    # Normalize input
    if hasattr(neurons, "get_roots"):
        root_ids = neurons.get_roots()
    elif isinstance(neurons, (int, str)):
        root_ids = np.array([neurons])
    elif isinstance(neurons, (list, np.ndarray)):
        # Validate that list/array contains only valid types (int, str)
        if isinstance(neurons, list):
            for item in neurons:
                if not isinstance(item, (int, str)):
                    logging.error(
                        f"Invalid input type for 'neurons': {type(neurons)}. Must be int, str, list, np.ndarray, or NeuronCriteria."
                    )
                    raise ValueError(
                        "Invalid input type for neurons. Must be int, str, list, np.ndarray, or NeuronCriteria."
                    )
        root_ids = np.array(neurons)
    else:
        logging.error(
            f"Invalid input type for 'neurons': {type(neurons)}. Must be int, str, list, np.ndarray, or NeuronCriteria."
        )
        raise ValueError(
            "Invalid input type for neurons. Must be int, str, list, np.ndarray, or NeuronCriteria."
        )

    # Convert to list of str
    root_ids = [str(rid) for rid in root_ids]
    return root_ids


def plot_em_image(x: int, y: int, z: int, size: Optional[int] = 1000) -> np.ndarray:
    """
    Fetch and return an EM image slice from the precomputed CloudVolume.
    Currently only supports slices through the Z axis (i.e. XY plane).

    Parameters
    ----------
    x : int
        The x coordinate of the center of the image slice.
    y : int
        The y coordinate of the center of the image slice.
    z : int
        The z coordinate of the image slice.
    size : int, optional
        The size of the image slice (default is 1000).

    Returns
    -------
    np.ndarray
        The EM image slice as a numpy array.
    """

    # Check size validity
    if size is None:
        size = 1000
    elif size % 2 != 0:
        raise ValueError("Size must be an even integer.")
    elif size < 100 or size > 5000:
        raise ValueError("Size must be between 100 and 5000.")

    # Initialize CloudVolume
    vol = cv.CloudVolume(ALIGNED_EM_URL, mip=0, use_https=True)

    # Check if CloudVolume exists
    if vol.info is None:
        raise ValueError("Could not access CloudVolume at the specified URL.")

    # Calculate bounding box coordinates
    half_size = size // 2
    x_start, x_end = x - half_size, x + half_size
    y_start, y_end = y - half_size, y + half_size
    z_start, z_end = z, z + 1

    # If coordinates are out of bounds, raise error
    if (
        x_start < 0
        or y_start < 0
        or z_start < 0
        or x_end > vol.shape[0]
        or y_end > vol.shape[1]
        or z_end > vol.shape[2]
    ):
        raise ValueError("Coordinates are out of bounds of the CloudVolume.")

    # Fetch the image slice
    img = vol[x_start:x_end, y_start:y_end, z_start:z_end]

    return img.squeeze()


def map_position_to_node(
    neuron: "navis.TreeNeuron",
    position: Union[List[float], np.ndarray],
    return_distance: bool = False,
) -> Union[int, tuple[int, float]]:
    """
    Map a spatial position to the nearest node in a skeleton.

    This utility function finds the closest node in a skeleton to a given position
    using a KDTree for efficient spatial lookup. Useful for soma detection,
    synapse attachment, and other spatial queries.

    Parameters
    ----------
    neuron : navis.TreeNeuron
        The skeleton neuron to search.
    position : list or np.ndarray
        The [x, y, z] coordinates to map. Should be in the same coordinate
        system as the neuron (typically nanometers).
    return_distance : bool, optional
        If True, also return the Euclidean distance to the nearest node.
        Default is False.

    Returns
    -------
    node_id : int
        The node_id of the nearest node.
    distance : float (optional)
        The Euclidean distance to the nearest node in nanometers.
        Only returned if return_distance=True.

    Examples
    --------
    >>> import crantpy as cp
    >>> import numpy as np
    >>> skel = cp.get_l2_skeleton(576460752664524086)
    >>> # Map a position to nearest node
    >>> node_id = cp.map_position_to_node(skel, [240000, 85000, 96000])
    >>> print(f"Nearest node: {node_id}")
    >>> # Get distance too
    >>> node_id, dist = cp.map_position_to_node(skel, [240000, 85000, 96000], return_distance=True)
    >>> print(f"Nearest node: {node_id} at distance {dist:.2f} nm")

    See Also
    --------
    reroot_at_soma : Reroot a skeleton at its soma location.
    detect_soma : Detect soma location in a neuron.
    """
    logger = logging.getLogger(__name__)

    # Validate input
    position = np.asarray(position).flatten()
    if len(position) != 3:
        raise ValueError(f"Position must be [x, y, z], got shape {position.shape}")

    # Build KDTree for efficient spatial queries
    tree = navis.neuron2KDTree(neuron, data="nodes")

    # Query for nearest node
    distance, nearest_idx = tree.query(position.reshape(1, -1))
    distance = float(distance[0])
    nearest_idx = int(nearest_idx[0])

    # Get the node_id from the index
    nearest_node_id = neuron.nodes.iloc[nearest_idx]["node_id"]

    logger.debug(
        f"Mapped position {position} to node {nearest_node_id} "
        f"at distance {distance:.2f} nm"
    )

    if return_distance:
        return int(nearest_node_id), distance
    else:
        return int(nearest_node_id)


def reroot_at_soma(
    neurons: Neurons,
    soma_coords: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    detect_soma_kwargs: Optional[Dict[str, Any]] = None,
    inplace: bool = True,
    progress: bool = False,
) -> Neurons:
    """
    Reroot skeleton(s) at their soma location.

    This convenience function combines soma detection and rerooting. If soma
    coordinates are not provided, they will be automatically detected using
    `detect_soma()`. The skeleton is then rerooted at the node nearest to
    the soma location.

    Parameters
    ----------
    neurons : TreeNeuron | NeuronList
        Single neuron or list of neurons to reroot.
    soma_coords : np.ndarray or list of np.ndarray, optional
        Soma coordinates in pixels [x, y, z]. If not provided, soma will be
        automatically detected using `detect_soma()`. For multiple neurons,
        provide a list of coordinates in the same order as neurons.
    detect_soma_kwargs : dict, optional
        Additional keyword arguments to pass to `detect_soma()` if soma
        coordinates are not provided.
    inplace : bool, optional
        If True, reroot neurons in place. If False, return rerooted copies.
        Default is True.
    progress : bool, optional
        If True, show progress bar when processing multiple neurons or
        detecting somas. Default is False.

    Returns
    -------
    TreeNeuron | NeuronList
        Rerooted neuron(s). Same as input if inplace=True, otherwise copies.

    Examples
    --------
    >>> import crantpy as cp
    >>> # Get skeleton
    >>> skel = cp.get_l2_skeleton(576460752664524086)
    >>> # Reroot at automatically detected soma
    >>> skel_rerooted = cp.reroot_at_soma(skel)
    >>> print(f"Root node: {skel_rerooted.root}")
    >>> # Reroot with provided soma coordinates
    >>> soma = [28000, 9000, 2200]  # in pixels
    >>> skel_rerooted = cp.reroot_at_soma(skel, soma_coords=soma)
    >>> # Process multiple neurons
    >>> skels = cp.get_l2_skeleton([576460752664524086, 576460752590602315])
    >>> skels_rerooted = cp.reroot_at_soma(skels, progress=True)

    See Also
    --------
    map_position_to_node : Map a position to the nearest node.
    detect_soma : Detect soma location in a neuron.
    """
    from crantpy.viz.mesh import detect_soma
    from crantpy.utils.config import SCALE_X, SCALE_Y, SCALE_Z

    logger = logging.getLogger(__name__)

    # Handle single neuron vs list
    if isinstance(neurons, navis.TreeNeuron):
        neurons_list = navis.NeuronList([neurons])
        return_single = True
    else:
        neurons_list = neurons
        return_single = False

    # Make copies if not inplace
    if not inplace:
        neurons_list = neurons_list.copy()

    # Detect soma if not provided
    if soma_coords is None:
        logger.debug("Detecting soma locations...")
        detect_soma_kwargs = detect_soma_kwargs or {}
        neuron_ids = [int(n.id) for n in neurons_list]
        soma_coords = detect_soma(neuron_ids, progress=progress, **detect_soma_kwargs)

    # Convert to list if numpy array
    if isinstance(soma_coords, np.ndarray):
        if soma_coords.ndim == 1:
            # Single soma coordinate
            soma_coords = [soma_coords]
        elif soma_coords.ndim == 2:
            # Multiple soma coordinates - convert each row to a list
            soma_coords = [soma_coords[i] for i in range(len(soma_coords))]
    elif not isinstance(soma_coords, list):
        # Single coordinate (not in list)
        soma_coords = [soma_coords]

    # Validate length
    if len(soma_coords) != len(neurons_list):
        raise ValueError(
            f"Number of soma coordinates ({len(soma_coords)}) does not match "
            f"number of neurons ({len(neurons_list)})"
        )

    # Scale factors for pixel to nm conversion
    scale_factors = np.array([SCALE_X, SCALE_Y, SCALE_Z])

    # Reroot each neuron
    iterator = zip(neurons_list, soma_coords)
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(list(iterator), desc="Rerooting neurons", disable=not progress)

    for neuron, soma_pos in iterator:
        # Convert soma position to nanometers
        soma_nm = np.array(soma_pos) * scale_factors

        # Find nearest node
        node_id, distance = map_position_to_node(neuron, soma_nm, return_distance=True)

        logger.debug(
            f"Rerooting neuron {neuron.id} at node {node_id} "
            f"(distance to soma: {distance:.2f} nm)"
        )

        # Reroot at that node
        neuron.reroot(node_id, inplace=True)

    if return_single:
        return neurons_list[0]
    else:
        return neurons_list
