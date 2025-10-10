# -*- coding: utf-8 -*-
"""
This module provides functions for querying synaptic connectivity in the CRANTb dataset.
Adapted from fafbseg-py (Philipp Schlegel) and the-BANC-fly-connectome (Jasper Phelps).

Function Overview
-----------------
This module contains four main functions for connectivity analysis, each serving
different use cases:

get_synapses()
    Returns individual synaptic connections as a detailed DataFrame.

    Use when you need:
    - Raw synapse-level data with all available columns (coordinates, scores, etc.)
    - Fine-grained analysis of individual synaptic connections
    - Custom aggregation or filtering of synapses
    - Access to synapse metadata (synapse_size, coordinates, quality scores)

    Returns: DataFrame with one row per synapse

get_adjacency()
    Returns a structured adjacency matrix showing connection strengths.

    Use when you need:
    - Matrix-based connectivity analysis
    - Network analysis with standard matrix operations
    - Symmetric connectivity matrices for undirected analysis
    - Integration with graph theory libraries (NetworkX, igraph)
    - Direct input for clustering or community detection algorithms

    Returns: DataFrame adjacency matrix (neurons x neurons)

get_connectivity()
    Returns aggregated connectivity as a simple edge list.

    Use when you need:
    - High-level connectivity overview between neurons
    - Partner analysis (finding strongest connections)
    - Simple edge lists for network visualization
    - Quick connectivity summaries without detailed synapse info
    - Input for graph visualization tools (Cytoscape, Gephi)

    Returns: DataFrame with columns [pre, post, weight]

get_synapse_counts()
    Returns summary statistics of synaptic connections per neuron.

    Use when you need:
    - Quick overview of neuron connectivity profiles
    - Total incoming and outgoing connection counts
    - Comparative analysis of neuron connectivity levels
    - Filtering neurons by connectivity thresholds
    - Summary statistics for large neuron populations

    Returns: DataFrame with columns [pre, post] indexed by neuron ID

"""

import datetime
import logging
from typing import List, Optional, Union, TYPE_CHECKING
import pandas as pd
import numpy as np
import navis
from crantpy.utils.cave.load import get_cave_client
from crantpy.utils.config import CRANT_VALID_DATASETS, SCALE_X, SCALE_Y, SCALE_Z
from crantpy.utils.decorators import inject_dataset, parse_neuroncriteria
from crantpy.utils.helpers import parse_root_ids, retry

if TYPE_CHECKING:
    from crantpy.queries.neurons import NeuronCriteria

logger = logging.getLogger(__name__)


@parse_neuroncriteria()
@inject_dataset(allowed=CRANT_VALID_DATASETS)
def get_synapses(
    pre_ids: Optional[Union[int, str, List[Union[int, str]], "NeuronCriteria"]] = None,
    post_ids: Optional[Union[int, str, List[Union[int, str]], "NeuronCriteria"]] = None,
    threshold: int = 1,
    min_size: Optional[int] = None,
    materialization: Optional[str] = "latest",
    return_pixels: bool = True,
    clean: bool = True,
    update_ids: bool = True,
    dataset: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch synapses for a given set of pre- and/or post-synaptic neuron IDs in CRANTb.

    Parameters
    ----------
    pre_ids : int, str, list of int/str, NeuronCriteria, optional
        Pre-synaptic neuron root ID(s) to include. Can be a single ID, list of IDs,
        or NeuronCriteria object.
    post_ids : int, str, list of int/str, NeuronCriteria, optional
        Post-synaptic neuron root ID(s) to include. Can be a single ID, list of IDs,
        or NeuronCriteria object.
    threshold : int, default 1
        Minimum number of synapses required for a partner to be retained.
        Currently we don't know what a good threshold is.
    min_size : int, optional
        Minimum size for filtering synapses. Currently we don't know what a good size is.
    materialization : str, default 'latest'
        Materialization version to use. 'latest' (default) or 'live' for live table.
    return_pixels : bool, default True
        Whether to convert coordinate columns from nanometers to pixels.
        If True (default), coordinates in ctr_pt_position, pre_pt_position, and
        post_pt_position are converted using dataset scale factors.
        If False, coordinates remain in nanometer units.
    clean : bool, default True
        Whether to perform cleanup of the synapse data:
        - Remove autapses (self-connections)
        - Remove connections involving neuron ID 0 (background)
    update_ids : bool, default True
        Whether to automatically update outdated root IDs to their latest versions
        before querying. This ensures accurate results even after segmentation edits.
        Uses efficient per-ID caching to minimize overhead for repeated queries.
        Set to False only if you're certain all IDs are current (faster but risky).
    dataset : str, optional
        Dataset to use for the query.

    Returns
    -------
    pd.DataFrame
        DataFrame of synaptic connections.

    Raises
    ------
    ValueError
        If neither pre_ids nor post_ids are provided.

    Notes
    -----
    - When update_ids=True (default), outdated root IDs are automatically updated
      using supervoxel IDs from annotations when available for fast, reliable updates
    - ID updates are cached per-ID, so repeated queries with overlapping IDs are efficient
    - Updated IDs are used for the query, but the original IDs are not modified in place

    See Also
    --------
    update_ids : Manually update root IDs to their latest versions
    """
    if pre_ids is None and post_ids is None:
        raise ValueError("You must provide at least one of pre_ids or post_ids")

    # Update IDs if requested
    if update_ids:
        from crantpy.utils.cave.segmentation import update_ids as _update_ids

        if pre_ids is not None:
            parsed_pre_ids = [int(x) for x in parse_root_ids(pre_ids)]
            update_result = _update_ids(parsed_pre_ids, dataset=dataset, progress=False)
            # Check for failed updates
            failed = update_result[update_result["confidence"] == 0]
            if len(failed) > 0:
                logger.warning(
                    f"Failed to update {len(failed)} pre-synaptic root ID(s). "
                    "These IDs may no longer exist or results may be inaccurate."
                )
            parsed_pre_ids = update_result["new_id"].tolist()
        else:
            parsed_pre_ids = None

        if post_ids is not None:
            parsed_post_ids = [int(x) for x in parse_root_ids(post_ids)]
            update_result = _update_ids(
                parsed_post_ids, dataset=dataset, progress=False
            )
            # Check for failed updates
            failed = update_result[update_result["confidence"] == 0]
            if len(failed) > 0:
                logger.warning(
                    f"Failed to update {len(failed)} post-synaptic root ID(s). "
                    "These IDs may no longer exist or results may be inaccurate."
                )
            parsed_post_ids = update_result["new_id"].tolist()
        else:
            parsed_post_ids = None
    else:
        # Don't update IDs
        if pre_ids is not None:
            parsed_pre_ids = [int(x) for x in parse_root_ids(pre_ids)]
        else:
            parsed_pre_ids = None

        if post_ids is not None:
            parsed_post_ids = [int(x) for x in parse_root_ids(post_ids)]
        else:
            parsed_post_ids = None

    # Get CAVE client
    client = get_cave_client(dataset=dataset)

    # Build filter dict
    filter_in_dict = {}
    if parsed_pre_ids is not None:
        filter_in_dict["pre_pt_root_id"] = parsed_pre_ids
    if parsed_post_ids is not None:
        filter_in_dict["post_pt_root_id"] = parsed_post_ids

    if materialization == "live":
        syn = retry(client.materialize.live_query)(
            table="synapses_v2",
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            filter_in_dict=filter_in_dict,
        )
    elif materialization == "latest":
        materialization = retry(client.materialize.most_recent_version)()
        syn = retry(client.materialize.query_table)(
            table="synapses_v2",
            materialization_version=materialization,
            filter_in_dict=filter_in_dict,
        )
    else:
        raise ValueError("materialization must be either 'live' or 'latest'")

    if syn.empty:
        return syn

    if min_size is not None and "size" in syn.columns:
        syn = syn[syn["size"] >= min_size]

    # Thresholding by connection counts between pre-post pairs
    # Count synapses for each pre-post pair
    pair_counts = syn.groupby(["pre_pt_root_id", "post_pt_root_id"]).size()
    valid_pairs = pair_counts[pair_counts >= threshold].index
    # Filter to keep only pairs that meet the threshold
    syn = syn.set_index(["pre_pt_root_id", "post_pt_root_id"])
    syn = syn.loc[syn.index.isin(valid_pairs)]
    syn = syn.reset_index()  # This preserves the columns instead of dropping them

    # Clean up synapses if requested
    if clean:
        # Remove autapses (self-connections)
        syn = syn[syn["pre_pt_root_id"] != syn["post_pt_root_id"]]
        # Remove connections involving background (ID 0)
        syn = syn[(syn["pre_pt_root_id"] != 0) & (syn["post_pt_root_id"] != 0)]

    # Convert coordinates to pixels if requested
    if return_pixels:
        syn = _convert_coordinates_to_pixels(syn)

    return syn


@parse_neuroncriteria()
@inject_dataset(allowed=CRANT_VALID_DATASETS)
def get_adjacency(
    pre_ids: Optional[Union[int, str, List[Union[int, str]], "NeuronCriteria"]] = None,
    post_ids: Optional[Union[int, str, List[Union[int, str]], "NeuronCriteria"]] = None,
    threshold: int = 1,
    min_size: Optional[int] = None,
    materialization: Optional[str] = "latest",
    symmetric: bool = False,
    clean: bool = True,
    update_ids: bool = True,
    dataset: Optional[str] = None,
) -> pd.DataFrame:
    """
    Construct an adjacency matrix from synaptic connections between neurons.

    This function queries the synapses table to get connections between specified
    pre- and post-synaptic neurons, then constructs an adjacency matrix showing
    the number of synapses between each pair.

    Parameters
    ----------
    pre_ids : int, str, list, NeuronCriteria, optional
        Pre-synaptic neuron root IDs or criteria. If None, all pre-synaptic
        neurons in the dataset will be included.
    post_ids : int, str, list, NeuronCriteria, optional
        Post-synaptic neuron root IDs or criteria. If None, all post-synaptic
        neurons in the dataset will be included.
    threshold : int, default 1
        Minimum number of synapses required between a pair to be included
        in the adjacency matrix.
    min_size : int, optional
        Minimum size for filtering synapses before constructing adjacency matrix.
    materialization : str, default 'latest'
        Materialization version to use. 'latest' (default) or 'live' for live table.
    symmetric : bool, default False
        If True, return a symmetric adjacency matrix with the same set of IDs on
        both rows and columns. The neuron set includes all neurons that appear
        in the filtered synapses data (union of all pre- and post-synaptic neurons).
        This provides a complete view of connectivity among all neurons involved
        in the queried connections.
        If False (default), rows represent pre-synaptic neurons and columns
        represent post-synaptic neurons from the actual synapses data.
    clean : bool, default True
        Whether to perform cleanup of the underlying synapse data:
        - Remove autapses (self-connections)
        - Remove connections involving neuron ID 0 (background)
        This parameter is passed to get_synapses().
    update_ids : bool, default True
        Whether to automatically update outdated root IDs to their latest versions
        before querying. This ensures accurate results even after segmentation edits.
        Uses efficient per-ID caching to minimize overhead for repeated queries.
        Set to False only if you're certain all IDs are current (faster but risky).
    dataset : str, optional
        Dataset to use for the query.

    Returns
    -------
    pd.DataFrame
        An adjacency matrix where each entry [i, j] represents the number of
        synapses from neuron i (pre-synaptic) to neuron j (post-synaptic).
        Rows are pre-synaptic neurons, columns are post-synaptic neurons.

    Examples
    --------
    >>> import crantpy as cp
    >>> # Get adjacency between specific neurons
    >>> adj = cp.get_adjacency(pre_ids=[576460752641833774], post_ids=[576460752777916050])
    >>>
    >>> # Get adjacency with minimum threshold
    >>> adj = cp.get_adjacency(pre_ids=[576460752641833774], post_ids=[576460752777916050], threshold=3)
    >>>
    >>> # Get symmetric adjacency matrix
    >>> adj = cp.get_adjacency(pre_ids=[576460752641833774], post_ids=[576460752777916050], symmetric=True)
    >>>
    >>> # Get adjacency matrix with autapses included
    >>> adj = cp.get_adjacency(pre_ids=[576460752641833774], post_ids=[576460752777916050], clean=False)
    >>>
    >>> # Skip ID updates for faster queries (use only if IDs are known to be current)
    >>> adj = cp.get_adjacency(pre_ids=[576460752641833774], post_ids=[576460752777916050], update_ids=False)

    Notes
    -----
    - This function uses get_synapses() internally to retrieve synaptic connections
    - If both pre_ids and post_ids are None, this will query all synapses in the dataset
    - The threshold parameter filters connection pairs, not individual synapses
    - When symmetric=True, the resulting matrix includes all neurons that appear in the
      filtered synapses data, ensuring complete connectivity visualization
    - When symmetric=False, the matrix may be rectangular with different neuron sets
      for rows (pre-synaptic) and columns (post-synaptic)
    - When clean=True (default), autapses and background connections are removed
    - When update_ids=True (default), IDs are automatically updated with efficient caching
    """

    # Get synapses using the same parameters
    synapses = get_synapses(
        pre_ids=pre_ids,
        post_ids=post_ids,
        threshold=threshold,
        min_size=min_size,
        materialization=materialization,
        clean=clean,
        update_ids=update_ids,
        dataset=dataset,
    )

    if synapses.empty:
        # Return empty adjacency matrix with appropriate dimensions
        if symmetric:
            # For symmetric case, determine the neuron set to use
            if pre_ids is not None and post_ids is not None:
                # Use union of provided pre and post IDs
                pre_set = set(int(x) for x in parse_root_ids(pre_ids))
                post_set = set(int(x) for x in parse_root_ids(post_ids))
                common_ids = sorted(list(pre_set.union(post_set)))
            elif pre_ids is not None:
                # Use only pre_ids
                common_ids = sorted([int(x) for x in parse_root_ids(pre_ids)])
            elif post_ids is not None:
                # Use only post_ids
                common_ids = sorted([int(x) for x in parse_root_ids(post_ids)])
            else:
                # Both are None
                common_ids = []
            return pd.DataFrame(0, index=common_ids, columns=common_ids, dtype=int)
        else:
            # Asymmetric case
            pre_list = (
                sorted([int(x) for x in parse_root_ids(pre_ids)])
                if pre_ids is not None
                else []
            )
            post_list = (
                sorted([int(x) for x in parse_root_ids(post_ids)])
                if post_ids is not None
                else []
            )
            return pd.DataFrame(0, index=pre_list, columns=post_list, dtype=int)

    # Extract pre and post IDs from synapses
    pre_neurons = synapses["pre_pt_root_id"].values
    post_neurons = synapses["post_pt_root_id"].values

    if symmetric:
        # For symmetric adjacency matrix, use all neurons that appear in synapses data
        index = sorted(list(set(pre_neurons).union(set(post_neurons))))
        columns = index
    else:
        # Asymmetric case: use actual neurons from the synapses data
        index = sorted(list(set(pre_neurons)))
        columns = sorted(list(set(post_neurons)))

    # Create adjacency matrix
    adj = pd.DataFrame(0, index=index, columns=columns, dtype=int)

    # Count synapses between each pair
    synapse_counts = (
        synapses.groupby(["pre_pt_root_id", "post_pt_root_id"])
        .size()
        .reset_index(name="count")
    )

    for _, row in synapse_counts.iterrows():
        pre_id = row["pre_pt_root_id"]
        post_id = row["post_pt_root_id"]
        count = row["count"]
        if pre_id in adj.index and post_id in adj.columns:
            adj.loc[pre_id, post_id] = count

    return adj


@parse_neuroncriteria()
@inject_dataset(allowed=CRANT_VALID_DATASETS)
def get_connectivity(
    neuron_ids: Union[int, str, List[Union[int, str]], "NeuronCriteria"],
    upstream: bool = True,
    downstream: bool = True,
    threshold: int = 1,
    min_size: Optional[int] = None,
    materialization: Optional[str] = "latest",
    clean: bool = True,
    update_ids: bool = True,
    dataset: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch connectivity information for given neuron(s) in CRANTb.

    This function retrieves synaptic connections for the specified neurons,
    returning a table of connections with pre-synaptic neurons, post-synaptic
    neurons, and synapse counts.

    Parameters
    ----------
    neuron_ids : int, str, list, NeuronCriteria
        Neuron root ID(s) to query connectivity for. Can be a single ID,
        list of IDs, or NeuronCriteria object.
    upstream : bool, default True
        Whether to fetch upstream (incoming) connectivity to the query neurons.
    downstream : bool, default True
        Whether to fetch downstream (outgoing) connectivity from the query neurons.
    threshold : int, default 1
        Minimum number of synapses required between a pair to be included
        in the results.
    min_size : int, optional
        Minimum size for filtering synapses before aggregating connections.
    materialization : str, default 'latest'
        Materialization version to use. 'latest' (default) or 'live' for live table.
    clean : bool, default True
        Whether to perform cleanup of the underlying synapse data:
        - Remove autapses (self-connections)
        - Remove connections involving neuron ID 0 (background)
        This parameter is passed to get_synapses().
    update_ids : bool, default True
        Whether to automatically update outdated root IDs to their latest versions
        before querying. This ensures accurate results even after segmentation edits.
        Uses efficient per-ID caching to minimize overhead for repeated queries.
        Set to False only if you're certain all IDs are current (faster but risky).
    dataset : str, optional
        Dataset to use for the query.

    Returns
    -------
    pd.DataFrame
        Connectivity table with columns:
        - 'pre': pre-synaptic neuron ID
        - 'post': post-synaptic neuron ID
        - 'weight': number of synapses between the pair

    Raises
    ------
    ValueError
        If both upstream and downstream are False.

    Examples
    --------
    >>> import crantpy as cp
    >>> # Get all connections for a neuron
    >>> conn = cp.get_connectivity(576460752641833774)
    >>>
    >>> # Get only downstream connections with threshold
    >>> conn = cp.get_connectivity(576460752641833774, upstream=False, threshold=3)
    >>>
    >>> # Get connectivity for multiple neurons
    >>> conn = cp.get_connectivity([576460752641833774, 576460752777916050])
    >>>
    >>> # Skip ID updates for faster queries (use only if IDs are known to be current)
    >>> conn = cp.get_connectivity(576460752641833774, update_ids=False)

    Notes
    -----
    - This function uses get_synapses() internally to retrieve synaptic connections
    - Results are aggregated by pre-post neuron pairs and sorted by synapse count
    - When clean=True, autapses and background connections are removed
    - When update_ids=True (default), IDs are automatically updated with efficient caching
    """
    if not upstream and not downstream:
        raise ValueError("Both `upstream` and `downstream` cannot be False")

    # Parse neuron IDs - keep as strings to match get_synapses expectations
    query_ids: List[Union[int, str]] = list(parse_root_ids(neuron_ids))

    # Collect all synapses
    synapses_list = []

    if upstream:
        # Get synapses where query neurons are post-synaptic (incoming)
        upstream_syns = get_synapses(
            pre_ids=None,
            post_ids=query_ids,
            threshold=1,  # Apply threshold later after aggregation
            min_size=min_size,
            materialization=materialization,
            clean=clean,
            update_ids=update_ids,
            dataset=dataset,
        )
        if not upstream_syns.empty:
            synapses_list.append(upstream_syns)

    if downstream:
        # Get synapses where query neurons are pre-synaptic (outgoing)
        downstream_syns = get_synapses(
            pre_ids=query_ids,
            post_ids=None,
            threshold=1,  # Apply threshold later after aggregation
            min_size=min_size,
            materialization=materialization,
            clean=clean,
            update_ids=update_ids,
            dataset=dataset,
        )
        if not downstream_syns.empty:
            synapses_list.append(downstream_syns)

    # Combine all synapses
    if not synapses_list:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=["pre", "post", "weight"])

    synapses = pd.concat(synapses_list, ignore_index=True)

    if synapses.empty:
        return pd.DataFrame(columns=["pre", "post", "weight"])

    # Remove duplicates (in case same synapse appears in both upstream/downstream)
    synapses = synapses.drop_duplicates("id")

    # Rename columns for consistency
    synapses = synapses.rename(
        columns={"pre_pt_root_id": "pre", "post_pt_root_id": "post"}
    )

    # Aggregate by pre-post pairs to get connection weights
    connectivity = (
        synapses.groupby(["pre", "post"], as_index=False)
        .size()
        .rename(columns={"size": "weight"})
    )

    # Apply threshold after aggregation
    connectivity = connectivity[connectivity["weight"] >= threshold]

    # Sort by weight (descending) and reset index
    connectivity = connectivity.sort_values("weight", ascending=False).reset_index(
        drop=True
    )

    return connectivity


@parse_neuroncriteria()
@inject_dataset(allowed=CRANT_VALID_DATASETS)
def get_synapse_counts(
    neuron_ids: Union[int, str, List[Union[int, str]], "NeuronCriteria"],
    threshold: int = 1,
    min_size: Optional[int] = None,
    materialization: Optional[str] = "latest",
    clean: bool = True,
    update_ids: bool = True,
    dataset: Optional[str] = None,
) -> pd.DataFrame:
    """
    Get synapse counts (pre and post) for given neuron IDs in CRANTb.

    This function returns the total number of presynaptic and postsynaptic
    connections for each specified neuron, aggregated across all their partners.

    Parameters
    ----------
    neuron_ids : int, str, list, NeuronCriteria
        Neuron root ID(s) to get synapse counts for. Can be a single ID,
        list of IDs, or NeuronCriteria object.
    threshold : int, default 1
        Minimum number of synapses required between a pair to be counted
        towards the total. Pairs with fewer synapses are excluded.
    min_size : int, optional
        Minimum size for filtering individual synapses before counting.
    materialization : str, default 'latest'
        Materialization version to use. 'latest' (default) or 'live' for live table.
    clean : bool, default True
        Whether to perform cleanup of the underlying synapse data:
        - Remove autapses (self-connections)
        - Remove connections involving neuron ID 0 (background)
        This parameter is passed to get_connectivity().
    update_ids : bool, default True
        Whether to automatically update outdated root IDs to their latest versions
        before querying. This ensures accurate results even after segmentation edits.
        Uses efficient per-ID caching to minimize overhead for repeated queries.
        Set to False only if you're certain all IDs are current (faster but risky).
    dataset : str, optional
        Dataset to use for the query.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - index: neuron IDs
        - 'pre': number of presynaptic connections (outgoing)
        - 'post': number of postsynaptic connections (incoming)

    Examples
    --------
    >>> import crantpy as cp
    >>> # Get synapse counts for a single neuron
    >>> counts = cp.get_synapse_counts(576460752641833774)
    >>>
    >>> # Get counts for multiple neurons with threshold
    >>> counts = cp.get_synapse_counts([576460752641833774, 576460752777916050], threshold=3)
    >>>
    >>> # Skip ID updates for faster queries (use only if IDs are known to be current)
    >>> counts = cp.get_synapse_counts(576460752641833774, update_ids=False)

    Notes
    -----
    - This function uses get_connectivity() internally to get connection data
    - Counts represent the number of distinct synaptic partners, not individual synapses
    - The threshold is applied at the connection level (pairs of neurons)
    - When update_ids=True (default), IDs are automatically updated with efficient caching
    """
    # Parse neuron IDs
    query_ids = [int(x) for x in parse_root_ids(neuron_ids)]

    # Get connectivity for all query neurons (both upstream and downstream)
    connectivity = get_connectivity(
        neuron_ids=neuron_ids,
        upstream=True,
        downstream=True,
        threshold=threshold,
        min_size=min_size,
        materialization=materialization,
        clean=clean,
        update_ids=update_ids,
        dataset=dataset,
    )

    # Initialize counts DataFrame with zeros
    counts = pd.DataFrame(
        index=pd.Index(query_ids, name="neuron_id"),
        columns=["pre", "post"],
        data=0,
        dtype=int,
    )

    if connectivity.empty:
        return counts

    # Count presynaptic connections (outgoing from query neurons)
    pre_counts = connectivity[connectivity["pre"].isin(query_ids)].groupby("pre").size()
    pre_counts.name = "pre"

    # Count postsynaptic connections (incoming to query neurons)
    post_counts = (
        connectivity[connectivity["post"].isin(query_ids)].groupby("post").size()
    )
    post_counts.name = "post"

    # Update counts DataFrame
    for neuron_id in query_ids:
        if neuron_id in pre_counts.index:
            counts.loc[neuron_id, "pre"] = pre_counts[neuron_id]
        if neuron_id in post_counts.index:
            counts.loc[neuron_id, "post"] = post_counts[neuron_id]

    return counts


def _convert_coordinates_to_pixels(synapses_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert synapse coordinates from nanometers to pixels.

    This function converts the coordinate columns in a synapses DataFrame
    from nanometer units to pixel units using the dataset-specific scale factors.

    Parameters
    ----------
    synapses_df : pd.DataFrame
        DataFrame containing synapse data with coordinate columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with coordinates converted to pixels.

    Notes
    -----
    The conversion uses scale factors defined in the config:
    - SCALE_X, SCALE_Y = 8 nm/pixel (x and y dimensions)
    - SCALE_Z = 42 nm/pixel (z dimension)

    Coordinate columns that are converted (if present):
    - ctr_pt_position (center point coordinates)
    - pre_pt_position (presynaptic site coordinates)
    - post_pt_position (postsynaptic site coordinates)
    """
    df = synapses_df.copy()

    # Define coordinate columns to convert
    coord_columns = ["ctr_pt_position", "pre_pt_position", "post_pt_position"]

    for col in coord_columns:
        if col in df.columns:
            # Convert coordinates from nm to pixels
            # Each coordinate is a 3-element array [x, y, z]
            df[col] = df[col].apply(
                lambda pos: (
                    [
                        int(pos[0] // SCALE_X),  # x coordinate
                        int(pos[1] // SCALE_Y),  # y coordinate
                        int(pos[2] // SCALE_Z),  # z coordinate
                    ]
                    if pos is not None and len(pos) >= 3
                    else pos
                )
            )

    return df


@inject_dataset(allowed=CRANT_VALID_DATASETS)
def attach_synapses(
    neurons: Union["navis.TreeNeuron", "navis.NeuronList"],
    pre: bool = True,
    post: bool = True,
    threshold: int = 1,
    min_size: Optional[int] = None,
    materialization: str = "latest",
    clean: bool = True,
    max_distance: float = 10000.0,
    update_ids: bool = True,
    dataset: Optional[str] = None,
) -> Union["navis.TreeNeuron", "navis.NeuronList"]:
    """
    Attach synapses as connectors to skeleton neurons.

    This function fetches synapses for the given neuron(s) and maps them to the
    closest node on each skeleton using a KD-tree. The synapses are attached as
    a `.connectors` table with columns for connector_id, x, y, z, type (pre/post),
    partner_id, and node_id.

    Adapted from fafbseg-py (Philipp Schlegel) to work with CRANTb data.

    Parameters
    ----------
    neurons : navis.TreeNeuron or navis.NeuronList
        Skeleton neuron(s) to attach synapses to. Must be TreeNeuron objects
        with node coordinates.
    pre : bool, default True
        Whether to fetch and attach presynapses (outputs) for the given neurons.
    post : bool, default True
        Whether to fetch and attach postsynapses (inputs) for the given neurons.
    threshold : int, default 1
        Minimum number of synapses required between neuron pairs to be included.
    min_size : int, optional
        Minimum synapse size for filtering.
    materialization : str, default 'latest'
        Materialization version to use. Either 'latest' or 'live'.
    clean : bool, default True
        Whether to perform cleanup of synapse data:
        - Remove autapses (self-connections)
        - Remove connections involving neuron ID 0 (background)
        - Remove synapses that are too far from skeleton nodes (see max_distance)
    max_distance : float, default 10000.0
        Maximum distance (in nanometers) between a synapse and its nearest skeleton
        node. Synapses further than this are removed if clean=True. The default of
        10um helps filter out spurious synapse annotations far from the actual neuron.
    update_ids : bool, default True
        Whether to automatically update outdated root IDs to their latest versions
        before querying. This ensures accurate results even after segmentation edits.
        Uses efficient per-ID caching to minimize overhead for repeated queries.
        Set to False only if you're certain all IDs are current (faster but risky).
    dataset : str, optional
        Dataset to use for queries.

    Returns
    -------
    navis.TreeNeuron or navis.NeuronList
        The same neuron(s) with `.connectors` table attached. The connectors table
        includes columns:
        - connector_id: Unique ID for each synapse (sequential)
        - x, y, z: Synapse coordinates in nanometers
        - type: 'pre' for presynapses, 'post' for postsynapses
        - partner_id: Root ID of the partner neuron
        - node_id: ID of the skeleton node closest to this synapse

        Note: The input neurons are modified in place and also returned.

    Raises
    ------
    TypeError
        If neurons is not a TreeNeuron or NeuronList of TreeNeurons.
    ValueError
        If both pre and post are False.

    Examples
    --------
    >>> import crantpy as cp
    >>> # Get a skeleton neuron
    >>> skeleton = cp.get_l2_skeleton(576460752664524086)
    >>>
    >>> # Attach synapses to it
    >>> skeleton = cp.attach_synapses(skeleton)
    >>>
    >>> # View the connectors table
    >>> print(skeleton.connectors.head())
    >>>
    >>> # Get only presynapses
    >>> skeleton = cp.attach_synapses(skeleton, post=False)
    >>>
    >>> # Filter distant synapses more aggressively
    >>> skeleton = cp.attach_synapses(skeleton, max_distance=5000)
    >>>
    >>> # Skip ID updates for faster queries (use only if IDs are known to be current)
    >>> skeleton = cp.attach_synapses(skeleton, update_ids=False)

    See Also
    --------
    get_synapses
        Fetch synapse data without attaching to neurons.

    Notes
    -----
    - This function modifies the input neurons in place by adding/updating the
      .connectors attribute.
    - Synapses are mapped to skeleton nodes using scipy's KDTree for efficient
      nearest neighbor search.
    - The connector_id is a sequential integer starting from 0, not the original
      synapse ID from the database.
    - If a neuron already has a .connectors table, it will be overwritten.
    - Synapse coordinates are automatically converted from pixels to nanometers
      to match skeleton coordinate system (using SCALE_X=8, SCALE_Y=8, SCALE_Z=42).
    - When update_ids=True (default), IDs are automatically updated with efficient caching
    """
    if not pre and not post:
        raise ValueError("`pre` and `post` must not both be False")

    # Handle single neuron vs NeuronList
    if isinstance(neurons, navis.core.BaseNeuron):
        neurons = navis.NeuronList([neurons])
        return_single = True
    else:
        return_single = False

    if not isinstance(neurons, navis.NeuronList):
        raise TypeError(f"Expected TreeNeuron or NeuronList, got {type(neurons)}")

    # Check that all neurons are TreeNeurons
    for n in neurons:
        if not isinstance(n, navis.TreeNeuron):
            raise TypeError(
                f"All neurons must be TreeNeurons, got {type(n)} for neuron {n.id}"
            )

    # Get neuron IDs
    neuron_ids = [int(n.id) for n in neurons]
    logger.debug(f"Fetching synapses for neuron IDs: {neuron_ids}")

    # Fetch synapses - need to make separate queries for pre and post
    # because querying with both creates an AND condition (autapses only)
    syn_list = []

    if pre:
        logger.debug("Fetching presynapses...")
        presyn = get_synapses(
            pre_ids=neuron_ids,
            post_ids=None,
            threshold=threshold,
            min_size=min_size,
            materialization=materialization,
            return_pixels=True,  # Get pixels so we can convert them ourselves
            clean=clean,
            update_ids=update_ids,
            dataset=dataset,
        )
        if not presyn.empty:
            syn_list.append(presyn)
            logger.debug(f"Retrieved {len(presyn)} presynapses")

    if post:
        logger.debug("Fetching postsynapses...")
        postsyn = get_synapses(
            pre_ids=None,
            post_ids=neuron_ids,
            threshold=threshold,
            min_size=min_size,
            materialization=materialization,
            return_pixels=True,  # Get pixels so we can convert them ourselves
            clean=clean,
            update_ids=update_ids,
            dataset=dataset,
        )
        if not postsyn.empty:
            syn_list.append(postsyn)
            logger.debug(f"Retrieved {len(postsyn)} postsynapses")

    # Combine pre and post synapses
    if not syn_list:
        syn = pd.DataFrame()
    else:
        syn = pd.concat(syn_list, axis=0, ignore_index=True)
        # Remove duplicates (in case same synapse appears in both queries)
        if "id" in syn.columns:
            syn = syn.drop_duplicates(subset=["id"])

    logger.debug(f"Total synapses after combining: {len(syn)}")
    if not syn.empty and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Synapse columns: {syn.columns.tolist()}")
        logger.debug(f"Sample synapse: {syn.iloc[0].to_dict()}")

    if syn.empty:
        logger.warning(f"No synapses found for neurons {neuron_ids}")
        # Attach empty connectors tables
        for n in neurons:
            n.connectors = pd.DataFrame(
                columns=["connector_id", "x", "y", "z", "type", "partner_id", "node_id"]
            )
        return neurons[0] if return_single else neurons  # Process each neuron
    for n in neurons:
        presyn = pd.DataFrame()
        postsyn = pd.DataFrame()
        neuron_id = int(n.id)

        logger.debug(f"Processing neuron {neuron_id}")

        # Extract presynapses
        if pre and "pre_pt_root_id" in syn.columns:
            presyn_data = syn[syn["pre_pt_root_id"] == neuron_id].copy()
            logger.debug(f"Found {len(presyn_data)} presynapses for neuron {neuron_id}")
            if not presyn_data.empty:
                # Extract coordinates - handle both array and individual column formats
                if "pre_pt_position" in presyn_data.columns:
                    # Coordinates as array - need to convert from pixels to nm
                    presyn = pd.DataFrame(
                        {
                            "x": presyn_data["pre_pt_position"].apply(
                                lambda p: (
                                    p[0] * SCALE_X  # Convert pixels to nm
                                    if isinstance(p, (list, np.ndarray)) and len(p) >= 3
                                    else np.nan
                                )
                            ),
                            "y": presyn_data["pre_pt_position"].apply(
                                lambda p: (
                                    p[1] * SCALE_Y  # Convert pixels to nm
                                    if isinstance(p, (list, np.ndarray)) and len(p) >= 3
                                    else np.nan
                                )
                            ),
                            "z": presyn_data["pre_pt_position"].apply(
                                lambda p: (
                                    p[2] * SCALE_Z  # Convert pixels to nm
                                    if isinstance(p, (list, np.ndarray)) and len(p) >= 3
                                    else np.nan
                                )
                            ),
                            "partner_id": presyn_data["post_pt_root_id"].values,
                            "type": "pre",
                        }
                    )
                else:
                    # Individual columns (shouldn't happen with current implementation, but be safe)
                    presyn = pd.DataFrame(
                        {
                            "x": presyn_data.get(
                                "pre_pt_position_x", presyn_data.get("pre_x", [])
                            )
                            * SCALE_X,
                            "y": presyn_data.get(
                                "pre_pt_position_y", presyn_data.get("pre_y", [])
                            )
                            * SCALE_Y,
                            "z": presyn_data.get(
                                "pre_pt_position_z", presyn_data.get("pre_z", [])
                            )
                            * SCALE_Z,
                            "partner_id": presyn_data["post_pt_root_id"].values,
                            "type": "pre",
                        }
                    )

        # Extract postsynapses
        if post and "post_pt_root_id" in syn.columns:
            postsyn_data = syn[syn["post_pt_root_id"] == neuron_id].copy()
            logger.debug(
                f"Found {len(postsyn_data)} postsynapses for neuron {neuron_id}"
            )
            if not postsyn_data.empty:
                # Extract coordinates
                if "post_pt_position" in postsyn_data.columns:
                    postsyn = pd.DataFrame(
                        {
                            "x": postsyn_data["post_pt_position"].apply(
                                lambda p: (
                                    p[0] * SCALE_X  # Convert pixels to nm
                                    if isinstance(p, (list, np.ndarray)) and len(p) >= 3
                                    else np.nan
                                )
                            ),
                            "y": postsyn_data["post_pt_position"].apply(
                                lambda p: (
                                    p[1] * SCALE_Y  # Convert pixels to nm
                                    if isinstance(p, (list, np.ndarray)) and len(p) >= 3
                                    else np.nan
                                )
                            ),
                            "z": postsyn_data["post_pt_position"].apply(
                                lambda p: (
                                    p[2] * SCALE_Z  # Convert pixels to nm
                                    if isinstance(p, (list, np.ndarray)) and len(p) >= 3
                                    else np.nan
                                )
                            ),
                            "partner_id": postsyn_data["pre_pt_root_id"].values,
                            "type": "post",
                        }
                    )
                else:
                    postsyn = pd.DataFrame(
                        {
                            "x": postsyn_data.get(
                                "post_pt_position_x", postsyn_data.get("post_x", [])
                            )
                            * SCALE_X,
                            "y": postsyn_data.get(
                                "post_pt_position_y", postsyn_data.get("post_y", [])
                            )
                            * SCALE_Y,
                            "z": postsyn_data.get(
                                "post_pt_position_z", postsyn_data.get("post_z", [])
                            )
                            * SCALE_Z,
                            "partner_id": postsyn_data["pre_pt_root_id"].values,
                            "type": "post",
                        }
                    )

        # Combine pre and post synapses
        connectors = pd.concat([presyn, postsyn], axis=0, ignore_index=True)

        logger.debug(f"Combined {len(connectors)} connectors for neuron {neuron_id}")

        if connectors.empty:
            logger.warning(f"No connectors after combining for neuron {neuron_id}")
            n.connectors = pd.DataFrame(
                columns=["connector_id", "x", "y", "z", "type", "partner_id", "node_id"]
            )
            continue

        # Drop any rows with NaN coordinates
        connectors = connectors.dropna(subset=["x", "y", "z"])
        logger.debug(f"After dropping NaNs: {len(connectors)} connectors")

        if connectors.empty:
            logger.warning(f"All connectors had NaN coordinates for neuron {neuron_id}")
            n.connectors = pd.DataFrame(
                columns=["connector_id", "x", "y", "z", "type", "partner_id", "node_id"]
            )
            continue

        # Map synapses to nearest skeleton nodes using KDTree
        tree = navis.neuron2KDTree(n, data="nodes")
        dist, ix = tree.query(connectors[["x", "y", "z"]].values)

        logger.debug(
            f"Mapped connectors to nodes. Min dist: {dist.min():.1f}, Max dist: {dist.max():.1f} nm"
        )

        # Filter synapses that are too far from skeleton if clean=True
        if clean:
            too_far = dist > max_distance
            if np.any(too_far):
                logger.debug(
                    f"Filtering {too_far.sum()} connectors that are > {max_distance} nm from skeleton"
                )
                connectors = connectors[~too_far].copy()
                ix = ix[~too_far]
                dist = dist[~too_far]

        # Add node IDs
        if len(connectors) > 0:
            connectors["node_id"] = n.nodes.node_id.values[ix]

            # Add sequential connector IDs
            connectors.insert(0, "connector_id", np.arange(len(connectors)))

            # Convert type to categorical to save memory
            connectors["type"] = connectors["type"].astype("category")

            # Attach to neuron
            n.connectors = connectors.reset_index(drop=True)
            logger.info(f"Attached {len(connectors)} connectors to neuron {neuron_id}")
        else:
            logger.warning(
                f"No connectors remaining after filtering for neuron {neuron_id}"
            )
            n.connectors = pd.DataFrame(
                columns=["connector_id", "x", "y", "z", "type", "partner_id", "node_id"]
            )

    return neurons[0] if return_single else neurons
