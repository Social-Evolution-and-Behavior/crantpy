# -*- coding: utf-8 -*-
"""
This module provides functions for querying synaptic connectivity in the CRANTb dataset.
Adapted from fafbseg-py (Philipp Schlegel) and the-BANC-fly-connectome (Jasper Phelps).

Function Overview
-----------------
This module contains three main functions for connectivity analysis, each serving
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

"""

import datetime
from typing import List, Optional, Union, TYPE_CHECKING
import pandas as pd
import numpy as np
from crantpy.utils.cave.load import get_cave_client
from crantpy.utils.config import CRANT_VALID_DATASETS
from crantpy.utils.decorators import inject_dataset, parse_neuroncriteria
from crantpy.utils.helpers import parse_root_ids, retry

if TYPE_CHECKING:
    from crantpy.queries.neurons import NeuronCriteria


@parse_neuroncriteria()
@inject_dataset(allowed=CRANT_VALID_DATASETS)
def get_synapses(
    pre_ids: Optional[Union[int, str, List[Union[int, str]], 'NeuronCriteria']] = None,
    post_ids: Optional[Union[int, str, List[Union[int, str]], 'NeuronCriteria']] = None,
    threshold: int = 1,
    min_size: Optional[int] = None,
    materialization: Optional[str] = 'latest',
    dataset: Optional[str] = None
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
    threshold : int, default 3
        Minimum number of synapses required for a partner to be retained.
        Currently we don't know what a good threshold is.
    min_size : int, optional
        Minimum size for filtering synapses. Currently we don't know what a good size is.
    materialization : str, default 'latest'
        Materialization version to use. 'latest' (default) or 'live' for live table.
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
    """
    if pre_ids is None and post_ids is None:
        raise ValueError("You must provide at least one of pre_ids or post_ids")

    # Get CAVE client
    client = get_cave_client(dataset=dataset)

    # Parse neuron IDs using the helper function
    filter_in_dict = {}
    if pre_ids is not None:
        parsed_pre_ids = [int(x) for x in parse_root_ids(pre_ids)]
        filter_in_dict['pre_pt_root_id'] = parsed_pre_ids
    if post_ids is not None:
        parsed_post_ids = [int(x) for x in parse_root_ids(post_ids)]
        filter_in_dict['post_pt_root_id'] = parsed_post_ids

    if materialization == 'live':
        syn = retry(client.materialize.live_query)(
            table='synapses_v2',
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            filter_in_dict=filter_in_dict
        )
    elif materialization == 'latest':
        materialization = retry(client.materialize.most_recent_version)()
        syn = retry(client.materialize.query_table)(
            table='synapses_v2',
            materialization_version=materialization,
            filter_in_dict=filter_in_dict
        )
    else:
        raise ValueError("materialization must be either 'live' or 'latest'") 

    if syn.empty:
        return syn

    if min_size is not None and 'size' in syn.columns:
        syn = syn[syn['size'] >= min_size]

    # Thresholding by connection counts between pre-post pairs
    # Count synapses for each pre-post pair
    pair_counts = syn.groupby(['pre_pt_root_id', 'post_pt_root_id']).size()
    valid_pairs = pair_counts[pair_counts >= threshold].index
    # Filter to keep only pairs that meet the threshold
    syn = syn.set_index(['pre_pt_root_id', 'post_pt_root_id'])
    syn = syn.loc[syn.index.isin(valid_pairs)]
    syn = syn.reset_index()  # This preserves the columns instead of dropping them

    return syn


@parse_neuroncriteria()
@inject_dataset(allowed=CRANT_VALID_DATASETS)
def get_adjacency(
    pre_ids: Optional[Union[int, str, List[Union[int, str]], 'NeuronCriteria']] = None,
    post_ids: Optional[Union[int, str, List[Union[int, str]], 'NeuronCriteria']] = None,
    threshold: int = 1,
    min_size: Optional[int] = None,
    materialization: Optional[str] = 'latest',
    symmetric: bool = False,
    dataset: Optional[str] = None
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
        both rows and columns. The neuron set is determined as follows:
        - If both pre_ids and post_ids are provided: use their union
        - If only one is provided: use that set
        - If neither is provided: use all neurons that appear in the synapses
        Only neurons that actually appear in the synapses data will be included.
        If False (default), rows represent pre-synaptic neurons and columns 
        represent post-synaptic neurons from the actual synapses data.
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

    Notes
    -----
    - This function uses get_synapses() internally to retrieve synaptic connections
    - If both pre_ids and post_ids are None, this will query all synapses in the dataset
    - The threshold parameter filters connection pairs, not individual synapses
    - When symmetric=True, the resulting matrix has the same neurons on rows and columns,
      ensuring that adj[i,j] and adj[j,i] positions both exist (though values may differ)
    - When symmetric=False, the matrix may be rectangular with different neuron sets
      for rows (pre-synaptic) and columns (post-synaptic)
    """
    
    # Get synapses using the same parameters
    synapses = get_synapses(
        pre_ids=pre_ids,
        post_ids=post_ids,
        threshold=threshold,
        min_size=min_size,
        materialization=materialization,
        dataset=dataset
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
            pre_list = sorted([int(x) for x in parse_root_ids(pre_ids)]) if pre_ids is not None else []
            post_list = sorted([int(x) for x in parse_root_ids(post_ids)]) if post_ids is not None else []
            return pd.DataFrame(0, index=pre_list, columns=post_list, dtype=int)
    
    # Extract pre and post IDs from synapses
    pre_neurons = synapses['pre_pt_root_id'].values
    post_neurons = synapses['post_pt_root_id'].values
    
    if symmetric:
        # For symmetric adjacency matrix, use the same set of neurons for both rows and columns
        if pre_ids is not None and post_ids is not None:
            # Use union of provided IDs, filtered to neurons that appear in synapses
            pre_set = set(int(x) for x in parse_root_ids(pre_ids))
            post_set = set(int(x) for x in parse_root_ids(post_ids))
            provided_union = pre_set.union(post_set)
            # Filter to neurons that actually appear in the synapses data
            data_neurons = set(pre_neurons).union(set(post_neurons))
            index = sorted(list(provided_union.intersection(data_neurons)))
        elif pre_ids is not None:
            # Use pre_ids that appear in the synapses data
            provided_pre = set(int(x) for x in parse_root_ids(pre_ids))
            data_neurons = set(pre_neurons).union(set(post_neurons))
            index = sorted(list(provided_pre.intersection(data_neurons)))
        elif post_ids is not None:
            # Use post_ids that appear in the synapses data
            provided_post = set(int(x) for x in parse_root_ids(post_ids))
            data_neurons = set(pre_neurons).union(set(post_neurons))
            index = sorted(list(provided_post.intersection(data_neurons)))
        else:
            # Neither provided, use all neurons that appear in synapses
            index = sorted(list(set(pre_neurons).union(set(post_neurons))))
        
        columns = index
    else:
        # Asymmetric case: use actual neurons from the synapses data
        index = sorted(list(set(pre_neurons)))
        columns = sorted(list(set(post_neurons)))
    
    # Create adjacency matrix
    adj = pd.DataFrame(0, index=index, columns=columns, dtype=int)
    
    # Count synapses between each pair
    synapse_counts = synapses.groupby(['pre_pt_root_id', 'post_pt_root_id']).size().reset_index(name='count')
    
    for _, row in synapse_counts.iterrows():
        pre_id = row['pre_pt_root_id']
        post_id = row['post_pt_root_id']
        count = row['count']
        if pre_id in adj.index and post_id in adj.columns:
            adj.loc[pre_id, post_id] = count

    return adj



@parse_neuroncriteria()
@inject_dataset(allowed=CRANT_VALID_DATASETS)
def get_connectivity(
    neuron_ids: Union[int, str, List[Union[int, str]], 'NeuronCriteria'],
    upstream: bool = True,
    downstream: bool = True,
    threshold: int = 1,
    min_size: Optional[int] = None,
    materialization: Optional[str] = 'latest',
    clean: bool = True,
    dataset: Optional[str] = None
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
        Whether to perform cleanup of the connectivity data:
        - Remove autapses (self-connections)
        - Remove connections involving neuron ID 0 (background)
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

    Notes
    -----
    - This function uses get_synapses() internally to retrieve synaptic connections
    - Results are aggregated by pre-post neuron pairs and sorted by synapse count
    - When clean=True, autapses and background connections are removed
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
            dataset=dataset
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
            dataset=dataset
        )
        if not downstream_syns.empty:
            synapses_list.append(downstream_syns)
    
    # Combine all synapses
    if not synapses_list:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['pre', 'post', 'weight'])
    
    synapses = pd.concat(synapses_list, ignore_index=True)
    
    if synapses.empty:
        return pd.DataFrame(columns=['pre', 'post', 'weight'])

    # Remove duplicates (in case same synapse appears in both upstream/downstream)
    synapses = synapses.drop_duplicates('id')
    
    # Rename columns for consistency
    synapses = synapses.rename(columns={
        'pre_pt_root_id': 'pre',
        'post_pt_root_id': 'post'
    })
    
    # Clean up connections if requested
    if clean:
        # Remove autapses (self-connections)
        synapses = synapses[synapses['pre'] != synapses['post']]
        # Remove connections involving background (ID 0)
        synapses = synapses[(synapses['pre'] != 0) & (synapses['post'] != 0)]
    
    # Aggregate by pre-post pairs to get connection weights
    connectivity = (
        synapses.groupby(['pre', 'post'], as_index=False)
        .size()
        .rename(columns={'size': 'weight'})
    )
    
    # Apply threshold after aggregation
    connectivity = connectivity[connectivity['weight'] >= threshold]
    
    # Sort by weight (descending) and reset index
    connectivity = connectivity.sort_values('weight', ascending=False).reset_index(drop=True)
    
    return connectivity
