# -*- coding: utf-8 -*-
"""
This module provides functions for querying synaptic connectivity in the CRANTb dataset.
Adapted from fafbseg-py (Philipp Schlegel) and the-BANC-fly-connectome (Jasper Phelps).
"""

#import warnings
import pandas as pd
import numpy as np

from crantpy.utils.cave import get_cave_client
#from crantpy.utils.decorators import inject_dataset, parse_neuroncriteria

__all__ = [
    "get_synapses",
    #"get_connectivity",
    #"get_transmitter_predictions",
    "get_adjacency",
    #"get_synapse_counts",
]

def get_synapses(
        pre_ids: object = None,
        post_ids: object = None,
        threshold: object = 3,
        drop_duplicates: object = True,
        min_score: object = None,
        materialization: object = 'latest',
        client: object = None,
        batch_size: object = 1000,
        live: object = False,
        verbose: object = False,
) -> object:
    """
    Fetch synapses for a given set of pre- and/or post-synaptic neuron IDs in CRANTb.

    Parameters
    ----------
    pre_ids : list of int, optional
        Pre-synaptic root IDs to include.
    post_ids : list of int, optional
        Post-synaptic root IDs to include.
    threshold : int, optional
        Minimum number of synapses required for a partner to be retained.
        Currently we don't know what a good threshold is.
    drop_duplicates : bool, optional
        Whether to drop duplicate synapses between the same supervoxel pair.
    min_score : int, optional
        Minimum cleft score for filtering synapses. Currently we don't know what a good score is.
    materialization : str or int, optional
        Materialization version to use. 'latest' (default) or specific integer.
    client : caveclient.CAVEclient, optional
        The CAVE client instance.
    batch_size : int, optional
        Number of IDs to query per batch.
    live : bool, optional
        Whether to query the live table instead of a materialized version.
    verbose : bool, optional
        Whether to print progress.

    Returns
    -------
    pd.DataFrame
        DataFrame of synaptic connections.
    """
    if client is None:
        client = get_cave_client()

    if pre_ids is None and post_ids is None:
        raise ValueError("You must provide at least one of pre_ids or post_ids")

    # Handle single int inputs
    if isinstance(pre_ids, (int, np.integer)):
        pre_ids = [pre_ids]
    if isinstance(post_ids, (int, np.integer)):
        post_ids = [post_ids]

    # Build filter dictionary
    filter_in_dict = {}
    if pre_ids is not None:
        filter_in_dict['pre_pt_root_id'] = pre_ids
    if post_ids is not None:
        filter_in_dict['post_pt_root_id'] = post_ids

    if live:
        syn = client.materialize.live_query(
            table='synapses_v2',
            timestamp=datetime.utcnow(),
            filter_in_dict=filter_in_dict
        )
    else:
        if materialization == 'latest':
            materialization = client.materialize.most_recent_version()
        syn = client.materialize.query_table(
            table='synapses_v2',
            materialization_version=materialization,
            filter_in_dict=filter_in_dict
        )

    if syn.empty:
        return syn

    if min_score is not None and 'cleft_score' in syn.columns:
        syn = syn[syn['cleft_score'] >= min_score]

    # Thresholding by partner
    if pre_ids is not None and post_ids is None:
        partner_counts = syn['post_pt_root_id'].value_counts()
        syn = syn[syn['post_pt_root_id'].isin(partner_counts[partner_counts >= threshold].index)]
    elif post_ids is not None and pre_ids is None:
        partner_counts = syn['pre_pt_root_id'].value_counts()
        syn = syn[syn['pre_pt_root_id'].isin(partner_counts[partner_counts >= threshold].index)]

    if drop_duplicates:
        syn = syn.drop_duplicates(subset=['pre_pt_supervoxel_id', 'post_pt_supervoxel_id'])

    return syn.reset_index(drop=True)

def get_adjacency(pre_ids, post_ids, symmetric=False):
    """
    Construct an adjacency matrix from pre- and post-synaptic neuron root IDs.

    Parameters
    ----------
    pre_ids : array-like
        A sequence of pre-synaptic root IDs (e.g., neuron A connects to neuron B).
    post_ids : array-like
        A sequence of post-synaptic root IDs, aligned with `pre_ids`.
    symmetric : bool, optional
        If True, return a symmetric adjacency matrix with the same set of IDs on
        both rows and columns (i.e., pre ∩ post). If False (default), use unique
        pre-synaptic IDs as rows and unique post-synaptic IDs as columns.

    Returns
    -------
    adj : pandas.DataFrame
        A DataFrame where each entry [i, j] represents the number of synapses
        from neuron i (pre-synaptic) to neuron j (post-synaptic).

    Notes
    -----
    - This function assumes that `pre_ids` and `post_ids` are aligned one-to-one,
      meaning each element in `pre_ids[k]` corresponds to `post_ids[k]` in a synaptic connection.
    - If `symmetric=True`, neurons not present in both `pre_ids` and `post_ids` are excluded.
    """

    pre_ids = np.array(pre_ids)
    post_ids = np.array(post_ids)

    if symmetric is True:
        index = list(set(pre_ids).intersection(post_ids))
        columns = index
    else:
        index = list(set(pre_ids))
        columns = list(set(post_ids))

    adj = pd.DataFrame(0, index=index, columns=columns, dtype=int)
    for i in adj.index:
        partners, synapses = np.unique(post_ids[pre_ids == i], return_counts=True)
        for j in range(len(partners)):
            adj.loc[i, partners[j]] = synapses[j]

    return adj
