# -*- coding: utf-8 -*-
"""
This module contains functions to query neuropil information from the CRANTb dataset.

"""

import datetime
import logging
from typing import List, Optional, Union, TYPE_CHECKING
import pandas as pd
import numpy as np
import navis
import trimesh as tm
from crantpy.utils.cave.load import get_cave_client
from crantpy.utils.config import CRANT_VALID_DATASETS, SCALE_X, SCALE_Y, SCALE_Z
from crantpy.utils.decorators import inject_dataset, parse_neuroncriteria
from crantpy.utils.helpers import parse_root_ids, retry
from crantpy.queries.connections import get_synapses
from crantpy.viz.mesh import load_neuropil_mesh
from crantpy.utils.config import NEUROPIL_MESH_DICT

if TYPE_CHECKING:
    from crantpy.queries.neurons import NeuronCriteria

logger = logging.getLogger(__name__)


@parse_neuroncriteria()
@inject_dataset(allowed=CRANT_VALID_DATASETS)
def count_synapses_in_mesh(
    neuron_ids: Union[int, str, List[Union[int, str]], "NeuronCriteria"],
    neuropil_mesh_names: Union[str, List[str]],
    materialization: Optional[str] = "latest",
    update_ids: bool = True,
    dataset: Optional[str] = None,
) -> pd.DataFrame:
    """
    Count the number of presynaptic outputs from specified neurons within neuropil meshes.

    This function queries all synapses where the specified neurons are presynaptic,
    then counts how many of these synapses fall within each specified neuropil mesh region.

    Parameters
    ----------
    neuron_ids : int, str, list of int/str, or NeuronCriteria
        Neuron root ID(s) to query. These neurons will be treated as presynaptic neurons.
        Can be a single ID, list of IDs, or NeuronCriteria object.
    neuropil_mesh_names : str or list of str
        Name(s) of neuropil mesh(es) to load and check against. Must be valid neuropil
        labels from the NEUROPIL_MESH_DICT configuration.
    materialization : str, default 'latest'
        Materialization version to use. 'latest' (default) or 'live' for live table.
        This is passed to get_synapses().
    return_pixels : bool, default True
        Whether to request synapse positions in pixels (True) or nanometers (False).
        Note: Internally, coordinates are converted to nanometers for mesh comparison
        regardless of this setting.
    update_ids : bool, default True
        Whether to automatically update outdated root IDs to their latest versions
        before querying. This is passed to get_synapses().
    dataset : str, optional
        Dataset to use for the query. If None, uses the default dataset.

    Returns
    -------
    pd.DataFrame
        A DataFrame with neurons as rows (indexed by neuron ID) and neuropil meshes
        as columns. Each cell contains the count of synapses from that neuron that
        fall within that neuropil mesh region.

    Raises
    ------
    ValueError
        If neuropil_mesh_names contains invalid neuropil labels.

    Examples
    --------
    >>> import crantpy as cp
    >>> # Count synapses in a single neuropil for one neuron
    >>> counts = cp.count_synapses_in_mesh(
    ...     neuron_ids=576460752641833774,
    ...     neuropil_mesh_names='LH'
    ... )
    >>>
    >>> # Count synapses in multiple neuropils for multiple neurons
    >>> counts = cp.count_synapses_in_mesh(
    ...     neuron_ids=[576460752641833774, 576460752777916050],
    ...     neuropil_mesh_names='antennal_lobe_left', 'mushroom_body_pedunculus_and_lobes_right']
    ... )
    >>>
    >>> # Use with NeuronCriteria
    >>> neurons = cp.NeuronCriteria(cell_type='PN')
    >>> counts = cp.count_synapses_in_mesh(
    ...     neuron_ids=neurons,
    ...     neuropil_mesh_names=['antennal_lobe_left', 'mushroom_body_pedunculus_and_lobes_right']
    ... )

    Notes
    -----
    - This function only considers synapses where the query neurons are **presynaptic**.
    - Synapse positions are extracted from the 'pre_pt_position' column.
    - Coordinates are automatically converted to nanometers for comparison with meshes.
    - Point-in-mesh testing uses ray casting, so meshes should be closed/watertight.
    - The function may take some time for large numbers of synapses or complex meshes.

    See Also
    --------
    get_synapses : Retrieve raw synapse data
    load_neuropil_mesh : Load neuropil mesh by name
    """
    # Parse neuron IDs - keep as List[Union[int, str]] for compatibility with get_synapses
    query_ids_parsed = [int(x) for x in parse_root_ids(neuron_ids)]
    
    # Normalize neuropil mesh names to list
    if isinstance(neuropil_mesh_names, str):
        neuropil_mesh_names = [neuropil_mesh_names]

    # Verify neuropil mesh names
    for neuropil_name in neuropil_mesh_names:
        if neuropil_name not in NEUROPIL_MESH_DICT.values():
            raise ValueError(f"Invalid neuropil mesh name: {neuropil_name}")
    
    # Get synapses where query neurons are presynaptic
    logger.info(f"Fetching synapses for {len(query_ids_parsed)} neuron(s)...")
    synapses = get_synapses(
        pre_ids=query_ids_parsed,
        post_ids=None,  # Don't filter by postsynaptic neurons
        threshold=1,
        min_size=None,
        materialization=materialization,
        return_pixels=False,
        clean=True,
        update_ids=update_ids,
        dataset=dataset,
    )
    
    # Initialize result DataFrame
    result_df = pd.DataFrame(
        0,
        index=pd.Index(query_ids_parsed, name='neuron_id'),
        columns=neuropil_mesh_names,
        dtype=int
    )
    
    # If no synapses found, return empty result
    if synapses.empty:
        logger.warning("No synapses found for the specified neurons")
        return result_df
    
    # Load neuropil meshes
    logger.info(f"Loading {len(neuropil_mesh_names)} neuropil mesh(es)...")
    meshes = {}
    for neuropil_name in neuropil_mesh_names:
        try:
            meshes[neuropil_name] = load_neuropil_mesh(neuropil_name)
            logger.debug(f"Loaded mesh for {neuropil_name}")
        except Exception as e:
            logger.error(f"Failed to load mesh for {neuropil_name}: {e}")
            raise
    
    # Extract synapse positions and convert to nanometers if needed
    # The pre_pt_position column contains [x, y, z] coordinates
    logger.info(f"Processing {len(synapses)} synapses...")
    
    # Convert synapse coordinates to numpy array
    synapse_coords = np.array([
        [pos[0], pos[1], pos[2]]
        for pos in synapses['pre_pt_position'].values
    ])
    
    # Add coordinates and neuron IDs to a working dataframe
    synapse_df = pd.DataFrame({
        'neuron_id': synapses['pre_pt_root_id'].values,
        'x': synapse_coords[:, 0],
        'y': synapse_coords[:, 1],
        'z': synapse_coords[:, 2],
    })
    
    # For each neuropil mesh, check which synapses are inside
    for neuropil_name, mesh in meshes.items():
        logger.info(f"Checking synapses against {neuropil_name} mesh...")
        
        # Check each synapse position against the mesh
        points = synapse_df[['x', 'y', 'z']].values
        inside_mask = mesh.contains(points)
        
        # Count synapses per neuron that are inside this mesh
        synapse_df['inside'] = inside_mask
        counts_per_neuron = synapse_df[synapse_df['inside']].groupby('neuron_id').size()
        
        # Update result DataFrame for this neuropil
        for neuron_id in query_ids_parsed:
            if neuron_id in counts_per_neuron.index:
                count_value = counts_per_neuron[neuron_id]
                result_df.loc[result_df.index == neuron_id, neuropil_name] = int(count_value)
        
        logger.debug(f"Found {inside_mask.sum()} synapses in {neuropil_name}")
    
    logger.info("Synapse counting complete")
    return result_df.reset_index()


