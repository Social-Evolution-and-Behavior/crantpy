# -*- coding: utf-8 -*-
"""
This module contains functions to query neuropil information from the CRANTb dataset.

"""

import datetime
import logging
from typing import List, Optional, Union, TYPE_CHECKING
import pandas as pd
import numpy as np
import trimesh as tm
from crantpy.utils.config import CRANT_VALID_DATASETS
from crantpy.utils.decorators import inject_dataset, parse_neuroncriteria
from crantpy.utils.helpers import parse_root_ids
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
    threshold: int = 1,
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
    threshold : int, default 1
        Minimum number of synapses required between neuron pairs to be included.
        This is passed to get_synapses().
    materialization : str, default 'latest'
        Materialization version to use. 'latest' (default) or 'live' for live table.
        This is passed to get_synapses().
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
        threshold=threshold,
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


@inject_dataset(allowed=CRANT_VALID_DATASETS)
def get_synapses_in_mesh(
    mesh: tm.Trimesh,
    min_synapses_per_neuron: int = 1,
    min_synapses_per_pair: int = 1,
    min_size: Optional[int] = None,
    materialization: Optional[str] = "latest",
    return_pixels: bool = True,
    clean: bool = True,
    mesh_coordinates: str = "nm",
    dataset: Optional[str] = None,
) -> pd.DataFrame:
    """
    Query all synapses within a given mesh volume.

    This function queries the synapse table and filters for synapses whose center point
    coordinates fall within the provided mesh volume. This is useful for analyzing
    synaptic connectivity within a specific anatomical region.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        A trimesh object representing the volume to query. Synapses whose center point
        coordinates fall within this mesh will be returned. The mesh should be in the
        same coordinate space as the synapse data (typically nanometers).
    min_synapses_per_neuron : int, default 1
        Minimum number of synapses required from a presynaptic neuron within the mesh
        to be retained. Presynaptic neurons with fewer synapses are filtered out entirely.
    min_synapses_per_pair : int, default 1
        Minimum number of synapses required between a neuron pair (pre-post) to be retained.
        Synaptic connections (pre-post pairs) with fewer synapses are filtered out.
    min_size : int, optional
        Minimum size for filtering synapses. If specified, only synapses with size
        greater than or equal to this value will be included.
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
    mesh_coordinates : str, default "nm"
        Coordinate system of the mesh. Either "nm" (nanometers) or "voxels" (pixels).
        If "nm", both the query and mesh.contains() will use nanometer coordinates.
        If "voxels", the mesh bounds will be converted to nanometers for the database
        query, and synapse coordinates will be converted to voxels for mesh.contains().
    dataset : str, optional
        Dataset to use for the query. If None, uses the default dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame of synapses within the mesh, with the same columns as returned by
        get_synapses(), including an additional boolean column 'in_mesh' that is True
        for all returned synapses.

    Examples
    --------
    >>> import crantpy as cp
    >>> import trimesh as tm
    >>> # Load a neuropil mesh
    >>> mesh = cp.load_neuropil_mesh('antennal_lobe_left')
    >>>
    >>> # Get all synapses within the mesh
    >>> synapses = cp.get_synapses_in_mesh(mesh)
    >>>
    >>> # Get synapses with minimum pair threshold (only pairs with 3+ synapses)
    >>> synapses = cp.get_synapses_in_mesh(mesh, min_synapses_per_pair=3)
    >>>
    >>> # Get synapses from neurons with at least 10 total synapses in the mesh
    >>> synapses = cp.get_synapses_in_mesh(mesh, min_synapses_per_neuron=10)
    >>>
    >>> # Get synapses with minimum size threshold
    >>> synapses = cp.get_synapses_in_mesh(mesh, min_size=50)
    >>>
    >>> # Get live data without cleaning
    >>> synapses = cp.get_synapses_in_mesh(mesh, materialization='live', clean=False)
    >>>
    >>> # Use a mesh in voxel coordinates instead of nanometers
    >>> synapses = cp.get_synapses_in_mesh(mesh_voxels, mesh_coordinates='voxels')

    Notes
    -----
    - This function first filters synapses using the mesh's bounding box, then performs
      precise point-in-mesh testing only on synapses within the bounding box. This is
      much more efficient than querying all synapses in the dataset.
    - Synapse positions are extracted from the 'ctr_pt_position' column (center point),
      which stores coordinates in nanometer space by default in the CAVE database.
    - The mesh_coordinates parameter controls coordinate conversion: meshes from
      load_neuropil_mesh() are typically in nanometers, while custom meshes may be
      in voxel space.
    - The min_synapses_per_neuron parameter filters out entire presynaptic neurons if
      they have fewer than the specified number of synapses within the mesh.
    - The min_synapses_per_pair parameter filters connection pairs (pre-post neuron pairs),
      keeping only pairs with the minimum number of synapses specified.
    - Both filters are applied independently; a synapse is retained only if its presynaptic
      neuron meets the min_synapses_per_neuron threshold AND its pre-post pair meets
      the min_synapses_per_pair threshold.
    - The mesh.contains() method uses ray casting, so meshes should be closed/watertight
      for accurate results.
    - The return_pixels parameter only affects the output coordinates, not the query.

    See Also
    --------
    get_synapses : Query synapses by pre/post neuron IDs
    count_synapses_in_mesh : Count synapses from specific neurons within meshes
    load_neuropil_mesh : Load neuropil mesh by name
    """
    from crantpy.utils.cave.load import get_cave_client
    from crantpy.utils.config import SCALE_X, SCALE_Y, SCALE_Z
    from crantpy.utils.helpers import retry

    # Get CAVE client
    client = get_cave_client(dataset=dataset)

    # Validate mesh_coordinates parameter
    if mesh_coordinates not in ["nm", "voxels"]:
        raise ValueError("mesh_coordinates must be either 'nm' or 'voxels'")

    # Get mesh bounding box for efficient spatial filtering
    min_coords, max_coords = mesh.bounds
    logger.info(f"Mesh bounding box ({mesh_coordinates}): min={min_coords}, max={max_coords}")
    
    # Convert mesh bounds to nanometers if needed (CAVE database uses nanometer coordinates)
    if mesh_coordinates == "voxels":
        # Convert from voxels to nanometers for the database query
        min_coords_nm = np.array([
            min_coords[0] * SCALE_X,
            min_coords[1] * SCALE_Y,
            min_coords[2] * SCALE_Z
        ])
        max_coords_nm = np.array([
            max_coords[0] * SCALE_X,
            max_coords[1] * SCALE_Y,
            max_coords[2] * SCALE_Z
        ])
        logger.info(f"Converted to nanometers for query: min={min_coords_nm}, max={max_coords_nm}")
    else:
        # Mesh is already in nanometers
        min_coords_nm = min_coords
        max_coords_nm = max_coords
    
    # Build spatial filter using bounding box in nanometer coordinates
    bbox = [min_coords_nm.tolist(), max_coords_nm.tolist()]
    filter_spatial_dict = {
        "ctr_pt_position": bbox
    }

    # Query synapses from the database with bounding box filter
    logger.info("Querying synapses within mesh bounding box...")
    try:
        if materialization == "live":
            syn = retry(client.materialize.live_query)(
                table="synapses_v2",
                timestamp=datetime.datetime.now(datetime.timezone.utc),
                filter_spatial_dict=filter_spatial_dict,
            )
        elif materialization == "latest":
            materialization_version = retry(client.materialize.most_recent_version)()
            syn = retry(client.materialize.query_table)(
                table="synapses_v2",
                materialization_version=materialization_version,
                filter_spatial_dict=filter_spatial_dict,
            )
        else:
            raise ValueError("materialization must be either 'live' or 'latest'")
    except Exception as e:
        # If spatial filtering not supported, fall back to querying all synapses
        logger.warning(f"Spatial filtering failed ({e}), falling back to full query")
        if materialization == "live":
            syn = retry(client.materialize.live_query)(
                table="synapses_v2",
                timestamp=datetime.datetime.now(datetime.timezone.utc),
            )
        elif materialization == "latest":
            materialization_version = retry(client.materialize.most_recent_version)()
            syn = retry(client.materialize.query_table)(
                table="synapses_v2",
                materialization_version=materialization_version,
            )
        else:
            raise ValueError("materialization must be either 'live' or 'latest'")

    if syn.empty:
        logger.warning("No synapses found in bounding box")
        return syn

    logger.info(f"Retrieved {len(syn)} synapses within bounding box")

    for col in ['ctr_pt_position', 'pre_pt_position', 'post_pt_position']:
        if col not in syn.columns:
            raise ValueError(f"Expected column '{col}' not found in synapse data")

        # Extract x, y, z coordinates from the specified column
        logger.info(f"Extracting coordinates from {col}...")
        
        # The specified column contains [x, y, z] coordinates in nanometers
        synapse_coords = np.array([
            [pos[0], pos[1], pos[2]]
            for pos in syn[col].values
        ])
        
        # Convert coordinates if needed for mesh.contains() check
        if mesh_coordinates == "voxels":
            # Convert synapse coordinates from nanometers to voxels to match mesh
            synapse_coords_for_mesh = synapse_coords.copy()
            synapse_coords_for_mesh[:, 0] = synapse_coords[:, 0] / SCALE_X
            synapse_coords_for_mesh[:, 1] = synapse_coords[:, 1] / SCALE_Y
            synapse_coords_for_mesh[:, 2] = synapse_coords[:, 2] / SCALE_Z
            logger.debug("Converted synapse coordinates from nanometers to voxels for mesh check")
        else:
            # Mesh is in nanometers, use coordinates as-is
            synapse_coords_for_mesh = synapse_coords
        
        # Check which synapses are inside the mesh (MAIN FILTERING STEP)
        logger.info(f"Checking {len(synapse_coords_for_mesh)} synapses against mesh using {col} column...")
        inside_mask = mesh.contains(synapse_coords_for_mesh)
        
        logger.info(f"Found {inside_mask.sum()} synapses using {col} column inside mesh")
        
        # Filter to only synapses inside the mesh
        syn = syn[inside_mask].copy()
        
    if syn.empty:
        logger.warning("No synapses found within mesh")
        return syn

    # Apply size filter if specified
    if min_size is not None and "size" in syn.columns:
        syn = syn[syn["size"] >= min_size]
        logger.info(f"After size filtering: {len(syn)} synapses")

    # Clean up synapses if requested
    if clean:
        # Remove autapses (self-connections)
        syn = syn[syn["pre_pt_root_id"] != syn["post_pt_root_id"]]
        # Remove connections involving background (ID 0)
        syn = syn[(syn["pre_pt_root_id"] != 0) & (syn["post_pt_root_id"] != 0)]
        logger.info(f"After cleaning: {len(syn)} synapses")

    if syn.empty:
        logger.warning("No synapses remaining after filtering")
        return syn

    # Apply neuron-level filtering if specified
    if min_synapses_per_neuron > 1:
        # Count synapses for each neuron (both as pre and post)
        pre_counts = syn["pre_pt_root_id"].value_counts()
        post_counts = syn["post_pt_root_id"].value_counts()
        # Combine counts: for each neuron, total synapses it participates in
        all_neuron_counts = pre_counts.add(post_counts, fill_value=0)
        valid_neurons = all_neuron_counts[all_neuron_counts >= min_synapses_per_neuron].index
        # Filter to keep only synapses where both pre and post neurons meet the threshold
        syn = syn[(syn["pre_pt_root_id"].isin(valid_neurons)) & (syn["post_pt_root_id"].isin(valid_neurons))]
        logger.info(f"After neuron filtering: {len(syn)} synapses from {len(valid_neurons)} neurons with >= {min_synapses_per_neuron} synapses")

    # Apply pair-level filtering if specified
    if min_synapses_per_pair > 1:
        # Count synapses for each pre-post pair
        pair_counts = syn.groupby(["pre_pt_root_id", "post_pt_root_id"]).size()
        valid_pairs = pair_counts[pair_counts >= min_synapses_per_pair].index
        # Filter to keep only pairs that meet the threshold
        syn = syn.set_index(["pre_pt_root_id", "post_pt_root_id"])
        syn = syn.loc[syn.index.isin(valid_pairs)]
        syn = syn.reset_index()  # This preserves the columns instead of dropping them
        logger.info(f"After pair filtering: {len(syn)} synapses with >= {min_synapses_per_pair} synapses per pair")

    if syn.empty:
        logger.warning("No synapses remaining after filtering")
        return syn

    # Convert coordinates to pixels if requested
    if return_pixels and not syn.empty:
        from crantpy.queries.connections import _convert_coordinates_to_pixels
        syn = _convert_coordinates_to_pixels(syn)
    
    logger.info(f"Returning {len(syn)} synapses within mesh")
    return syn



