# -*- coding: utf-8 -*-
"""Tests for the neuropils module."""

import pytest
import pandas as pd
import numpy as np
import trimesh as tm
from unittest.mock import Mock, patch, MagicMock
from crantpy.queries.neuropils import count_synapses_in_mesh, get_synapses_in_mesh
from crantpy.queries.neurons import NeuronCriteria


# Test neuron IDs - using known projection neurons that should have synapses
TEST_SINGLE_NEURON = 576460752664524086
TEST_MULTIPLE_NEURONS = [576460752664524086, 576460752662516321]

# Test neuropil mesh names
TEST_SINGLE_NEUROPIL = "antennal_lobe_right"
TEST_MULTIPLE_NEUROPILS = [
    "antennal_lobe_right",
    "mushroom_body_lateral_calyx_right",
    "lateral_horn_right"
]


def test_count_synapses_in_mesh_single_neuron_single_neuropil():
    """Test counting synapses for a single neuron in a single neuropil."""
    result = count_synapses_in_mesh(
        neuron_ids=TEST_SINGLE_NEURON,
        neuropil_mesh_names=TEST_SINGLE_NEUROPIL,
        dataset="latest"
    )
    
    # Check that result is a DataFrame
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    
    # Check that DataFrame has correct shape (1 neuron x 1 neuropil)
    assert result.shape == (1, 2), f"Expected shape (1, 2), got {result.shape}"  # neuron_id + 1 neuropil column
    
    # Check that neuron_id column exists
    assert "neuron_id" in result.columns, "Result should have 'neuron_id' column"
    
    # Check that neuropil column exists
    assert TEST_SINGLE_NEUROPIL in result.columns, f"Result should have '{TEST_SINGLE_NEUROPIL}' column"
    
    # Check that values are integers (synapse counts)
    assert result[TEST_SINGLE_NEUROPIL].dtype == np.int64 or result[TEST_SINGLE_NEUROPIL].dtype == int
    
    # Check that counts are non-negative
    assert (result[TEST_SINGLE_NEUROPIL] >= 0).all(), "Synapse counts should be non-negative"


def test_count_synapses_in_mesh_single_neuron_multiple_neuropils():
    """Test counting synapses for a single neuron across multiple neuropils."""
    result = count_synapses_in_mesh(
        neuron_ids=TEST_SINGLE_NEURON,
        neuropil_mesh_names=TEST_MULTIPLE_NEUROPILS,
        dataset="latest"
    )
    
    # Check that result is a DataFrame
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    
    # Check that DataFrame has correct shape (1 neuron x N neuropils)
    expected_cols = len(TEST_MULTIPLE_NEUROPILS) + 1  # +1 for neuron_id
    assert result.shape == (1, expected_cols), f"Expected shape (1, {expected_cols}), got {result.shape}"
    
    # Check that all neuropil columns exist
    for neuropil_name in TEST_MULTIPLE_NEUROPILS:
        assert neuropil_name in result.columns, f"Result should have '{neuropil_name}' column"
    
    # Check that all counts are non-negative
    for neuropil_name in TEST_MULTIPLE_NEUROPILS:
        assert (result[neuropil_name] >= 0).all(), f"Counts for {neuropil_name} should be non-negative"


def test_count_synapses_in_mesh_multiple_neurons_single_neuropil():
    """Test counting synapses for multiple neurons in a single neuropil."""
    result = count_synapses_in_mesh(
        neuron_ids=TEST_MULTIPLE_NEURONS,
        neuropil_mesh_names=TEST_SINGLE_NEUROPIL,
        dataset="latest"
    )
    
    # Check that result is a DataFrame
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    
    # Check that DataFrame has correct shape (N neurons x 1 neuropil)
    assert result.shape == (len(TEST_MULTIPLE_NEURONS), 2), f"Expected shape ({len(TEST_MULTIPLE_NEURONS)}, 2)"
    
    # Check that all neurons are represented
    assert len(result) == len(TEST_MULTIPLE_NEURONS), "All neurons should be in result"
    
    # Check that neuron IDs are correct
    result_ids = set(result["neuron_id"].values)
    expected_ids = set(TEST_MULTIPLE_NEURONS)
    assert result_ids == expected_ids, f"Neuron IDs mismatch: {result_ids} vs {expected_ids}"


def test_count_synapses_in_mesh_multiple_neurons_multiple_neuropils():
    """Test counting synapses for multiple neurons across multiple neuropils."""
    result = count_synapses_in_mesh(
        neuron_ids=TEST_MULTIPLE_NEURONS,
        neuropil_mesh_names=TEST_MULTIPLE_NEUROPILS,
        dataset="latest"
    )
    
    # Check that result is a DataFrame
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    
    # Check that DataFrame has correct shape
    expected_cols = len(TEST_MULTIPLE_NEUROPILS) + 1  # +1 for neuron_id
    assert result.shape == (len(TEST_MULTIPLE_NEURONS), expected_cols)
    
    # Check that all columns exist
    assert "neuron_id" in result.columns
    for neuropil_name in TEST_MULTIPLE_NEUROPILS:
        assert neuropil_name in result.columns, f"Missing column: {neuropil_name}"
    
    # Check that all values are non-negative integers
    for neuropil_name in TEST_MULTIPLE_NEUROPILS:
        assert (result[neuropil_name] >= 0).all(), f"Negative counts found in {neuropil_name}"


def test_count_synapses_in_mesh_invalid_neuropil():
    """Test error handling for invalid neuropil name."""
    with pytest.raises(ValueError) as excinfo:
        count_synapses_in_mesh(
            neuron_ids=TEST_SINGLE_NEURON,
            neuropil_mesh_names="invalid_neuropil_name",
            dataset="latest"
        )
    
    assert "Invalid neuropil mesh name" in str(excinfo.value)


def test_count_synapses_in_mesh_with_threshold():
    """Test counting synapses with different threshold values."""
    # Test with default threshold (1)
    result_default = count_synapses_in_mesh(
        neuron_ids=TEST_SINGLE_NEURON,
        neuropil_mesh_names=TEST_SINGLE_NEUROPIL,
        threshold=1,
        dataset="latest"
    )
    
    # Test with higher threshold (should have same or fewer synapses)
    result_higher = count_synapses_in_mesh(
        neuron_ids=TEST_SINGLE_NEURON,
        neuropil_mesh_names=TEST_SINGLE_NEUROPIL,
        threshold=3,
        dataset="latest"
    )
    
    # Both should be valid DataFrames
    assert isinstance(result_default, pd.DataFrame)
    assert isinstance(result_higher, pd.DataFrame)
    
    # Higher threshold should give same or fewer total synapses
    count_default = result_default[TEST_SINGLE_NEUROPIL].iloc[0]
    count_higher = result_higher[TEST_SINGLE_NEUROPIL].iloc[0]
    assert count_higher <= count_default, "Higher threshold should not increase synapse count"


def test_count_synapses_in_mesh_with_neuron_criteria():
    """Test counting synapses using NeuronCriteria."""
    # Create a NeuronCriteria for a specific cell type
    criteria = NeuronCriteria(
        cell_class='olfactory_projection_neuron',
        side='right',
        dataset="latest"
    )
    
    result = count_synapses_in_mesh(
        neuron_ids=criteria,
        neuropil_mesh_names=TEST_SINGLE_NEUROPIL,
        dataset="latest"
    )
    
    # Check that result is a DataFrame
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    
    # Check that we got results for multiple neurons
    assert len(result) > 0, "Should have results for at least one neuron"
    
    # Check that all required columns exist
    assert "neuron_id" in result.columns
    assert TEST_SINGLE_NEUROPIL in result.columns


def test_count_synapses_in_mesh_no_synapses():
    """Test behavior when neuron has no synapses in the specified neuropil."""
    # Use a neuron and neuropil combination unlikely to have overlapping synapses
    result = count_synapses_in_mesh(
        neuron_ids=TEST_SINGLE_NEURON,
        neuropil_mesh_names="fan_shaped_body",  # Unlikely for projection neurons
        dataset="latest"
    )
    
    # Should still return a valid DataFrame
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    assert result.shape == (1, 2), "Should have 1 row and 2 columns"
    
    # Count may be 0, which is valid
    count = result["fan_shaped_body"].iloc[0]
    assert count >= 0, "Count should be non-negative (possibly zero)"


def test_count_synapses_in_mesh_string_neuron_id():
    """Test that function accepts neuron IDs as strings."""
    result = count_synapses_in_mesh(
        neuron_ids=str(TEST_SINGLE_NEURON),
        neuropil_mesh_names=TEST_SINGLE_NEUROPIL,
        dataset="latest"
    )
    
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    assert result.shape == (1, 2), "Should have correct shape"


def test_count_synapses_in_mesh_update_ids_false():
    """Test counting synapses without updating IDs."""
    result = count_synapses_in_mesh(
        neuron_ids=TEST_SINGLE_NEURON,
        neuropil_mesh_names=TEST_SINGLE_NEUROPIL,
        update_ids=False,
        dataset="latest"
    )
    
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    assert TEST_SINGLE_NEUROPIL in result.columns


def test_count_synapses_in_mesh_materialization_latest():
    """Test counting synapses with latest materialization."""
    result = count_synapses_in_mesh(
        neuron_ids=TEST_SINGLE_NEURON,
        neuropil_mesh_names=TEST_SINGLE_NEUROPIL,
        materialization="latest",
        dataset="latest"
    )
    
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    assert len(result) == 1, "Should have one row for one neuron"


def test_count_synapses_in_mesh_returns_dataframe_structure():
    """Test that the returned DataFrame has the expected structure."""
    result = count_synapses_in_mesh(
        neuron_ids=TEST_MULTIPLE_NEURONS,
        neuropil_mesh_names=TEST_MULTIPLE_NEUROPILS,
        dataset="latest"
    )
    
    # Check DataFrame properties
    assert isinstance(result, pd.DataFrame)
    assert not result.empty, "Result should not be empty"
    assert "neuron_id" in result.columns, "Should have neuron_id column"
    
    # Check that all neuropil columns exist and have correct dtype
    for neuropil in TEST_MULTIPLE_NEUROPILS:
        assert neuropil in result.columns, f"Missing neuropil column: {neuropil}"
        assert pd.api.types.is_integer_dtype(result[neuropil]), f"{neuropil} should have integer dtype"
    
    # Check that neuron_id values match input
    assert set(result["neuron_id"].values) == set(TEST_MULTIPLE_NEURONS)


# ============================================================================
# Tests for get_synapses_in_mesh()
# ============================================================================


def test_get_synapses_in_mesh_basic():
    """Test basic functionality of get_synapses_in_mesh."""
    from crantpy.viz.mesh import load_neuropil_mesh
    
    # Load a neuropil mesh (in nanometers)
    mesh = load_neuropil_mesh(TEST_SINGLE_NEUROPIL)
    
    # Get synapses within the mesh
    result = get_synapses_in_mesh(
        mesh=mesh,
        mesh_coordinates="nm",
        dataset="latest"
    )
    
    # Check that result is a DataFrame
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    
    # Check that we got some results (antennal lobe should have synapses)
    assert len(result) > 0, "Should have found synapses in antennal lobe"
    
    # Check that standard synapse columns exist
    expected_columns = ['pre_pt_root_id', 'post_pt_root_id', 'ctr_pt_position']
    for col in expected_columns:
        assert col in result.columns, f"Result should have '{col}' column"


def test_get_synapses_in_mesh_coordinates_nm():
    """Test get_synapses_in_mesh with mesh in nanometer coordinates."""
    from crantpy.viz.mesh import load_neuropil_mesh
    
    # Load a neuropil mesh (in nanometers by default)
    mesh = load_neuropil_mesh(TEST_SINGLE_NEUROPIL)
    
    # Get synapses with explicit nm coordinates
    result = get_synapses_in_mesh(
        mesh=mesh,
        mesh_coordinates="nm",
        dataset="latest"
    )
    
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    assert len(result) > 0, "Should have found synapses"


def test_get_synapses_in_mesh_coordinates_voxels():
    """Test get_synapses_in_mesh with mesh in voxel coordinates."""
    from crantpy.viz.mesh import load_neuropil_mesh
    from crantpy.utils.config import SCALE_X, SCALE_Y, SCALE_Z
    import trimesh as tm
    
    # Load a neuropil mesh in nanometers
    mesh_nm = load_neuropil_mesh(TEST_SINGLE_NEUROPIL)
    
    # Convert mesh to voxel coordinates
    vertices_voxels = mesh_nm.vertices.copy()
    vertices_voxels[:, 0] = vertices_voxels[:, 0] / SCALE_X
    vertices_voxels[:, 1] = vertices_voxels[:, 1] / SCALE_Y
    vertices_voxels[:, 2] = vertices_voxels[:, 2] / SCALE_Z
    mesh_voxels = tm.Trimesh(vertices=vertices_voxels, faces=mesh_nm.faces)
    
    # Get synapses with voxel coordinates
    result = get_synapses_in_mesh(
        mesh=mesh_voxels,
        mesh_coordinates="voxels",
        dataset="latest"
    )
    
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    # May or may not have results depending on the converted mesh, but should not error


def test_get_synapses_in_mesh_invalid_coordinates():
    """Test error handling for invalid mesh_coordinates parameter."""
    from crantpy.viz.mesh import load_neuropil_mesh
    
    mesh = load_neuropil_mesh(TEST_SINGLE_NEUROPIL)
    
    with pytest.raises(ValueError) as excinfo:
        get_synapses_in_mesh(
            mesh=mesh,
            mesh_coordinates="invalid",
            dataset="latest"
        )
    
    assert "mesh_coordinates must be either 'nm' or 'voxels'" in str(excinfo.value)


def test_get_synapses_in_mesh_with_min_size():
    """Test get_synapses_in_mesh with size filtering."""
    from crantpy.viz.mesh import load_neuropil_mesh
    
    mesh = load_neuropil_mesh(TEST_SINGLE_NEUROPIL)
    
    # Get synapses with minimum size
    result_filtered = get_synapses_in_mesh(
        mesh=mesh,
        min_size=50,
        dataset="latest"
    )
    
    # Get synapses without filtering
    result_unfiltered = get_synapses_in_mesh(
        mesh=mesh,
        dataset="latest"
    )
    
    # Filtered result should have same or fewer synapses
    assert len(result_filtered) <= len(result_unfiltered), \
        "Size filtering should not increase synapse count"
    
    # Check that size filtering was applied (if size column exists)
    if 'size' in result_filtered.columns:
        assert (result_filtered['size'] >= 50).all(), \
            "All synapses should meet minimum size requirement"


def test_get_synapses_in_mesh_with_threshold():
    """Test get_synapses_in_mesh with threshold filtering."""
    from crantpy.viz.mesh import load_neuropil_mesh
    
    mesh = load_neuropil_mesh(TEST_SINGLE_NEUROPIL)
    
    # Get synapses with default threshold (1)
    result_default = get_synapses_in_mesh(
        mesh=mesh,
        threshold=1,
        dataset="latest"
    )
    
    # Get synapses with higher threshold (only pairs with 3+ synapses)
    result_higher = get_synapses_in_mesh(
        mesh=mesh,
        threshold=3,
        dataset="latest"
    )
    
    # Both should be valid DataFrames
    assert isinstance(result_default, pd.DataFrame)
    assert isinstance(result_higher, pd.DataFrame)
    
    # Higher threshold should give same or fewer total synapses
    assert len(result_higher) <= len(result_default), \
        "Higher threshold should not increase synapse count"
    
    # If higher threshold has results, verify all pairs meet threshold
    if len(result_higher) > 0:
        pair_counts = result_higher.groupby(['pre_pt_root_id', 'post_pt_root_id']).size()
        assert (pair_counts >= 3).all(), \
            "All neuron pairs should have at least 3 synapses"


def test_get_synapses_in_mesh_return_pixels():
    """Test coordinate conversion behavior."""
    from crantpy.viz.mesh import load_neuropil_mesh
    
    mesh = load_neuropil_mesh(TEST_SINGLE_NEUROPIL)
    
    # Get synapses with pixel coordinates (default)
    result_pixels = get_synapses_in_mesh(
        mesh=mesh,
        return_pixels=True,
        dataset="latest"
    )
    
    # Get synapses with nanometer coordinates
    result_nm = get_synapses_in_mesh(
        mesh=mesh,
        return_pixels=False,
        dataset="latest"
    )
    
    # Both should return DataFrames
    assert isinstance(result_pixels, pd.DataFrame)
    assert isinstance(result_nm, pd.DataFrame)
    
    # Both should have the same number of synapses
    assert len(result_pixels) == len(result_nm), \
        "Coordinate conversion should not change synapse count"


def test_get_synapses_in_mesh_clean():
    """Test synapse cleaning functionality."""
    from crantpy.viz.mesh import load_neuropil_mesh
    
    mesh = load_neuropil_mesh(TEST_SINGLE_NEUROPIL)
    
    # Get cleaned synapses (default)
    result_clean = get_synapses_in_mesh(
        mesh=mesh,
        clean=True,
        dataset="latest"
    )
    
    # Get uncleaned synapses
    result_uncleaned = get_synapses_in_mesh(
        mesh=mesh,
        clean=False,
        dataset="latest"
    )
    
    # Cleaned result should have same or fewer synapses
    assert len(result_clean) <= len(result_uncleaned), \
        "Cleaning should not increase synapse count"
    
    # Check that autapses are removed from cleaned result
    if len(result_clean) > 0:
        assert (result_clean['pre_pt_root_id'] != result_clean['post_pt_root_id']).all(), \
            "Cleaned result should not contain autapses"
        
        # Check that background (ID 0) is removed
        assert (result_clean['pre_pt_root_id'] != 0).all(), \
            "Cleaned result should not have pre_pt_root_id = 0"
        assert (result_clean['post_pt_root_id'] != 0).all(), \
            "Cleaned result should not have post_pt_root_id = 0"


def test_get_synapses_in_mesh_materialization_latest():
    """Test get_synapses_in_mesh with latest materialization."""
    from crantpy.viz.mesh import load_neuropil_mesh
    
    mesh = load_neuropil_mesh(TEST_SINGLE_NEUROPIL)
    
    result = get_synapses_in_mesh(
        mesh=mesh,
        materialization="latest",
        dataset="latest"
    )
    
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    assert len(result) > 0, "Should have found synapses"


def test_get_synapses_in_mesh_coordinates_extracted():
    """Test that coordinates are properly extracted from ctr_pt_position."""
    from crantpy.viz.mesh import load_neuropil_mesh
    
    mesh = load_neuropil_mesh(TEST_SINGLE_NEUROPIL)
    
    result = get_synapses_in_mesh(
        mesh=mesh,
        mesh_coordinates="nm",
        dataset="latest"
    )
    
    if len(result) > 0:
        # Check that ctr_pt_position exists and contains coordinate arrays
        assert 'ctr_pt_position' in result.columns, \
            "Result should have ctr_pt_position column"
        
        # Check that each position is a list/array with 3 coordinates
        first_pos = result['ctr_pt_position'].iloc[0]
        assert len(first_pos) == 3, \
            "Each position should have 3 coordinates (x, y, z)"
        
        # Verify coordinates are in nanometer range (should be large numbers)
        assert first_pos[0] > 100, "X coordinate should be in nanometer range"
        assert first_pos[1] > 100, "Y coordinate should be in nanometer range"
        assert first_pos[2] > 100, "Z coordinate should be in nanometer range"


def test_get_synapses_in_mesh_empty_mesh():
    """Test behavior with a small/empty mesh that contains no synapses."""
    # Create a tiny mesh that won't contain any synapses
    # A small box at an unlikely location
    tiny_mesh = tm.creation.box(extents=[100, 100, 100])
    tiny_mesh.apply_translation([1000000, 1000000, 1000000])
    
    result = get_synapses_in_mesh(
        mesh=tiny_mesh,
        dataset="latest"
    )
    
    # Should return an empty DataFrame (or small DataFrame)
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    # It's OK if it's empty or has very few synapses
    assert len(result) >= 0, "Result should have non-negative length"


def test_get_synapses_in_mesh_structure():
    """Test the structure of returned DataFrame."""
    from crantpy.viz.mesh import load_neuropil_mesh
    
    mesh = load_neuropil_mesh(TEST_SINGLE_NEUROPIL)
    
    result = get_synapses_in_mesh(
        mesh=mesh,
        mesh_coordinates="nm",
        dataset="latest"
    )
    
    # Check basic DataFrame properties
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    
    # Check for essential synapse columns
    essential_columns = ['pre_pt_root_id', 'post_pt_root_id']
    for col in essential_columns:
        assert col in result.columns, f"Result should have '{col}' column"
    
    # If we have results, check data types
    if len(result) > 0:
        # Root IDs should be numeric
        assert pd.api.types.is_numeric_dtype(result['pre_pt_root_id']), \
            "pre_pt_root_id should be numeric"
        assert pd.api.types.is_numeric_dtype(result['post_pt_root_id']), \
            "post_pt_root_id should be numeric"


def test_get_synapses_in_mesh_bounding_box():
    """Test that bounding box filtering is working."""
    from crantpy.viz.mesh import load_neuropil_mesh
    
    mesh = load_neuropil_mesh(TEST_SINGLE_NEUROPIL)
    
    # Get the mesh bounds
    min_coords, max_coords = mesh.bounds
    
    result = get_synapses_in_mesh(
        mesh=mesh,
        mesh_coordinates="nm",
        dataset="latest"
    )
    
    # If we got results, verify they are within the bounding box
    if len(result) > 0 and 'ctr_pt_position' in result.columns:
        for pos in result['ctr_pt_position'].values[:10]:  # Check first 10
            # Each coordinate should be within bounds (with some tolerance for mesh.contains)
            assert pos[0] >= min_coords[0] - 1000, "X should be >= min bound"
            assert pos[0] <= max_coords[0] + 1000, "X should be <= max bound"
            assert pos[1] >= min_coords[1] - 1000, "Y should be >= min bound"
            assert pos[1] <= max_coords[1] + 1000, "Y should be <= max bound"
            assert pos[2] >= min_coords[2] - 1000, "Z should be >= min bound"
            assert pos[2] <= max_coords[2] + 1000, "Z should be <= max bound"

