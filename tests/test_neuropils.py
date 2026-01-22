# -*- coding: utf-8 -*-
"""Tests for the neuropils module."""

import pytest
import pandas as pd
import numpy as np
import trimesh as tm
from unittest.mock import Mock, patch, MagicMock, create_autospec
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


# Mock synapse data for testing
def create_mock_synapse_df(num_synapses=100):
    """Create a mock synapse DataFrame for testing."""
    return pd.DataFrame({
        'pre_pt_root_id': np.random.randint(100000, 999999, num_synapses),
        'post_pt_root_id': np.random.randint(100000, 999999, num_synapses),
        'ctr_pt_position': [[np.random.randint(10000, 50000) for _ in range(3)] for _ in range(num_synapses)],
        'size': np.random.randint(10, 100, num_synapses)
    })


def create_mock_mesh_with_contains(inside_count, total=60):
    """Create a mock mesh with a contains method that returns specific results."""
    mock_mesh = MagicMock()
    mock_mesh.contains.return_value = np.array([True] * inside_count + [False] * (total - inside_count))
    return mock_mesh


def create_synapse_data_for_neurons(neuron_ids, synapses_per_neuron=30):
    """Create mock synapse data for specified neurons."""
    all_pre = []
    all_post = []
    all_pos = []
    
    for neuron_id in neuron_ids:
        all_pre.extend([neuron_id] * synapses_per_neuron)
        all_post.extend([888888] * synapses_per_neuron)
        all_pos.extend([[10000, 20000, 30000] for _ in range(synapses_per_neuron)])
    
    return pd.DataFrame({
        'pre_pt_root_id': all_pre,
        'post_pt_root_id': all_post,
        'pre_pt_position': all_pos,
    })


@patch('crantpy.queries.neuropils.load_neuropil_mesh')
@patch('crantpy.queries.neuropils.get_synapses')
def test_count_synapses_in_mesh_single_neuron_single_neuropil(mock_get_synapses, mock_load_mesh):
    """Test counting synapses for a single neuron in a single neuropil."""
    # Mock mesh with contains method
    mock_mesh = MagicMock()
    # Simulate 50 synapses inside mesh, rest outside
    mock_mesh.contains.return_value = np.array([True] * 50 + [False] * 10)
    mock_load_mesh.return_value = mock_mesh
    
    # Mock synapse data with positions
    mock_synapses = pd.DataFrame({
        'pre_pt_root_id': [TEST_SINGLE_NEURON] * 60,
        'post_pt_root_id': [888888] * 60,
        'pre_pt_position': [[10000, 20000, 30000] for _ in range(60)],
    })
    mock_get_synapses.return_value = mock_synapses
    
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
    
    # Verify the count is correct (50 inside)
    assert result[TEST_SINGLE_NEUROPIL].iloc[0] == 50


@patch('crantpy.queries.neuropils.load_neuropil_mesh')
@patch('crantpy.queries.neuropils.get_synapses')
def test_count_synapses_in_mesh_single_neuron_multiple_neuropils(mock_get_synapses, mock_load_mesh):
    """Test counting synapses for a single neuron across multiple neuropils."""
    # Mock mesh with different inside counts for each mesh
    def create_mesh_with_contains(inside_count, total=60):
        mock_mesh = MagicMock()
        mock_mesh.contains.return_value = np.array([True] * inside_count + [False] * (total - inside_count))
        return mock_mesh
    
    # Return different meshes for each call
    mock_load_mesh.side_effect = [
        create_mesh_with_contains(30),
        create_mesh_with_contains(20),
        create_mesh_with_contains(10),
    ]
    
    # Mock synapse data
    mock_synapses = pd.DataFrame({
        'pre_pt_root_id': [TEST_SINGLE_NEURON] * 60,
        'post_pt_root_id': [888888] * 60,
        'pre_pt_position': [[10000, 20000, 30000] for _ in range(60)],
    })
    mock_get_synapses.return_value = mock_synapses
    
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


@patch('crantpy.queries.neuropils.load_neuropil_mesh')
@patch('crantpy.queries.neuropils.get_synapses')
def test_count_synapses_in_mesh_multiple_neurons_single_neuropil(mock_get_synapses, mock_load_mesh):
    """Test counting synapses for multiple neurons in a single neuropil."""
    # Mock mesh
    mock_mesh = MagicMock()
    # First 40 synapses inside, rest outside
    mock_mesh.contains.return_value = np.array([True] * 40 + [False] * 20)
    mock_load_mesh.return_value = mock_mesh
    
    # Mock synapse data with both test neurons
    mock_synapses = pd.DataFrame({
        'pre_pt_root_id': [TEST_MULTIPLE_NEURONS[0]] * 30 + [TEST_MULTIPLE_NEURONS[1]] * 30,
        'post_pt_root_id': [888888] * 60,
        'pre_pt_position': [[10000, 20000, 30000] for _ in range(60)],
    })
    mock_get_synapses.return_value = mock_synapses
    
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


@patch('crantpy.queries.neuropils.load_neuropil_mesh')
@patch('crantpy.queries.neuropils.get_synapses')
def test_count_synapses_in_mesh_multiple_neurons_multiple_neuropils(mock_get_synapses, mock_load_mesh):
    """Test counting synapses for multiple neurons across multiple neuropils."""
    # Mock mesh
    mock_mesh = MagicMock()
    mock_mesh.contains.return_value = np.array([True] * 30 + [False] * 30)
    mock_load_mesh.return_value = mock_mesh
    
    # Mock synapse data
    mock_synapses = pd.DataFrame({
        'pre_pt_root_id': [TEST_MULTIPLE_NEURONS[0]] * 30 + [TEST_MULTIPLE_NEURONS[1]] * 30,
        'post_pt_root_id': [888888] * 60,
        'pre_pt_position': [[10000, 20000, 30000] for _ in range(60)],
    })
    mock_get_synapses.return_value = mock_synapses
    
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


@patch('crantpy.queries.neuropils.load_neuropil_mesh')
def test_count_synapses_in_mesh_invalid_neuropil(mock_load_mesh):
    """Test error handling for invalid neuropil name."""
    # Mock load_neuropil_mesh to raise ValueError for invalid names
    mock_load_mesh.side_effect = ValueError("Invalid neuropil mesh name: invalid_neuropil_name")
    
    with pytest.raises(ValueError) as excinfo:
        count_synapses_in_mesh(
            neuron_ids=TEST_SINGLE_NEURON,
            neuropil_mesh_names="invalid_neuropil_name",
            dataset="latest"
        )
    
    assert "Invalid neuropil mesh name" in str(excinfo.value)


@patch('crantpy.queries.neuropils.load_neuropil_mesh')
@patch('crantpy.queries.neuropils.get_synapses')
def test_count_synapses_in_mesh_with_min_synapses_per_neuron(mock_get_synapses, mock_load_mesh):
    """Test counting synapses with min_synapses_per_neuron filtering."""
    # Mock synapse data with multiple neurons - both have >= 20 synapses
    mock_synapses = pd.DataFrame({
        'pre_pt_root_id': [TEST_MULTIPLE_NEURONS[0]] * 30 + [TEST_MULTIPLE_NEURONS[1]] * 30,
        'post_pt_root_id': [888888] * 60,
        'pre_pt_position': [[10000, 20000, 30000] for _ in range(60)],
    })
    mock_get_synapses.return_value = mock_synapses
    
    # Mock mesh with 60 synapses (all inside)
    mock_mesh = create_mock_mesh_with_contains(inside_count=60, total=60)
    mock_load_mesh.return_value = mock_mesh
    
    # Test with min_synapses_per_neuron=1 - should include both neurons
    result_permissive = count_synapses_in_mesh(
        neuron_ids=TEST_MULTIPLE_NEURONS,
        neuropil_mesh_names=TEST_SINGLE_NEUROPIL,
        min_synapses_per_neuron=1,
        dataset="latest"
    )
    
    # Both should be valid DataFrames
    assert isinstance(result_permissive, pd.DataFrame)
    assert len(result_permissive) > 0, "Should have rows for valid neurons"


@patch('crantpy.queries.neuropils.load_neuropil_mesh')
@patch('crantpy.queries.neuropils.get_synapses')
def test_count_synapses_in_mesh_with_min_synapses_per_pair(mock_get_synapses, mock_load_mesh):
    """Test counting synapses with min_synapses_per_pair filtering."""
    # Mock synapse data with multiple connections, all with >= 25 synapses per pair
    mock_synapses = pd.DataFrame({
        'pre_pt_root_id': [TEST_SINGLE_NEURON] * 60,
        'post_pt_root_id': [888888] * 30 + [999999] * 30,  # Two different postsynaptic neurons
        'pre_pt_position': [[10000, 20000, 30000] for _ in range(60)],
    })
    mock_get_synapses.return_value = mock_synapses
    
    # Mock mesh with 60 synapses (all inside)
    mock_mesh = create_mock_mesh_with_contains(inside_count=60, total=60)
    mock_load_mesh.return_value = mock_mesh
    
    # Test with min_synapses_per_pair=1 - should include both pairs
    result_permissive = count_synapses_in_mesh(
        neuron_ids=TEST_SINGLE_NEURON,
        neuropil_mesh_names=TEST_SINGLE_NEUROPIL,
        min_synapses_per_pair=1,
        dataset="latest"
    )
    
    # Both should be valid DataFrames
    assert isinstance(result_permissive, pd.DataFrame)


@patch('crantpy.queries.neuropils.load_neuropil_mesh')
@patch('crantpy.queries.neuropils.get_synapses')
def test_count_synapses_in_mesh_with_neuron_criteria(mock_get_synapses, mock_load_mesh):
    """Test counting synapses using NeuronCriteria."""
    # Mock mesh
    mock_mesh = create_mock_mesh_with_contains(inside_count=40, total=90)
    mock_load_mesh.return_value = mock_mesh
    
    # Mock synapses for multiple neurons
    neuron_ids = [111111, 222222, 333333]
    mock_synapses = create_synapse_data_for_neurons(neuron_ids, synapses_per_neuron=30)
    mock_get_synapses.return_value = mock_synapses
    
    # Create mock NeuronCriteria that properly returns the neuron IDs
    mock_criteria = MagicMock(spec=NeuronCriteria)
    # Make it pass the type check
    mock_criteria.__class__ = NeuronCriteria
    # Make get_roots() return the list
    mock_criteria.get_roots.return_value = neuron_ids
    # Make is_empty return False so it passes validation
    mock_criteria.is_empty = False
    
    result = count_synapses_in_mesh(
        neuron_ids=mock_criteria,
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


@patch('crantpy.queries.neuropils.load_neuropil_mesh')
@patch('crantpy.queries.neuropils.get_synapses')
def test_count_synapses_in_mesh_no_synapses(mock_get_synapses, mock_load_mesh):
    """Test behavior when neuron has no synapses in the specified neuropil."""
    # Mock mesh
    mock_mesh = MagicMock()
    mock_mesh.contains.return_value = np.array([])  # No synapses
    mock_load_mesh.return_value = mock_mesh
    
    # Mock empty synapse DataFrame
    mock_synapses = pd.DataFrame({
        'pre_pt_root_id': [],
        'post_pt_root_id': [],
        'pre_pt_position': [],
    })
    mock_get_synapses.return_value = mock_synapses
    
    # Use a neuron and neuropil combination unlikely to have overlapping synapses
    result = count_synapses_in_mesh(
        neuron_ids=TEST_SINGLE_NEURON,
        neuropil_mesh_names="fan_shaped_body",  # Unlikely for projection neurons
        dataset="latest"
    )
    
    # Should still return a valid DataFrame
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    # When there are no synapses, the result is reset_index() called on the initialized DataFrame
    # which converts the neuron_id index to a column
    assert len(result) == 1, "Should have 1 row"
    # The result has neuron_id and the neuropil column
    assert "neuron_id" in result.columns or result.index.name == 'neuron_id', "Should have neuron_id"
    assert "fan_shaped_body" in result.columns, "Should have neuropil column"
    
    # Check the count
    count = result["fan_shaped_body"].iloc[0]
    assert count == 0, "Count should be zero when no synapses found"


@patch('crantpy.queries.neuropils.load_neuropil_mesh')
@patch('crantpy.queries.neuropils.get_synapses')
def test_count_synapses_in_mesh_string_neuron_id(mock_get_synapses, mock_load_mesh):
    """Test that function accepts neuron IDs as strings."""
    # Mock mesh
    mock_mesh = create_mock_mesh_with_contains(inside_count=30, total=30)
    mock_load_mesh.return_value = mock_mesh
    
    # Mock synapse data
    mock_synapses = create_synapse_data_for_neurons([TEST_SINGLE_NEURON], synapses_per_neuron=30)
    mock_get_synapses.return_value = mock_synapses
    
    result = count_synapses_in_mesh(
        neuron_ids=str(TEST_SINGLE_NEURON),
        neuropil_mesh_names=TEST_SINGLE_NEUROPIL,
        dataset="latest"
    )
    
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    assert result.shape == (1, 2), "Should have correct shape"


@patch('crantpy.queries.neuropils.load_neuropil_mesh')
@patch('crantpy.queries.neuropils.get_synapses')
def test_count_synapses_in_mesh_update_ids_false(mock_get_synapses, mock_load_mesh):
    """Test counting synapses without updating IDs."""
    # Mock mesh
    mock_mesh = create_mock_mesh_with_contains(inside_count=30, total=30)
    mock_load_mesh.return_value = mock_mesh
    
    # Mock synapse data
    mock_synapses = create_synapse_data_for_neurons([TEST_SINGLE_NEURON], synapses_per_neuron=30)
    mock_get_synapses.return_value = mock_synapses
    
    result = count_synapses_in_mesh(
        neuron_ids=TEST_SINGLE_NEURON,
        neuropil_mesh_names=TEST_SINGLE_NEUROPIL,
        update_ids=False,
        dataset="latest"
    )
    
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    assert TEST_SINGLE_NEUROPIL in result.columns


@patch('crantpy.queries.neuropils.load_neuropil_mesh')
@patch('crantpy.queries.neuropils.get_synapses')
def test_count_synapses_in_mesh_materialization_latest(mock_get_synapses, mock_load_mesh):
    """Test counting synapses with latest materialization."""
    # Mock mesh
    mock_mesh = create_mock_mesh_with_contains(inside_count=30, total=30)
    mock_load_mesh.return_value = mock_mesh
    
    # Mock synapse data
    mock_synapses = create_synapse_data_for_neurons([TEST_SINGLE_NEURON], synapses_per_neuron=30)
    mock_get_synapses.return_value = mock_synapses
    
    result = count_synapses_in_mesh(
        neuron_ids=TEST_SINGLE_NEURON,
        neuropil_mesh_names=TEST_SINGLE_NEUROPIL,
        materialization="latest",
        dataset="latest"
    )
    
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    assert len(result) == 1, "Should have one row for one neuron"


@patch('crantpy.queries.neuropils.load_neuropil_mesh')
@patch('crantpy.queries.neuropils.get_synapses')
def test_count_synapses_in_mesh_returns_dataframe_structure(mock_get_synapses, mock_load_mesh):
    """Test that the returned DataFrame has the expected structure."""
    # Mock mesh
    mock_mesh = create_mock_mesh_with_contains(inside_count=50, total=60)
    mock_load_mesh.return_value = mock_mesh
    
    # Mock synapse data
    mock_synapses = create_synapse_data_for_neurons(TEST_MULTIPLE_NEURONS, synapses_per_neuron=30)
    mock_get_synapses.return_value = mock_synapses
    
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
# Note: Only validation tests are included. Full functional tests would  
# require mocking the entire CAVE client infrastructure and are considered
# integration tests rather than unit tests.
# ============================================================================


def test_get_synapses_in_mesh_invalid_coordinates():
    """Test error handling for invalid mesh_coordinates parameter."""
    mock_mesh = tm.creation.box(extents=[10000, 10000, 10000])
    
    with pytest.raises(ValueError) as excinfo:
        get_synapses_in_mesh(
            mesh=mock_mesh,
            mesh_coordinates="invalid",
            dataset="latest"
        )
    
    assert "mesh_coordinates must be either 'nm' or 'voxels'" in str(excinfo.value)




