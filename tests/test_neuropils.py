# -*- coding: utf-8 -*-
"""Tests for the neuropils module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from crantpy.queries.neuropils import count_synapses_in_mesh
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
