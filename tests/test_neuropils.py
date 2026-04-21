# -*- coding: utf-8 -*-
"""Tests for the neuropils module."""

import logging
import time

import pytest
import pandas as pd
import numpy as np
import requests
import trimesh as tm
from unittest.mock import patch, MagicMock
import crantpy.queries.neuropils as neuropils_module
from crantpy.queries.neurons import NeuronCriteria
from crantpy.queries.neuropils import (
    count_synapses_in_mesh,
    get_synapses_in_mesh,
    get_synapses_in_neuropils,
    _query_with_breaker,
    _query_synapses_in_bbox,
    _cave_breaker,
    CAVE_ROW_LIMIT,
)


@pytest.fixture(autouse=True)
def _reset_circuit_breaker():
    """Reset the shared circuit breaker before each test."""
    _cave_breaker._state = _cave_breaker.CLOSED
    _cave_breaker._failure_count = 0
    _cave_breaker._last_failure_time = None
    yield


@pytest.fixture(autouse=True)
def _stub_update_ids(monkeypatch):
    """Keep neuropil tests off the real ID-update path unless a test overrides it."""

    def _identity_update_ids(x, *args, **kwargs):
        ids = np.atleast_1d(np.asarray(x, dtype=np.int64))
        return pd.DataFrame(
            {
                "old_id": ids,
                "new_id": ids,
                "confidence": np.ones(len(ids), dtype=float),
                "changed": np.zeros(len(ids), dtype=bool),
            }
        )

    monkeypatch.setattr(
        "crantpy.utils.cave.segmentation.update_ids",
        _identity_update_ids,
    )


# Test neuron IDs - using known projection neurons that should have synapses
TEST_SINGLE_NEURON = 576460752664524086
TEST_MULTIPLE_NEURONS = [576460752664524086, 576460752662516321]

# Test neuropil mesh names
TEST_SINGLE_NEUROPIL = "antennal_lobe_right"
TEST_MULTIPLE_NEUROPILS = [
    "antennal_lobe_right",
    "mushroom_body_lateral_calyx_right",
    "lateral_horn_right",
]


# Mock synapse data for testing
def create_mock_synapse_df(num_synapses=100):
    """Create a mock synapse DataFrame for testing."""
    return pd.DataFrame(
        {
            "pre_pt_root_id": np.random.randint(100000, 999999, num_synapses),
            "post_pt_root_id": np.random.randint(100000, 999999, num_synapses),
            "ctr_pt_position": [
                [np.random.randint(10000, 50000) for _ in range(3)]
                for _ in range(num_synapses)
            ],
            "size": np.random.randint(10, 100, num_synapses),
        }
    )


def create_mock_mesh_with_contains(inside_count, total=60):
    """Create a mock mesh with a contains method that returns specific results."""
    mock_mesh = MagicMock()
    mock_mesh.contains.return_value = np.array(
        [True] * inside_count + [False] * (total - inside_count)
    )
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

    return pd.DataFrame(
        {
            "pre_pt_root_id": all_pre,
            "post_pt_root_id": all_post,
            "pre_pt_position": all_pos,
        }
    )


@patch("crantpy.queries.neuropils.load_neuropil_mesh")
@patch("crantpy.queries.neuropils.get_synapses")
def test_count_synapses_in_mesh_single_neuron_single_neuropil(
    mock_get_synapses, mock_load_mesh
):
    """Test counting synapses for a single neuron in a single neuropil."""
    # Mock mesh with contains method
    mock_mesh = MagicMock()
    # Simulate 50 synapses inside mesh, rest outside
    mock_mesh.contains.return_value = np.array([True] * 50 + [False] * 10)
    mock_load_mesh.return_value = mock_mesh

    # Mock synapse data with positions
    mock_synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [TEST_SINGLE_NEURON] * 60,
            "post_pt_root_id": [888888] * 60,
            "pre_pt_position": [[10000, 20000, 30000] for _ in range(60)],
        }
    )
    mock_get_synapses.return_value = mock_synapses

    result = count_synapses_in_mesh(
        neuron_ids=TEST_SINGLE_NEURON,
        neuropil_mesh_names=TEST_SINGLE_NEUROPIL,
        dataset="latest",
    )

    # Check that result is a DataFrame
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"

    # Check that DataFrame has correct shape (1 neuron x 1 neuropil)
    assert result.shape == (
        1,
        2,
    ), f"Expected shape (1, 2), got {result.shape}"  # neuron_id + 1 neuropil column

    # Check that neuron_id column exists
    assert "neuron_id" in result.columns, "Result should have 'neuron_id' column"

    # Check that neuropil column exists
    assert (
        TEST_SINGLE_NEUROPIL in result.columns
    ), f"Result should have '{TEST_SINGLE_NEUROPIL}' column"

    # Check that values are integers (synapse counts)
    assert (
        result[TEST_SINGLE_NEUROPIL].dtype == np.int64
        or result[TEST_SINGLE_NEUROPIL].dtype == int
    )

    # Check that counts are non-negative
    assert (
        result[TEST_SINGLE_NEUROPIL] >= 0
    ).all(), "Synapse counts should be non-negative"

    # Verify the count is correct (50 inside)
    assert result[TEST_SINGLE_NEUROPIL].iloc[0] == 50


@patch("crantpy.queries.neuropils.load_neuropil_mesh")
@patch("crantpy.queries.neuropils.get_synapses")
def test_count_synapses_in_mesh_single_neuron_multiple_neuropils(
    mock_get_synapses, mock_load_mesh
):
    """Test counting synapses for a single neuron across multiple neuropils."""

    # Mock mesh with different inside counts for each mesh
    def create_mesh_with_contains(inside_count, total=60):
        mock_mesh = MagicMock()
        mock_mesh.contains.return_value = np.array(
            [True] * inside_count + [False] * (total - inside_count)
        )
        return mock_mesh

    # Return different meshes for each call
    mock_load_mesh.side_effect = [
        create_mesh_with_contains(30),
        create_mesh_with_contains(20),
        create_mesh_with_contains(10),
    ]

    # Mock synapse data
    mock_synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [TEST_SINGLE_NEURON] * 60,
            "post_pt_root_id": [888888] * 60,
            "pre_pt_position": [[10000, 20000, 30000] for _ in range(60)],
        }
    )
    mock_get_synapses.return_value = mock_synapses

    result = count_synapses_in_mesh(
        neuron_ids=TEST_SINGLE_NEURON,
        neuropil_mesh_names=TEST_MULTIPLE_NEUROPILS,
        dataset="latest",
    )

    # Check that result is a DataFrame
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"

    # Check that DataFrame has correct shape (1 neuron x N neuropils)
    expected_cols = len(TEST_MULTIPLE_NEUROPILS) + 1  # +1 for neuron_id
    assert result.shape == (
        1,
        expected_cols,
    ), f"Expected shape (1, {expected_cols}), got {result.shape}"

    # Check that all neuropil columns exist
    for neuropil_name in TEST_MULTIPLE_NEUROPILS:
        assert (
            neuropil_name in result.columns
        ), f"Result should have '{neuropil_name}' column"

    # Check that all counts are non-negative
    for neuropil_name in TEST_MULTIPLE_NEUROPILS:
        assert (
            result[neuropil_name] >= 0
        ).all(), f"Counts for {neuropil_name} should be non-negative"


@patch("crantpy.queries.neuropils.load_neuropil_mesh")
@patch("crantpy.queries.neuropils.get_synapses")
def test_count_synapses_in_mesh_multiple_neurons_single_neuropil(
    mock_get_synapses, mock_load_mesh
):
    """Test counting synapses for multiple neurons in a single neuropil."""
    # Mock mesh
    mock_mesh = MagicMock()
    # First 40 synapses inside, rest outside
    mock_mesh.contains.return_value = np.array([True] * 40 + [False] * 20)
    mock_load_mesh.return_value = mock_mesh

    # Mock synapse data with both test neurons
    mock_synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [TEST_MULTIPLE_NEURONS[0]] * 30
            + [TEST_MULTIPLE_NEURONS[1]] * 30,
            "post_pt_root_id": [888888] * 60,
            "pre_pt_position": [[10000, 20000, 30000] for _ in range(60)],
        }
    )
    mock_get_synapses.return_value = mock_synapses

    result = count_synapses_in_mesh(
        neuron_ids=TEST_MULTIPLE_NEURONS,
        neuropil_mesh_names=TEST_SINGLE_NEUROPIL,
        dataset="latest",
    )

    # Check that result is a DataFrame
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"

    # Check that DataFrame has correct shape (N neurons x 1 neuropil)
    assert result.shape == (
        len(TEST_MULTIPLE_NEURONS),
        2,
    ), f"Expected shape ({len(TEST_MULTIPLE_NEURONS)}, 2)"

    # Check that all neurons are represented
    assert len(result) == len(TEST_MULTIPLE_NEURONS), "All neurons should be in result"

    # Check that neuron IDs are correct
    result_ids = set(result["neuron_id"].values)
    expected_ids = set(TEST_MULTIPLE_NEURONS)
    assert (
        result_ids == expected_ids
    ), f"Neuron IDs mismatch: {result_ids} vs {expected_ids}"


@patch("crantpy.queries.neuropils.load_neuropil_mesh")
@patch("crantpy.queries.neuropils.get_synapses")
def test_count_synapses_in_mesh_multiple_neurons_multiple_neuropils(
    mock_get_synapses, mock_load_mesh
):
    """Test counting synapses for multiple neurons across multiple neuropils."""
    # Mock mesh
    mock_mesh = MagicMock()
    mock_mesh.contains.return_value = np.array([True] * 30 + [False] * 30)
    mock_load_mesh.return_value = mock_mesh

    # Mock synapse data
    mock_synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [TEST_MULTIPLE_NEURONS[0]] * 30
            + [TEST_MULTIPLE_NEURONS[1]] * 30,
            "post_pt_root_id": [888888] * 60,
            "pre_pt_position": [[10000, 20000, 30000] for _ in range(60)],
        }
    )
    mock_get_synapses.return_value = mock_synapses

    result = count_synapses_in_mesh(
        neuron_ids=TEST_MULTIPLE_NEURONS,
        neuropil_mesh_names=TEST_MULTIPLE_NEUROPILS,
        dataset="latest",
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
        assert (
            result[neuropil_name] >= 0
        ).all(), f"Negative counts found in {neuropil_name}"


@patch("crantpy.queries.neuropils.load_neuropil_mesh")
def test_count_synapses_in_mesh_invalid_neuropil(mock_load_mesh):
    """Test error handling for invalid neuropil name."""
    with pytest.raises(ValueError) as excinfo:
        count_synapses_in_mesh(
            neuron_ids=TEST_SINGLE_NEURON,
            neuropil_mesh_names="invalid_neuropil_name",
            dataset="latest",
        )

    assert "Invalid neuropil name" in str(excinfo.value)
    mock_load_mesh.assert_not_called()


@patch("crantpy.queries.neuropils.load_neuropil_mesh")
@patch("crantpy.queries.neuropils.get_synapses")
def test_count_synapses_in_mesh_preserves_existing_positional_arguments(
    mock_get_synapses,
    mock_load_mesh,
):
    """Adding loc should not remap legacy positional materialization arguments."""
    mock_mesh = MagicMock()
    mock_mesh.contains.return_value = np.array([True])
    mock_load_mesh.return_value = mock_mesh
    mock_get_synapses.return_value = pd.DataFrame(
        {
            "pre_pt_root_id": [TEST_SINGLE_NEURON],
            "post_pt_root_id": [888888],
            "pre_pt_position": [[10000, 20000, 30000]],
        }
    )

    count_synapses_in_mesh(
        TEST_SINGLE_NEURON,
        TEST_SINGLE_NEUROPIL,
        1,
        1,
        "live",
        False,
        "latest",
    )

    assert mock_get_synapses.call_args.kwargs["materialization"] == "live"
    assert mock_get_synapses.call_args.kwargs["update_ids"] is False
    assert mock_get_synapses.call_args.kwargs["dataset"] == "latest"


@patch("crantpy.queries.neuropils.load_neuropil_mesh")
@patch("crantpy.queries.neuropils.get_synapses")
def test_count_synapses_in_mesh_uses_ctr_positions_when_requested(
    mock_get_synapses, mock_load_mesh
):
    """loc='ctr' should use ctr_pt_position for mesh containment."""
    mock_mesh = MagicMock()
    mock_mesh.contains.return_value = np.array([True])
    mock_load_mesh.return_value = mock_mesh
    mock_get_synapses.return_value = pd.DataFrame(
        {
            "pre_pt_root_id": [TEST_SINGLE_NEURON],
            "post_pt_root_id": [888888],
            "pre_pt_position": [[10000, 20000, 30000]],
            "ctr_pt_position": [[11000, 21000, 31000]],
        }
    )

    count_synapses_in_mesh(
        neuron_ids=TEST_SINGLE_NEURON,
        neuropil_mesh_names=TEST_SINGLE_NEUROPIL,
        loc="ctr",
        dataset="latest",
    )

    np.testing.assert_array_equal(
        mock_mesh.contains.call_args.args[0],
        np.array([[11000, 21000, 31000]]),
    )


@patch("crantpy.queries.neuropils.load_neuropil_mesh")
@patch("crantpy.queries.neuropils.get_synapses")
def test_count_synapses_in_mesh_accepts_nodulus_right_alias(
    mock_get_synapses, mock_load_mesh
):
    """Compatibility alias should pass validation and load as a mesh name."""
    mock_mesh = MagicMock()
    mock_mesh.contains.return_value = np.array([True, False])
    mock_load_mesh.return_value = mock_mesh
    mock_get_synapses.return_value = pd.DataFrame(
        {
            "pre_pt_root_id": [TEST_SINGLE_NEURON, TEST_SINGLE_NEURON],
            "post_pt_root_id": [888888, 999999],
            "pre_pt_position": [[10000, 20000, 30000], [11000, 21000, 31000]],
        }
    )

    result = count_synapses_in_mesh(
        neuron_ids=TEST_SINGLE_NEURON,
        neuropil_mesh_names="nodulus_right",
        dataset="latest",
    )

    assert "nodulus_right" in result.columns
    assert result["nodulus_right"].iloc[0] == 1
    mock_load_mesh.assert_called_once_with("nodulus_right")


@patch("crantpy.queries.neuropils.load_neuropil_mesh")
@patch("crantpy.queries.neuropils.get_synapses")
def test_count_synapses_in_mesh_with_min_synapses_per_neuron(
    mock_get_synapses, mock_load_mesh
):
    """Test counting synapses with min_synapses_per_neuron filtering."""
    # Mock synapse data with multiple neurons - both have >= 20 synapses
    mock_synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [TEST_MULTIPLE_NEURONS[0]] * 30
            + [TEST_MULTIPLE_NEURONS[1]] * 30,
            "post_pt_root_id": [888888] * 60,
            "pre_pt_position": [[10000, 20000, 30000] for _ in range(60)],
        }
    )
    mock_get_synapses.return_value = mock_synapses

    # Mock mesh with 60 synapses (all inside)
    mock_mesh = create_mock_mesh_with_contains(inside_count=60, total=60)
    mock_load_mesh.return_value = mock_mesh

    # Test with min_synapses_per_neuron=1 - should include both neurons
    result_permissive = count_synapses_in_mesh(
        neuron_ids=TEST_MULTIPLE_NEURONS,
        neuropil_mesh_names=TEST_SINGLE_NEUROPIL,
        min_synapses_per_neuron=1,
        dataset="latest",
    )

    # Both should be valid DataFrames
    assert isinstance(result_permissive, pd.DataFrame)
    assert len(result_permissive) > 0, "Should have rows for valid neurons"


@patch("crantpy.queries.neuropils.load_neuropil_mesh")
@patch("crantpy.queries.neuropils.get_synapses")
def test_count_synapses_in_mesh_with_min_synapses_per_pair(
    mock_get_synapses, mock_load_mesh
):
    """Test counting synapses with min_synapses_per_pair filtering."""
    # Mock synapse data with multiple connections, all with >= 25 synapses per pair
    mock_synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [TEST_SINGLE_NEURON] * 60,
            "post_pt_root_id": [888888] * 30
            + [999999] * 30,  # Two different postsynaptic neurons
            "pre_pt_position": [[10000, 20000, 30000] for _ in range(60)],
        }
    )
    mock_get_synapses.return_value = mock_synapses

    # Mock mesh with 60 synapses (all inside)
    mock_mesh = create_mock_mesh_with_contains(inside_count=60, total=60)
    mock_load_mesh.return_value = mock_mesh

    # Test with min_synapses_per_pair=1 - should include both pairs
    result_permissive = count_synapses_in_mesh(
        neuron_ids=TEST_SINGLE_NEURON,
        neuropil_mesh_names=TEST_SINGLE_NEUROPIL,
        min_synapses_per_pair=1,
        dataset="latest",
    )

    # Both should be valid DataFrames
    assert isinstance(result_permissive, pd.DataFrame)


@patch("crantpy.queries.neuropils.load_neuropil_mesh")
@patch("crantpy.queries.neuropils.get_synapses")
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
        dataset="latest",
    )

    # Check that result is a DataFrame
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"

    # Check that we got results for multiple neurons
    assert len(result) > 0, "Should have results for at least one neuron"

    # Check that all required columns exist
    assert "neuron_id" in result.columns
    assert TEST_SINGLE_NEUROPIL in result.columns


@patch("crantpy.queries.neuropils.load_neuropil_mesh")
@patch("crantpy.queries.neuropils.get_synapses")
def test_count_synapses_in_mesh_no_synapses(mock_get_synapses, mock_load_mesh):
    """Test behavior when neuron has no synapses in the specified neuropil."""
    # Mock mesh
    mock_mesh = MagicMock()
    mock_mesh.contains.return_value = np.array([])  # No synapses
    mock_load_mesh.return_value = mock_mesh

    # Mock empty synapse DataFrame
    mock_synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [],
            "post_pt_root_id": [],
            "pre_pt_position": [],
        }
    )
    mock_get_synapses.return_value = mock_synapses

    # Use a neuron and neuropil combination unlikely to have overlapping synapses
    result = count_synapses_in_mesh(
        neuron_ids=TEST_SINGLE_NEURON,
        neuropil_mesh_names="fan_shaped_body",  # Unlikely for projection neurons
        dataset="latest",
    )

    # Should still return a valid DataFrame
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    # When there are no synapses, the result is reset_index() called on the initialized DataFrame
    # which converts the neuron_id index to a column
    assert len(result) == 1, "Should have 1 row"
    # The result has neuron_id and the neuropil column
    assert (
        "neuron_id" in result.columns or result.index.name == "neuron_id"
    ), "Should have neuron_id"
    assert "fan_shaped_body" in result.columns, "Should have neuropil column"

    # Check the count
    count = result["fan_shaped_body"].iloc[0]
    assert count == 0, "Count should be zero when no synapses found"


@patch("crantpy.queries.neuropils.load_neuropil_mesh")
@patch("crantpy.queries.neuropils.get_synapses")
def test_count_synapses_in_mesh_string_neuron_id(mock_get_synapses, mock_load_mesh):
    """Test that function accepts neuron IDs as strings."""
    # Mock mesh
    mock_mesh = create_mock_mesh_with_contains(inside_count=30, total=30)
    mock_load_mesh.return_value = mock_mesh

    # Mock synapse data
    mock_synapses = create_synapse_data_for_neurons(
        [TEST_SINGLE_NEURON], synapses_per_neuron=30
    )
    mock_get_synapses.return_value = mock_synapses

    result = count_synapses_in_mesh(
        neuron_ids=str(TEST_SINGLE_NEURON),
        neuropil_mesh_names=TEST_SINGLE_NEUROPIL,
        dataset="latest",
    )

    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    assert result.shape == (1, 2), "Should have correct shape"


@patch("crantpy.queries.neuropils.load_neuropil_mesh")
@patch("crantpy.queries.neuropils.get_synapses")
def test_count_synapses_in_mesh_update_ids_false(mock_get_synapses, mock_load_mesh):
    """Test counting synapses without updating IDs."""
    # Mock mesh
    mock_mesh = create_mock_mesh_with_contains(inside_count=30, total=30)
    mock_load_mesh.return_value = mock_mesh

    # Mock synapse data
    mock_synapses = create_synapse_data_for_neurons(
        [TEST_SINGLE_NEURON], synapses_per_neuron=30
    )
    mock_get_synapses.return_value = mock_synapses

    result = count_synapses_in_mesh(
        neuron_ids=TEST_SINGLE_NEURON,
        neuropil_mesh_names=TEST_SINGLE_NEUROPIL,
        update_ids=False,
        dataset="latest",
    )

    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    assert TEST_SINGLE_NEUROPIL in result.columns


@patch("crantpy.queries.neuropils.load_neuropil_mesh")
@patch("crantpy.queries.neuropils.get_synapses")
def test_count_synapses_in_mesh_materialization_latest(
    mock_get_synapses, mock_load_mesh
):
    """Test counting synapses with latest materialization."""
    # Mock mesh
    mock_mesh = create_mock_mesh_with_contains(inside_count=30, total=30)
    mock_load_mesh.return_value = mock_mesh

    # Mock synapse data
    mock_synapses = create_synapse_data_for_neurons(
        [TEST_SINGLE_NEURON], synapses_per_neuron=30
    )
    mock_get_synapses.return_value = mock_synapses

    result = count_synapses_in_mesh(
        neuron_ids=TEST_SINGLE_NEURON,
        neuropil_mesh_names=TEST_SINGLE_NEUROPIL,
        materialization="latest",
        dataset="latest",
    )

    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    assert len(result) == 1, "Should have one row for one neuron"


@patch("crantpy.queries.neuropils.load_neuropil_mesh")
@patch("crantpy.queries.neuropils.get_synapses")
def test_count_synapses_in_mesh_returns_dataframe_structure(
    mock_get_synapses, mock_load_mesh
):
    """Test that the returned DataFrame has the expected structure."""
    # Mock mesh
    mock_mesh = create_mock_mesh_with_contains(inside_count=50, total=60)
    mock_load_mesh.return_value = mock_mesh

    # Mock synapse data
    mock_synapses = create_synapse_data_for_neurons(
        TEST_MULTIPLE_NEURONS, synapses_per_neuron=30
    )
    mock_get_synapses.return_value = mock_synapses

    result = count_synapses_in_mesh(
        neuron_ids=TEST_MULTIPLE_NEURONS,
        neuropil_mesh_names=TEST_MULTIPLE_NEUROPILS,
        dataset="latest",
    )

    # Check DataFrame properties
    assert isinstance(result, pd.DataFrame)
    assert not result.empty, "Result should not be empty"
    assert "neuron_id" in result.columns, "Should have neuron_id column"

    # Check that all neuropil columns exist and have correct dtype
    for neuropil in TEST_MULTIPLE_NEUROPILS:
        assert neuropil in result.columns, f"Missing neuropil column: {neuropil}"
        assert pd.api.types.is_integer_dtype(
            result[neuropil]
        ), f"{neuropil} should have integer dtype"

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
            mesh=mock_mesh, mesh_coordinates="invalid", dataset="latest"
        )

    assert "mesh_coordinates must be either 'nm' or 'voxels'" in str(excinfo.value)


def test_get_synapses_in_mesh_invalid_loc():
    """Test error handling for invalid loc parameter."""
    mock_mesh = tm.creation.box(extents=[10000, 10000, 10000])

    with pytest.raises(ValueError) as excinfo:
        get_synapses_in_mesh(
            mesh=mock_mesh,
            loc="pre",
            dataset="latest",
        )

    assert "loc must be either 'ctr' or 'all'" in str(excinfo.value)


@patch("crantpy.queries.neuropils._query_synapses_in_bbox")
@patch("crantpy.utils.cave.load.get_cave_client")
def test_get_synapses_in_mesh_uses_raw_bbox_without_offset(
    mock_get_client, mock_query_bbox
):
    """Mesh bounds should be used directly for the CAVE bbox."""
    mock_get_client.return_value = MagicMock()
    mock_query_bbox.return_value = (pd.DataFrame(), 42)

    mock_mesh = MagicMock()
    mock_mesh.bounds = (
        np.array([100.0, 200.0, 300.0]),
        np.array([400.0, 500.0, 600.0]),
    )

    get_synapses_in_mesh(
        mesh=mock_mesh,
        return_pixels=False,
        dataset="latest",
    )

    bbox = mock_query_bbox.call_args.args[1]
    np.testing.assert_allclose(bbox[0], [100.0, 200.0, 300.0])
    np.testing.assert_allclose(bbox[1], [400.0, 500.0, 600.0])


@patch("crantpy.queries.neuropils._query_synapses_in_bbox")
@patch("crantpy.utils.cave.load.get_cave_client")
def test_get_synapses_in_mesh_always_clears_cached_cave_client(
    mock_get_client,
    mock_query_bbox,
):
    """Mesh queries should bypass the cached client to avoid stale materialization state."""
    mock_get_client.return_value = MagicMock()
    mock_query_bbox.return_value = (pd.DataFrame(), 42)

    mock_mesh = MagicMock()
    mock_mesh.bounds = (
        np.array([0.0, 0.0, 0.0]),
        np.array([100.0, 100.0, 100.0]),
    )

    get_synapses_in_mesh(
        mesh=mock_mesh,
        return_pixels=False,
        dataset="latest",
    )

    mock_get_client.assert_called_once_with(dataset="latest", clear_cache=True)


@patch("crantpy.utils.cave.segmentation.update_ids")
@patch("crantpy.queries.neuropils._batched_mesh_contains")
@patch("crantpy.queries.neuropils._query_synapses_in_bbox")
@patch("crantpy.utils.cave.load.get_cave_client")
def test_get_synapses_in_mesh_normalizes_root_ids_before_cleaning(
    mock_get_client,
    mock_query_bbox,
    mock_contains,
    mock_update_ids,
):
    """Stale roots that merge to the same latest root should be cleaned as autapses."""
    mock_get_client.return_value = MagicMock()
    mock_query_bbox.return_value = (
        pd.DataFrame(
            {
                "ctr_pt_position": [[10.0, 20.0, 30.0]],
                "pre_pt_position": [[40.0, 50.0, 60.0]],
                "post_pt_position": [[70.0, 80.0, 90.0]],
                "pre_pt_root_id": [111],
                "post_pt_root_id": [222],
            }
        ),
        42,
    )
    mock_contains.return_value = np.array([True], dtype=bool)
    mock_update_ids.return_value = pd.DataFrame(
        {
            "old_id": [111, 222],
            "new_id": [333, 333],
            "confidence": [1.0, 1.0],
            "changed": [True, True],
        }
    )

    mock_mesh = MagicMock()
    mock_mesh.bounds = (
        np.array([0.0, 0.0, 0.0]),
        np.array([100.0, 100.0, 100.0]),
    )

    result = get_synapses_in_mesh(
        mesh=mock_mesh,
        return_pixels=False,
        dataset="latest",
    )

    assert result.empty
    mock_update_ids.assert_called_once_with(
        [111, 222],
        dataset="latest",
        progress=False,
        clear_cache=True,
    )


@patch("crantpy.queries.neuropils._batched_mesh_contains")
@patch("crantpy.queries.neuropils._query_synapses_in_bbox")
@patch("crantpy.utils.cave.load.get_cave_client")
def test_get_synapses_in_mesh_uses_raw_synapse_coordinates_without_offset(
    mock_get_client,
    mock_query_bbox,
    mock_contains,
):
    """Point-in-mesh tests should receive raw synapse coordinates."""
    mock_get_client.return_value = MagicMock()
    mock_query_bbox.return_value = (
        pd.DataFrame(
            {
                "ctr_pt_position": [[10.0, 20.0, 30.0]],
                "pre_pt_position": [[40.0, 50.0, 60.0]],
                "post_pt_position": [[70.0, 80.0, 90.0]],
                "pre_pt_root_id": [111],
                "post_pt_root_id": [222],
            }
        ),
        42,
    )
    mock_contains.side_effect = lambda _mesh, points, batch_size=50000: np.ones(
        len(points), dtype=bool
    )

    mock_mesh = MagicMock()
    mock_mesh.bounds = (
        np.array([0.0, 0.0, 0.0]),
        np.array([100.0, 100.0, 100.0]),
    )

    get_synapses_in_mesh(
        mesh=mock_mesh,
        return_pixels=False,
        dataset="latest",
    )

    assert mock_contains.call_count == 1
    np.testing.assert_allclose(
        mock_contains.call_args_list[0].args[1], [[10.0, 20.0, 30.0]]
    )


@patch("crantpy.queries.neuropils._batched_mesh_contains")
@patch("crantpy.queries.neuropils._query_synapses_in_bbox")
@patch("crantpy.utils.cave.load.get_cave_client")
def test_get_synapses_in_mesh_retains_synapse_when_only_center_is_in_mesh(
    mock_get_client,
    mock_query_bbox,
    mock_contains,
):
    """Center-point containment alone should be sufficient to keep a synapse."""
    mock_get_client.return_value = MagicMock()
    mock_query_bbox.return_value = (
        pd.DataFrame(
            {
                "ctr_pt_position": [[10.0, 20.0, 30.0]],
                "pre_pt_position": [[400.0, 500.0, 600.0]],
                "post_pt_position": [[700.0, 800.0, 900.0]],
                "pre_pt_root_id": [111],
                "post_pt_root_id": [222],
            }
        ),
        42,
    )
    mock_contains.return_value = np.array([True], dtype=bool)

    mock_mesh = MagicMock()
    mock_mesh.bounds = (
        np.array([0.0, 0.0, 0.0]),
        np.array([100.0, 100.0, 100.0]),
    )

    result = get_synapses_in_mesh(
        mesh=mock_mesh,
        return_pixels=False,
        dataset="latest",
    )

    assert len(result) == 1
    assert result.iloc[0]["pre_pt_root_id"] == 111
    assert result.iloc[0]["post_pt_root_id"] == 222


@patch("crantpy.queries.neuropils._batched_mesh_contains")
@patch("crantpy.queries.neuropils._query_synapses_in_bbox")
@patch("crantpy.utils.cave.load.get_cave_client")
def test_get_synapses_in_mesh_loc_all_checks_ctr_pre_and_post(
    mock_get_client,
    mock_query_bbox,
    mock_contains,
):
    """loc='all' should require center, pre, and post positions to pass."""
    mock_get_client.return_value = MagicMock()
    mock_query_bbox.return_value = (
        pd.DataFrame(
            {
                "ctr_pt_position": [[10.0, 20.0, 30.0]],
                "pre_pt_position": [[40.0, 50.0, 60.0]],
                "post_pt_position": [[70.0, 80.0, 90.0]],
                "pre_pt_root_id": [111],
                "post_pt_root_id": [222],
            }
        ),
        42,
    )
    mock_contains.side_effect = [
        np.array([True], dtype=bool),
        np.array([True], dtype=bool),
        np.array([True], dtype=bool),
    ]

    mock_mesh = MagicMock()
    mock_mesh.bounds = (
        np.array([0.0, 0.0, 0.0]),
        np.array([100.0, 100.0, 100.0]),
    )

    result = get_synapses_in_mesh(
        mesh=mock_mesh,
        loc="all",
        return_pixels=False,
        dataset="latest",
    )

    assert len(result) == 1
    assert mock_contains.call_count == 3
    np.testing.assert_allclose(
        mock_contains.call_args_list[0].args[1], [[10.0, 20.0, 30.0]]
    )
    np.testing.assert_allclose(
        mock_contains.call_args_list[1].args[1], [[40.0, 50.0, 60.0]]
    )
    np.testing.assert_allclose(
        mock_contains.call_args_list[2].args[1], [[70.0, 80.0, 90.0]]
    )


@patch("crantpy.queries.neuropils._batched_mesh_contains")
@patch("crantpy.queries.neuropils._query_synapses_in_bbox")
@patch("crantpy.utils.cave.load.get_cave_client")
def test_get_synapses_in_mesh_loc_all_drops_synapse_if_any_position_is_outside(
    mock_get_client,
    mock_query_bbox,
    mock_contains,
):
    """loc='all' should drop a synapse if any one of the three location tests fails."""
    mock_get_client.return_value = MagicMock()
    mock_query_bbox.return_value = (
        pd.DataFrame(
            {
                "ctr_pt_position": [[10.0, 20.0, 30.0]],
                "pre_pt_position": [[40.0, 50.0, 60.0]],
                "post_pt_position": [[70.0, 80.0, 90.0]],
                "pre_pt_root_id": [111],
                "post_pt_root_id": [222],
            }
        ),
        42,
    )
    mock_contains.side_effect = [
        np.array([True], dtype=bool),
        np.array([False], dtype=bool),
    ]

    mock_mesh = MagicMock()
    mock_mesh.bounds = (
        np.array([0.0, 0.0, 0.0]),
        np.array([100.0, 100.0, 100.0]),
    )

    result = get_synapses_in_mesh(
        mesh=mock_mesh,
        loc="all",
        return_pixels=False,
        dataset="latest",
    )

    assert result.empty
    assert mock_contains.call_count == 2


@patch("crantpy.queries.neuropils._batched_mesh_contains")
@patch("crantpy.queries.neuropils._query_synapses_in_bbox")
@patch("crantpy.utils.cave.load.get_cave_client")
def test_get_synapses_in_mesh_batches_row_level_containment_without_changing_results(
    mock_get_client,
    mock_query_bbox,
    mock_contains,
    monkeypatch,
):
    """Large synapse tables should be checked in row batches while preserving output."""
    monkeypatch.setattr(neuropils_module, "_SYNAPSE_MESH_ROW_BATCH_SIZE", 2)

    mock_get_client.return_value = MagicMock()
    mock_query_bbox.return_value = (
        pd.DataFrame(
            {
                "ctr_pt_position": [
                    [10.0, 20.0, 30.0],
                    [40.0, 50.0, 60.0],
                    [70.0, 80.0, 90.0],
                ],
                "pre_pt_position": [
                    [11.0, 21.0, 31.0],
                    [41.0, 51.0, 61.0],
                    [71.0, 81.0, 91.0],
                ],
                "post_pt_position": [
                    [12.0, 22.0, 32.0],
                    [42.0, 52.0, 62.0],
                    [72.0, 82.0, 92.0],
                ],
                "pre_pt_root_id": [111, 222, 333],
                "post_pt_root_id": [444, 555, 666],
            }
        ),
        42,
    )
    mock_contains.side_effect = [
        np.array([True, False], dtype=bool),
        np.array([True], dtype=bool),
    ]

    mock_mesh = MagicMock()
    mock_mesh.bounds = (
        np.array([0.0, 0.0, 0.0]),
        np.array([100.0, 100.0, 100.0]),
    )

    result = get_synapses_in_mesh(
        mesh=mock_mesh,
        return_pixels=False,
        dataset="latest",
    )

    assert list(result["pre_pt_root_id"]) == [111, 333]
    assert mock_contains.call_count == 2
    np.testing.assert_allclose(
        mock_contains.call_args_list[0].args[1],
        [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
    )
    np.testing.assert_allclose(
        mock_contains.call_args_list[1].args[1],
        [[70.0, 80.0, 90.0]],
    )


# ============================================================================
# Tests for _query_synapses_in_bbox() subdivision logic
# ============================================================================


def _make_synapse_df(n, id_start=0):
    """Create a minimal synapse DataFrame with n rows and unique ids."""
    return pd.DataFrame(
        {
            "id": range(id_start, id_start + n),
            "ctr_pt_position": [[1000, 2000, 3000]] * n,
            "pre_pt_root_id": [111] * n,
            "post_pt_root_id": [222] * n,
        }
    )


def test_query_synapses_no_subdivision():
    """When result count < CAVE_ROW_LIMIT, no subdivision occurs."""
    mock_client = MagicMock()
    mock_client.materialize.most_recent_version.return_value = 42
    small_df = _make_synapse_df(100)
    mock_client.materialize.query_table.return_value = small_df

    result, mat_version = _query_synapses_in_bbox(
        mock_client,
        [[0, 0, 0], [10000, 10000, 10000]],
        "latest",
    )

    assert len(result) == 100
    assert mat_version == 42
    # Should have been called exactly once (no subdivision)
    assert mock_client.materialize.query_table.call_count == 1


def test_query_synapses_subdivision_triggered():
    """When result count >= CAVE_ROW_LIMIT, bbox is split and queried again."""
    mock_client = MagicMock()
    mock_client.materialize.most_recent_version.return_value = 42

    # First call returns exactly CAVE_ROW_LIMIT rows (triggers subdivision)
    # Sub-calls return small results (no further subdivision)
    big_df = _make_synapse_df(CAVE_ROW_LIMIT, id_start=0)
    small_a = _make_synapse_df(90_000, id_start=0)
    small_b = _make_synapse_df(80_000, id_start=90_000)

    mock_client.materialize.query_table.side_effect = [big_df, small_a, small_b]

    result, mat_version = _query_synapses_in_bbox(
        mock_client,
        [[0, 0, 0], [10000, 20000, 5000]],  # longest axis is Y
        "latest",
    )

    # 3 total queries: 1 original + 2 sub-queries
    assert mock_client.materialize.query_table.call_count == 3
    # Should have 90k + 80k = 170k unique rows
    assert len(result) == 170_000
    assert mat_version == 42


def test_query_synapses_deduplication():
    """Synapses on the boundary appear in both halves; should be deduplicated."""
    mock_client = MagicMock()
    mock_client.materialize.most_recent_version.return_value = 1

    big_df = _make_synapse_df(CAVE_ROW_LIMIT)
    # Both sub-queries return overlapping IDs (0..49)
    overlap_a = _make_synapse_df(50, id_start=0)
    overlap_b = _make_synapse_df(50, id_start=0)  # same IDs

    mock_client.materialize.query_table.side_effect = [big_df, overlap_a, overlap_b]

    result, _ = _query_synapses_in_bbox(
        mock_client,
        [[0, 0, 0], [100, 100, 100]],
        "latest",
    )

    assert len(result) == 50  # duplicates removed


def test_query_synapses_max_depth():
    """Recursion should stop at _MAX_SUBDIVISION_DEPTH even if still truncated."""
    mock_client = MagicMock()
    mock_client.materialize.most_recent_version.return_value = 1

    big_df = _make_synapse_df(CAVE_ROW_LIMIT)
    mock_client.materialize.query_table.return_value = big_df

    # At depth=8 (_MAX_SUBDIVISION_DEPTH), it should NOT subdivide further
    result, _ = _query_synapses_in_bbox(
        mock_client,
        [[0, 0, 0], [100, 100, 100]],
        "latest",
        materialization_version=1,
        depth=8,
    )

    # Only 1 call because at max depth it returns without subdividing
    assert mock_client.materialize.query_table.call_count == 1
    assert len(result) == CAVE_ROW_LIMIT


def test_query_synapses_live_mode():
    """Test that live mode uses live_query instead of query_table."""
    mock_client = MagicMock()
    small_df = _make_synapse_df(50)
    mock_client.materialize.live_query.return_value = small_df

    result, mat_version = _query_synapses_in_bbox(
        mock_client,
        [[0, 0, 0], [100, 100, 100]],
        "live",
    )

    assert len(result) == 50
    assert mat_version is None  # live mode doesn't resolve a version
    mock_client.materialize.live_query.assert_called_once()
    mock_client.materialize.query_table.assert_not_called()


# ============================================================================
# Tests for get_synapses_in_neuropils()
# ============================================================================


def _make_full_synapse_df(n, id_start=0):
    """Create a synapse DataFrame with all required columns."""
    return pd.DataFrame(
        {
            "id": range(id_start, id_start + n),
            "ctr_pt_position": [[15000, 25000, 35000]] * n,
            "pre_pt_position": [[15000, 25000, 35000]] * n,
            "post_pt_position": [[15000, 25000, 35000]] * n,
            "pre_pt_root_id": np.tile([111, 222], (n + 1) // 2)[:n],
            "post_pt_root_id": np.tile([333, 444], (n + 1) // 2)[:n],
            "size": [50] * n,
        }
    )


@patch("crantpy.queries.neuropils._query_synapses_in_bbox")
@patch("crantpy.queries.neuropils.load_neuropil_mesh")
@patch("crantpy.utils.cave.load.get_cave_client")
def test_get_synapses_in_neuropils_basic(
    mock_get_client, mock_load_mesh, mock_query_bbox
):
    """Test basic multi-neuropil query."""
    # Setup mock client
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    # Setup mock meshes - two neuropils with different contains results
    mesh_a = MagicMock()
    mesh_a.bounds = (np.array([0, 0, 0]), np.array([30000, 30000, 30000]))
    mesh_a.contains.return_value = np.array(
        [True, True, False, False, True, True, False, False, True, True]
    )

    mesh_b = MagicMock()
    mesh_b.bounds = (np.array([10000, 10000, 10000]), np.array([40000, 40000, 40000]))
    mesh_b.contains.return_value = np.array(
        [False, False, True, True, False, False, True, True, False, False]
    )

    mock_load_mesh.side_effect = [mesh_a, mesh_b]

    # Setup mock query result
    syn_df = _make_full_synapse_df(10)
    mock_query_bbox.return_value = (syn_df, 42)

    result = get_synapses_in_neuropils(
        neuropil_names=["fan_shaped_body", "ellipsoid_body"],
        dataset="latest",
    )

    assert isinstance(result, dict)
    assert set(result.keys()) == {
        "fan_shaped_body",
        "ellipsoid_body",
        "fan_shaped_body, ellipsoid_body",
    }
    assert len(result["fan_shaped_body"]) == 6  # 6 True in mesh_a.contains mask
    assert len(result["ellipsoid_body"]) == 4  # 4 True in mesh_b.contains mask
    assert len(result["fan_shaped_body, ellipsoid_body"]) == 10
    # Only one CAVE query should have been made
    mock_query_bbox.assert_called_once()
    mock_get_client.assert_called_once_with(dataset="latest", clear_cache=True)


@patch("crantpy.queries.neuropils._query_synapses_in_bbox")
@patch("crantpy.queries.neuropils.load_neuropil_mesh")
@patch("crantpy.utils.cave.load.get_cave_client")
def test_get_synapses_in_neuropils_empty_result(
    mock_get_client, mock_load_mesh, mock_query_bbox
):
    """Test when no synapses found in union bounding box."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mesh_a = MagicMock()
    mesh_a.bounds = (np.array([0, 0, 0]), np.array([100, 100, 100]))
    mock_load_mesh.return_value = mesh_a

    empty_df = pd.DataFrame()
    mock_query_bbox.return_value = (empty_df, 42)

    result = get_synapses_in_neuropils(
        neuropil_names=["fan_shaped_body", "ellipsoid_body"],
        dataset="latest",
    )

    assert isinstance(result, dict)
    assert "fan_shaped_body" in result
    assert "ellipsoid_body" in result
    assert "fan_shaped_body, ellipsoid_body" in result
    assert result["fan_shaped_body"].empty
    assert result["ellipsoid_body"].empty
    assert result["fan_shaped_body, ellipsoid_body"].empty


@patch("crantpy.queries.neuropils._query_synapses_in_bbox")
@patch("crantpy.queries.neuropils.load_neuropil_mesh")
@patch("crantpy.utils.cave.load.get_cave_client")
def test_get_synapses_in_neuropils_with_filtering(
    mock_get_client, mock_load_mesh, mock_query_bbox
):
    """Test that per-neuropil filtering is applied independently."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mesh = MagicMock()
    mesh.bounds = (np.array([0, 0, 0]), np.array([50000, 50000, 50000]))
    # All synapses are inside this mesh
    mesh.contains.return_value = np.array([True] * 20)
    mock_load_mesh.return_value = mesh

    # Create synapses: neuron 111 has 10 synapses with post 333, neuron 222 has 10 with post 444
    syn_df = _make_full_synapse_df(20)
    mock_query_bbox.return_value = (syn_df, 42)

    result = get_synapses_in_neuropils(
        neuropil_names=["fan_shaped_body"],
        min_synapses_per_neuron=1,
        min_synapses_per_pair=1,
        dataset="latest",
        return_pixels=False,
    )

    assert isinstance(result, dict)
    assert "fan_shaped_body" in result
    assert len(result["fan_shaped_body"]) == 20


@patch("crantpy.utils.cave.segmentation.update_ids")
@patch("crantpy.queries.neuropils._query_synapses_in_bbox")
@patch("crantpy.queries.neuropils.load_neuropil_mesh")
@patch("crantpy.utils.cave.load.get_cave_client")
def test_get_synapses_in_neuropils_normalizes_root_ids(
    mock_get_client,
    mock_load_mesh,
    mock_query_bbox,
    mock_update_ids,
):
    """Returned synapse IDs should be mapped to latest roots after the spatial query."""
    mock_get_client.return_value = MagicMock()

    mesh = MagicMock()
    mesh.bounds = (np.array([0, 0, 0]), np.array([50000, 50000, 50000]))
    mesh.contains.return_value = np.array([True])
    mock_load_mesh.return_value = mesh

    mock_query_bbox.return_value = (
        pd.DataFrame(
            {
                "ctr_pt_position": [[15000, 25000, 35000]],
                "pre_pt_position": [[15000, 25000, 35000]],
                "post_pt_position": [[15000, 25000, 35000]],
                "pre_pt_root_id": [111],
                "post_pt_root_id": [222],
                "size": [50],
            }
        ),
        42,
    )
    mock_update_ids.return_value = pd.DataFrame(
        {
            "old_id": [111, 222],
            "new_id": [1001, 2002],
            "confidence": [1.0, 1.0],
            "changed": [True, True],
        }
    )

    result = get_synapses_in_neuropils(
        neuropil_names=["fan_shaped_body"],
        return_pixels=False,
        dataset="latest",
    )

    assert result["fan_shaped_body"]["pre_pt_root_id"].tolist() == [1001]
    assert result["fan_shaped_body"]["post_pt_root_id"].tolist() == [2002]
    mock_update_ids.assert_called_once_with(
        [111, 222],
        dataset="latest",
        progress=False,
        clear_cache=True,
    )


def test_get_synapses_in_neuropils_invalid_name():
    """Test that invalid neuropil names raise ValueError."""
    with pytest.raises(ValueError, match="Invalid neuropil name"):
        get_synapses_in_neuropils(
            neuropil_names=["not_a_real_neuropil"],
            dataset="latest",
        )


@patch("crantpy.queries.neuropils._query_synapses_in_bbox")
@patch("crantpy.queries.neuropils.load_neuropil_mesh")
@patch("crantpy.utils.cave.load.get_cave_client")
def test_get_synapses_in_neuropils_accepts_nodulus_right_alias(
    mock_get_client,
    mock_load_mesh,
    mock_query_bbox,
):
    """The nodulus_right alias should pass validation in multi-neuropil queries."""
    mock_get_client.return_value = MagicMock()
    mock_query_bbox.return_value = (pd.DataFrame(), 42)

    mock_mesh = MagicMock()
    mock_mesh.bounds = (
        np.array([0.0, 0.0, 0.0]),
        np.array([100.0, 100.0, 100.0]),
    )
    mock_load_mesh.return_value = mock_mesh

    result = get_synapses_in_neuropils(
        neuropil_names=["nodulus_right"],
        dataset="latest",
    )

    assert "nodulus_right" in result
    mock_load_mesh.assert_called_once_with("nodulus_right")


@patch("crantpy.queries.neuropils._query_synapses_in_bbox")
@patch("crantpy.queries.neuropils.load_neuropil_mesh")
@patch("crantpy.utils.cave.load.get_cave_client")
def test_get_synapses_in_neuropils_uses_raw_union_bbox_without_offset(
    mock_get_client,
    mock_load_mesh,
    mock_query_bbox,
):
    """Union bbox should use raw mesh bounds without translation."""
    mock_get_client.return_value = MagicMock()
    mock_query_bbox.return_value = (pd.DataFrame(), 42)

    mesh_a = MagicMock()
    mesh_a.bounds = (
        np.array([10.0, 20.0, 30.0]),
        np.array([40.0, 50.0, 60.0]),
    )
    mesh_b = MagicMock()
    mesh_b.bounds = (
        np.array([0.0, 100.0, 5.0]),
        np.array([80.0, 120.0, 90.0]),
    )
    mock_load_mesh.side_effect = [mesh_a, mesh_b]

    get_synapses_in_neuropils(
        neuropil_names=["fan_shaped_body", "ellipsoid_body"],
        dataset="latest",
    )

    bbox = mock_query_bbox.call_args.args[1]
    np.testing.assert_allclose(bbox[0], [0.0, 20.0, 5.0])
    np.testing.assert_allclose(bbox[1], [80.0, 120.0, 90.0])


@patch("crantpy.queries.neuropils._batched_mesh_contains")
@patch("crantpy.queries.neuropils._query_synapses_in_bbox")
@patch("crantpy.queries.neuropils.load_neuropil_mesh")
@patch("crantpy.utils.cave.load.get_cave_client")
def test_get_synapses_in_neuropils_uses_raw_ctr_coords_without_offset(
    mock_get_client,
    mock_load_mesh,
    mock_query_bbox,
    mock_contains,
):
    """Center-point coordinates should be passed to mesh.contains without translation."""
    mock_get_client.return_value = MagicMock()
    mock_query_bbox.return_value = (
        pd.DataFrame(
            {
                "ctr_pt_position": [[10.0, 20.0, 30.0]],
                "pre_pt_position": [[10.0, 20.0, 30.0]],
                "post_pt_position": [[10.0, 20.0, 30.0]],
                "pre_pt_root_id": [111],
                "post_pt_root_id": [222],
            }
        ),
        42,
    )
    mock_contains.side_effect = lambda _mesh, points, batch_size=50000: np.ones(
        len(points), dtype=bool
    )

    mock_mesh = MagicMock()
    mock_mesh.bounds = (
        np.array([0.0, 0.0, 0.0]),
        np.array([100.0, 100.0, 100.0]),
    )
    mock_load_mesh.return_value = mock_mesh

    get_synapses_in_neuropils(
        neuropil_names=["fan_shaped_body"],
        return_pixels=False,
        dataset="latest",
    )

    np.testing.assert_allclose(mock_contains.call_args.args[1], [[10.0, 20.0, 30.0]])


def test_get_synapses_in_neuropils_invalid_materialization():
    """Test that invalid materialization raises ValueError."""
    with pytest.raises(ValueError, match="materialization"):
        get_synapses_in_neuropils(
            neuropil_names=["fan_shaped_body"],
            materialization="invalid",
            dataset="latest",
        )


# ============================================================================
# Tests for _query_with_breaker() logging
# ============================================================================


def test_query_with_breaker_logs_retry_cycle_and_success(caplog):
    """Retry logging should show the retry cycle and eventual success."""
    attempts = {"count": 0}

    def flaky_query():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise requests.RequestException("temporary outage")
        return "ok"

    caplog.set_level(logging.DEBUG, logger="crantpy.queries.neuropils")

    with patch("time.sleep", return_value=None):
        result = _query_with_breaker(flaky_query, operation="query_table")

    assert result == "ok"
    assert attempts["count"] == 3
    assert "failed on attempt 1; retrying with up to 15 attempts" in caplog.text
    assert "succeeded after 3 attempts (2 retries)" in caplog.text
    assert "CAVE query 'query_table' attempt 2/15" in caplog.text


def test_query_with_breaker_logs_final_failure(caplog):
    """Exhausted retries should log the final failure and preserve the exception."""

    def failing_query():
        raise requests.RequestException("still failing")

    caplog.set_level(logging.INFO, logger="crantpy.queries.neuropils")

    with patch("time.sleep", return_value=None):
        with pytest.raises(requests.RequestException, match="still failing"):
            _query_with_breaker(failing_query, operation="live_query")

    assert _cave_breaker._failure_count == 1
    assert "failed on attempt 1; retrying with up to 15 attempts" in caplog.text
    assert (
        "failed after 15 attempts with RequestException: still failing" in caplog.text
    )


def test_query_with_breaker_logs_wait_when_open(caplog):
    """An OPEN breaker should log the wait before attempting a probe."""
    _cave_breaker._state = _cave_breaker.OPEN
    _cave_breaker._last_failure_time = time.time()

    caplog.set_level(logging.INFO, logger="crantpy.queries.neuropils")

    with patch("time.sleep", return_value=None):
        result = _query_with_breaker(lambda: "ok", operation="most_recent_version")

    assert result == "ok"
    assert "Circuit breaker is OPEN - waiting" in caplog.text
    assert "most_recent_version" in caplog.text


@patch("crantpy.queries.neuropils._query_synapses_in_bbox")
@patch("crantpy.queries.neuropils.load_neuropil_mesh")
def test_get_synapses_in_neuropils_logs_cache_hit(
    mock_load_mesh,
    mock_query_bbox,
    tmp_path,
    caplog,
):
    """Cache hits should log the load path and cached row count."""
    cache_path = tmp_path / "synapse_cache.parquet"
    cache_path.write_text("placeholder")

    mock_mesh = MagicMock()
    mock_mesh.bounds = (np.array([0, 0, 0]), np.array([50000, 50000, 50000]))
    mock_mesh.contains.return_value = np.array([True])
    mock_load_mesh.return_value = mock_mesh

    syn_df = _make_full_synapse_df(1)
    caplog.set_level(logging.INFO, logger="crantpy.queries.neuropils")

    with patch("pandas.read_parquet", return_value=syn_df) as mock_read_parquet:
        result = get_synapses_in_neuropils(
            neuropil_names=["fan_shaped_body"],
            cache_path=str(cache_path),
            return_pixels=False,
            dataset="latest",
        )

    assert "fan_shaped_body" in result
    mock_read_parquet.assert_called_once_with(str(cache_path))
    mock_query_bbox.assert_not_called()
    assert f"Cache hit for raw union-bbox synapse data at {cache_path}" in caplog.text
    assert (
        "Loaded 1 raw synapses from cache; downstream filtering and mesh.contains() will still be applied"
        in caplog.text
    )


@patch("crantpy.queries.neuropils._query_synapses_in_bbox")
@patch("crantpy.queries.neuropils.load_neuropil_mesh")
@patch("crantpy.utils.cave.load.get_cave_client")
def test_get_synapses_in_neuropils_logs_cache_miss_and_write(
    mock_get_client,
    mock_load_mesh,
    mock_query_bbox,
    tmp_path,
    caplog,
):
    """Cache misses should log the miss, query retrieval, and cache write."""
    mock_get_client.return_value = MagicMock()

    mock_mesh = MagicMock()
    mock_mesh.bounds = (np.array([0, 0, 0]), np.array([50000, 50000, 50000]))
    mock_mesh.contains.return_value = np.array([True, True])
    mock_load_mesh.return_value = mock_mesh

    syn_df = _make_full_synapse_df(2)
    mock_query_bbox.return_value = (syn_df, 42)

    cache_path = tmp_path / "new_cache.parquet"
    caplog.set_level(logging.INFO, logger="crantpy.queries.neuropils")

    with patch.object(pd.DataFrame, "to_parquet", autospec=True) as mock_to_parquet:
        result = get_synapses_in_neuropils(
            neuropil_names=["fan_shaped_body"],
            cache_path=str(cache_path),
            return_pixels=False,
            dataset="latest",
        )

    assert "fan_shaped_body" in result
    mock_query_bbox.assert_called_once()
    mock_to_parquet.assert_called_once()
    assert f"Cache miss for raw union-bbox synapse data at {cache_path}" in caplog.text
    assert (
        "Retrieved raw union-bbox synapse data from CAVE; downstream filtering and mesh.contains() will still be applied"
        in caplog.text
    )
    assert f"Cached 2 raw synapses to {cache_path}" in caplog.text


# ============================================================================
# Tests for _CircuitBreaker
# ============================================================================


def test_circuit_breaker_stays_closed_on_success():
    """Breaker stays CLOSED when requests succeed."""
    assert _cave_breaker.state == _cave_breaker.CLOSED
    _cave_breaker.record_success()
    assert _cave_breaker.state == _cave_breaker.CLOSED
    assert _cave_breaker._failure_count == 0


def test_circuit_breaker_opens_after_threshold():
    """Breaker transitions CLOSED -> OPEN after failure_threshold failures."""
    for i in range(_cave_breaker.failure_threshold - 1):
        _cave_breaker.record_failure()
        assert (
            _cave_breaker.state == _cave_breaker.CLOSED
        ), f"Should still be CLOSED after {i+1} failures"

    # One more failure should trip it
    _cave_breaker.record_failure()
    assert _cave_breaker._state == _cave_breaker.OPEN


def test_circuit_breaker_resets_on_success():
    """A success resets the failure counter even before threshold."""
    _cave_breaker.record_failure()
    _cave_breaker.record_failure()
    assert _cave_breaker._failure_count == 2

    _cave_breaker.record_success()
    assert _cave_breaker._failure_count == 0
    assert _cave_breaker.state == _cave_breaker.CLOSED


def test_circuit_breaker_half_open_after_timeout():
    """Breaker transitions OPEN -> HALF_OPEN after recovery_timeout."""
    import time as _time

    # Trip the breaker
    for _ in range(_cave_breaker.failure_threshold):
        _cave_breaker.record_failure()
    assert _cave_breaker._state == _cave_breaker.OPEN

    # Simulate time passing beyond recovery_timeout
    _cave_breaker._last_failure_time = _time.time() - _cave_breaker.recovery_timeout - 1

    assert _cave_breaker.state == _cave_breaker.HALF_OPEN


def test_circuit_breaker_half_open_success_closes():
    """A successful probe in HALF_OPEN transitions back to CLOSED."""
    _cave_breaker._state = _cave_breaker.HALF_OPEN
    _cave_breaker.record_success()
    assert _cave_breaker.state == _cave_breaker.CLOSED
    assert _cave_breaker._failure_count == 0


def test_circuit_breaker_half_open_failure_reopens():
    """A failed probe in HALF_OPEN transitions back to OPEN."""
    _cave_breaker._state = _cave_breaker.HALF_OPEN
    _cave_breaker.record_failure()
    assert _cave_breaker._state == _cave_breaker.OPEN
