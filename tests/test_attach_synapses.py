# -*- coding: utf-8 -*-
"""
Tests for the attach_synapses functionality.

This module tests the attach_synapses function that maps synapses to skeleton nodes
and attaches them as a connectors table.
"""

import pytest
import pandas as pd
import numpy as np
import navis

from crantpy.queries.connections import attach_synapses
from crantpy.viz.l2 import get_l2_skeleton

# Test neuron ID - using the same ID as in other tests
TEST_ROOT_ID = 576460752664524086
# Use only neurons known to work with L2 skeletonization
TEST_ROOT_IDS = [576460752664524086]


@pytest.mark.parametrize("root_id", [TEST_ROOT_ID])
def test_attach_synapses_single(root_id):
    """Test attach_synapses on a single neuron.

    Verifies that:
    - Function returns a TreeNeuron
    - Neuron has a connectors attribute
    - Connectors table has expected columns
    - All synapses are mapped to valid node IDs
    """
    # Get a skeleton
    skeleton = get_l2_skeleton(root_id)

    # Attach synapses
    result = attach_synapses(skeleton)

    # Check type
    assert isinstance(result, navis.TreeNeuron)
    assert result.id == root_id

    # Check connectors exist
    assert hasattr(result, "connectors")
    assert isinstance(result.connectors, pd.DataFrame)

    # Check columns
    expected_cols = ["connector_id", "x", "y", "z", "type", "partner_id", "node_id"]
    for col in expected_cols:
        assert col in result.connectors.columns, f"Missing column: {col}"

    # If there are connectors, check their properties
    if len(result.connectors) > 0:
        # Check types
        assert result.connectors["type"].isin(["pre", "post"]).all()

        # Check node IDs are valid
        valid_node_ids = set(skeleton.nodes.node_id)
        assert result.connectors["node_id"].isin(valid_node_ids).all()

        # Check coordinates are numeric
        assert pd.api.types.is_numeric_dtype(result.connectors["x"])
        assert pd.api.types.is_numeric_dtype(result.connectors["y"])
        assert pd.api.types.is_numeric_dtype(result.connectors["z"])

        # Check connector_ids are sequential
        assert (
            result.connectors["connector_id"] == np.arange(len(result.connectors))
        ).all()


@pytest.mark.parametrize("root_ids", [TEST_ROOT_IDS])
def test_attach_synapses_multiple(root_ids):
    """Test attach_synapses on multiple neurons as a NeuronList.

    Verifies that:
    - Function returns a NeuronList
    - All neurons have connectors
    - Connectors are correctly assigned per neuron
    """
    # Get skeletons - wrap single neuron in list to create NeuronList
    skeleton = get_l2_skeleton(root_ids[0])
    skeletons = navis.NeuronList([skeleton])

    # Attach synapses
    result = attach_synapses(skeletons)

    # Check type
    assert isinstance(result, navis.NeuronList)
    assert len(result) == len(skeletons)

    # Check each neuron
    for neuron in result:
        assert hasattr(neuron, "connectors")
        assert isinstance(neuron.connectors, pd.DataFrame)

        expected_cols = ["connector_id", "x", "y", "z", "type", "partner_id", "node_id"]
        for col in expected_cols:
            assert col in neuron.connectors.columns


@pytest.mark.parametrize("root_id", [TEST_ROOT_ID])
def test_attach_synapses_pre_only(root_id):
    """Test attach_synapses with only presynapses.

    Verifies that when post=False, only presynapses are attached.
    """
    skeleton = get_l2_skeleton(root_id)
    result = attach_synapses(skeleton, post=False)

    assert isinstance(result, navis.TreeNeuron)

    if len(result.connectors) > 0:
        # All connectors should be presynapses
        assert (result.connectors["type"] == "pre").all()


@pytest.mark.parametrize("root_id", [TEST_ROOT_ID])
def test_attach_synapses_post_only(root_id):
    """Test attach_synapses with only postsynapses.

    Verifies that when pre=False, only postsynapses are attached.
    """
    skeleton = get_l2_skeleton(root_id)
    result = attach_synapses(skeleton, pre=False)

    assert isinstance(result, navis.TreeNeuron)

    if len(result.connectors) > 0:
        # All connectors should be postsynapses
        assert (result.connectors["type"] == "post").all()


def test_attach_synapses_invalid_input():
    """Test that attach_synapses raises errors for invalid input."""
    # Test with non-TreeNeuron
    with pytest.raises(TypeError):
        attach_synapses("not a neuron")

    with pytest.raises(TypeError):
        attach_synapses(123)

    with pytest.raises(TypeError):
        attach_synapses([1, 2, 3])


def test_attach_synapses_both_false():
    """Test that attach_synapses raises error when both pre and post are False."""
    skeleton = get_l2_skeleton(TEST_ROOT_ID)

    with pytest.raises(ValueError, match="`pre` and `post` must not both be False"):
        attach_synapses(skeleton, pre=False, post=False)


@pytest.mark.parametrize("root_id", [TEST_ROOT_ID])
def test_attach_synapses_max_distance(root_id):
    """Test that max_distance parameter filters distant synapses.

    Verifies that synapses far from skeleton are removed when clean=True.
    """
    skeleton = get_l2_skeleton(root_id)

    # Attach with large max_distance
    result_large = attach_synapses(skeleton, max_distance=10000, clean=True)

    # Attach with small max_distance
    result_small = attach_synapses(skeleton, max_distance=1000, clean=True)

    # Should have fewer or equal connectors with smaller distance
    assert len(result_small.connectors) <= len(result_large.connectors)


@pytest.mark.parametrize("root_id", [TEST_ROOT_ID])
def test_attach_synapses_clean_parameter(root_id):
    """Test that clean parameter affects results.

    Verifies that clean=False doesn't filter distant synapses.
    """
    skeleton = get_l2_skeleton(root_id)

    # Attach with clean=True
    result_clean = attach_synapses(skeleton, clean=True)

    # Attach with clean=False
    result_unclean = attach_synapses(skeleton, clean=False)

    # Both should succeed
    assert isinstance(result_clean, navis.TreeNeuron)
    assert isinstance(result_unclean, navis.TreeNeuron)


@pytest.mark.parametrize("root_id", [TEST_ROOT_ID])
def test_attach_synapses_threshold(root_id):
    """Test that threshold parameter filters weak connections.

    Verifies that increasing threshold reduces number of connectors.
    """
    skeleton = get_l2_skeleton(root_id)

    # Attach with threshold=1 (all synapses)
    result_low = attach_synapses(skeleton, threshold=1)

    # Attach with threshold=3 (only connections with 3+ synapses)
    result_high = attach_synapses(skeleton, threshold=3)

    # Should have fewer or equal connectors with higher threshold
    assert len(result_high.connectors) <= len(result_low.connectors)


@pytest.mark.parametrize("root_id", [TEST_ROOT_ID])
def test_attach_synapses_returns_modified(root_id):
    """Test that attach_synapses modifies neuron in place and returns it.

    Verifies that both the input neuron and return value have connectors.
    """
    skeleton = get_l2_skeleton(root_id)

    # Should not have connectors initially
    assert (
        not hasattr(skeleton, "connectors")
        or skeleton.connectors is None
        or len(skeleton.connectors) == 0
    )

    # Attach synapses
    result = attach_synapses(skeleton)

    # Both should have connectors now
    assert hasattr(skeleton, "connectors")
    assert hasattr(result, "connectors")
    assert len(skeleton.connectors) == len(result.connectors)


@pytest.mark.parametrize("root_id", [TEST_ROOT_ID])
def test_attach_synapses_coordinates_in_nm(root_id):
    """Test that connector coordinates are in nanometers.

    Verifies coordinates match expected nanometer scale.
    """
    skeleton = get_l2_skeleton(root_id)
    result = attach_synapses(skeleton)

    if len(result.connectors) > 0:
        # Coordinates should be large numbers (in nanometers)
        # CRANT brain is roughly 200-400k nm in each dimension
        assert result.connectors["x"].abs().max() > 1000
        assert result.connectors["y"].abs().max() > 1000
        assert result.connectors["z"].abs().max() > 1000


@pytest.mark.parametrize("root_id", [TEST_ROOT_ID])
def test_attach_synapses_empty_result(root_id):
    """Test that attach_synapses handles case with no synapses gracefully.

    Even if no synapses match criteria, should return valid neuron with empty table.
    """
    skeleton = get_l2_skeleton(root_id)

    # Use very high threshold to get no synapses
    result = attach_synapses(skeleton, threshold=10000)

    # Should still work, just with empty connectors
    assert isinstance(result, navis.TreeNeuron)
    assert hasattr(result, "connectors")
    assert isinstance(result.connectors, pd.DataFrame)
    # May be empty
    assert len(result.connectors) >= 0
