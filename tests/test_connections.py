# -*- coding: utf-8 -*-
"""Tests for the connections module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from crantpy.queries.connections import (
    get_synapses,
    get_adjacency,
    get_connectivity,
    get_synapse_counts,
)
from typing import List, Union


# Mock data for testing
def create_mock_synapses_data():
    """Create mock synapse data for testing."""
    return pd.DataFrame(
        {
            "pre_pt_root_id": [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3],
            "post_pt_root_id": [2, 2, 2, 3, 3, 3, 1, 1, 1, 1, 2, 2],
            "pre_pt_supervoxel_id": [10, 11, 12, 13, 20, 21, 22, 30, 31, 32, 33, 34],
            "post_pt_supervoxel_id": [25, 26, 27, 35, 36, 37, 15, 16, 17, 18, 28, 29],
            "cleft_score": [
                0.8,
                0.9,
                0.7,
                0.85,
                0.6,
                0.75,
                0.8,
                0.7,
                0.9,
                0.8,
                0.6,
                0.7,
            ],
            "size": [10, 5, 15, 8, 20, 12, 9, 7, 11, 13, 14, 16],
            "x": [100, 110, 120, 130, 200, 210, 220, 300, 310, 320, 330, 340],
            "y": [
                1000,
                1100,
                1200,
                1300,
                2000,
                2100,
                2200,
                3000,
                3100,
                3200,
                3300,
                3400,
            ],
            "z": [
                5000,
                5100,
                5200,
                5300,
                6000,
                6100,
                6200,
                7000,
                7100,
                7200,
                7300,
                7400,
            ],
        }
    )


def create_mock_cave_client():
    """Create a mock CAVE client for testing."""
    mock_client = Mock()
    mock_materialize = Mock()
    mock_client.materialize = mock_materialize

    # Mock the most_recent_version method
    mock_materialize.most_recent_version.return_value = 123

    # Mock query_table and live_query methods
    mock_materialize.query_table.return_value = create_mock_synapses_data()
    mock_materialize.live_query.return_value = create_mock_synapses_data()

    return mock_client


class TestGetSynapses:
    """Test cases for get_synapses function."""

    @patch("crantpy.queries.connections.get_cave_client")
    def test_get_synapses_with_pre_ids_only(self, mock_get_client):
        """Test get_synapses with only pre_ids specified."""
        mock_get_client.return_value = create_mock_cave_client()

        result = get_synapses(pre_ids=[1, 2])

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        mock_get_client.assert_called_once()

    @patch("crantpy.queries.connections.get_cave_client")
    def test_get_synapses_with_post_ids_only(self, mock_get_client):
        """Test get_synapses with only post_ids specified."""
        mock_get_client.return_value = create_mock_cave_client()

        result = get_synapses(post_ids=[2, 3])

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        mock_get_client.assert_called_once()

    @patch("crantpy.queries.connections.get_cave_client")
    def test_get_synapses_with_both_ids(self, mock_get_client):
        """Test get_synapses with both pre_ids and post_ids specified."""
        mock_get_client.return_value = create_mock_cave_client()

        result = get_synapses(pre_ids=[1], post_ids=[2, 3])

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        mock_get_client.assert_called_once()

    @patch("crantpy.queries.connections.get_cave_client")
    def test_get_synapses_single_int_input(self, mock_get_client):
        """Test get_synapses with single integer inputs."""
        mock_get_client.return_value = create_mock_cave_client()

        result = get_synapses(pre_ids=1, post_ids=2)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        mock_get_client.assert_called_once()

    @patch("crantpy.queries.connections.get_cave_client")
    def test_get_synapses_string_input(self, mock_get_client):
        """Test get_synapses with string inputs."""
        mock_get_client.return_value = create_mock_cave_client()

        result = get_synapses(pre_ids="1", post_ids=["2", "3"])

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        mock_get_client.assert_called_once()

    def test_get_synapses_no_ids_raises_error(self):
        """Test that get_synapses raises ValueError when no IDs are provided."""
        with pytest.raises(
            ValueError, match="You must provide at least one of pre_ids or post_ids"
        ):
            get_synapses()

    @patch("crantpy.queries.connections.get_cave_client")
    def test_get_synapses_with_threshold(self, mock_get_client):
        """Test get_synapses with threshold parameter."""
        mock_client = create_mock_cave_client()
        mock_get_client.return_value = mock_client

        result = get_synapses(pre_ids=[1], threshold=2)

        assert isinstance(result, pd.DataFrame)
        mock_get_client.assert_called_once()

    @patch("crantpy.queries.connections.get_cave_client")
    def test_get_synapses_with_min_size(self, mock_get_client):
        """Test get_synapses with min_size parameter."""
        mock_client = create_mock_cave_client()
        mock_get_client.return_value = mock_client

        # Create mock data with size column
        mock_data = create_mock_synapses_data()
        mock_client.materialize.query_table.return_value = mock_data

        result = get_synapses(pre_ids=[1], min_size=10)

        assert isinstance(result, pd.DataFrame)
        # Should filter out rows with size < 10
        if not result.empty and "size" in result.columns:
            assert all(result["size"] >= 10)

    @patch("crantpy.queries.connections.get_cave_client")
    def test_get_synapses_live_query(self, mock_get_client):
        """Test get_synapses with materialization='live' parameter."""
        mock_client = create_mock_cave_client()
        mock_get_client.return_value = mock_client

        result = get_synapses(pre_ids=[1], materialization="live")

        assert isinstance(result, pd.DataFrame)
        # Should call live_query instead of query_table
        mock_client.materialize.live_query.assert_called_once()
        mock_client.materialize.query_table.assert_not_called()

    @patch("crantpy.queries.connections.get_cave_client")
    def test_get_synapses_default_materialization(self, mock_get_client):
        """Test get_synapses with default materialization='latest'."""
        mock_client = create_mock_cave_client()
        mock_get_client.return_value = mock_client

        result = get_synapses(pre_ids=[1], materialization="latest")

        assert isinstance(result, pd.DataFrame)
        # Should use query_table with most_recent_version
        mock_client.materialize.query_table.assert_called_once()
        mock_client.materialize.most_recent_version.assert_called_once()

    @patch("crantpy.queries.connections.get_cave_client")
    def test_get_synapses_invalid_materialization(self, mock_get_client):
        """Test get_synapses with invalid materialization value."""
        mock_client = create_mock_cave_client()
        mock_get_client.return_value = mock_client

        with pytest.raises(
            ValueError, match="materialization must be either 'live' or 'latest'"
        ):
            get_synapses(pre_ids=[1], materialization="invalid")

    @patch("crantpy.queries.connections.get_cave_client")
    def test_get_synapses_empty_result(self, mock_get_client):
        """Test get_synapses when query returns empty DataFrame."""
        mock_client = create_mock_cave_client()
        mock_client.materialize.query_table.return_value = pd.DataFrame()
        mock_get_client.return_value = mock_client

        result = get_synapses(pre_ids=[999])  # Non-existent ID

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @patch("crantpy.queries.connections.get_cave_client")
    def test_get_synapses_with_dataset(self, mock_get_client):
        """Test get_synapses with dataset parameter."""
        mock_client = create_mock_cave_client()
        mock_get_client.return_value = mock_client

        result = get_synapses(pre_ids=[1], dataset="latest")

        assert isinstance(result, pd.DataFrame)
        # Should pass dataset to get_cave_client
        mock_get_client.assert_called_with(dataset="latest")


class TestGetAdjacency:
    """Test cases for get_adjacency function."""

    @patch("crantpy.queries.connections.get_synapses")
    def test_get_adjacency_basic(self, mock_get_synapses):
        """Test basic functionality of get_adjacency."""
        # Mock get_synapses to return sample synapse data
        mock_synapses = pd.DataFrame(
            {
                "pre_pt_root_id": [1, 1, 2, 2, 3],
                "post_pt_root_id": [2, 3, 3, 1, 1],
            }
        )
        mock_get_synapses.return_value = mock_synapses

        result = get_adjacency(pre_ids=[1, 2, 3], post_ids=[1, 2, 3])

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 3)  # 3 unique pre_ids, 3 unique post_ids
        assert result.loc[1, 2] == 1  # 1 synapse from neuron 1 to neuron 2
        assert result.loc[1, 3] == 1  # 1 synapse from neuron 1 to neuron 3
        assert result.loc[2, 3] == 1  # 1 synapse from neuron 2 to neuron 3
        assert result.loc[2, 1] == 1  # 1 synapse from neuron 2 to neuron 1
        assert result.loc[3, 1] == 1  # 1 synapse from neuron 3 to neuron 1

        # Verify get_synapses was called with correct parameters
        mock_get_synapses.assert_called_once()

    @patch("crantpy.queries.connections.get_synapses")
    def test_get_adjacency_multiple_synapses(self, mock_get_synapses):
        """Test get_adjacency with multiple synapses between same pair."""
        # Mock get_synapses to return data with multiple synapses between pairs
        mock_synapses = pd.DataFrame(
            {
                "pre_pt_root_id": [1, 1, 1, 2, 2],
                "post_pt_root_id": [2, 2, 3, 3, 3],
            }
        )
        mock_get_synapses.return_value = mock_synapses

        result = get_adjacency(pre_ids=[1, 2], post_ids=[2, 3])

        assert result.loc[1, 2] == 2  # 2 synapses from neuron 1 to neuron 2
        assert result.loc[1, 3] == 1  # 1 synapse from neuron 1 to neuron 3
        assert result.loc[2, 3] == 2  # 2 synapses from neuron 2 to neuron 3

    @patch("crantpy.queries.connections.get_synapses")
    def test_get_adjacency_symmetric(self, mock_get_synapses):
        """Test get_adjacency with symmetric=True."""
        mock_synapses = pd.DataFrame(
            {
                "pre_pt_root_id": [1, 2, 3, 4],
                "post_pt_root_id": [2, 3, 1, 5],
            }
        )
        mock_get_synapses.return_value = mock_synapses

        result = get_adjacency(
            pre_ids=[1, 2, 3, 4], post_ids=[1, 2, 3, 5], symmetric=True
        )

        # Should use union of pre_ids and post_ids: [1,2,3,4] ∪ [1,2,3,5] = [1,2,3,4,5]
        # But filtered to neurons that appear in synapses: [1,2,3,4,5] ∩ [1,2,3,4,5] = [1,2,3,4,5]
        expected_neurons = sorted([1, 2, 3, 4, 5])
        assert sorted(result.index) == expected_neurons
        assert sorted(result.columns) == expected_neurons
        assert result.shape == (5, 5)

    @patch("crantpy.queries.connections.get_synapses")
    def test_get_adjacency_asymmetric(self, mock_get_synapses):
        """Test get_adjacency with symmetric=False (default)."""
        mock_synapses = pd.DataFrame(
            {
                "pre_pt_root_id": [1, 2, 3],
                "post_pt_root_id": [2, 3, 4],
            }
        )
        mock_get_synapses.return_value = mock_synapses

        result = get_adjacency(pre_ids=[1, 2, 3], post_ids=[2, 3, 4], symmetric=False)

        # Should have all unique pre neurons as rows, all unique post neurons as columns
        assert sorted(result.index) == [1, 2, 3]
        assert sorted(result.columns) == [2, 3, 4]
        assert result.shape == (3, 3)

    @patch("crantpy.queries.connections.get_synapses")
    def test_get_adjacency_empty_synapses(self, mock_get_synapses):
        """Test get_adjacency when get_synapses returns empty DataFrame."""
        mock_get_synapses.return_value = pd.DataFrame()

        result = get_adjacency(pre_ids=[1, 2], post_ids=[3, 4])

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (
            2,
            2,
        )  # Should return empty matrix with specified dimensions
        assert (result == 0).all().all()  # All entries should be 0

    @patch("crantpy.queries.connections.get_synapses")
    def test_get_adjacency_with_threshold(self, mock_get_synapses):
        """Test get_adjacency with threshold parameter."""
        mock_synapses = pd.DataFrame(
            {
                "pre_pt_root_id": [1, 1, 2],
                "post_pt_root_id": [2, 2, 3],
            }
        )
        mock_get_synapses.return_value = mock_synapses

        result = get_adjacency(pre_ids=[1, 2], post_ids=[2, 3], threshold=2)

        # Verify get_synapses was called with threshold parameter
        mock_get_synapses.assert_called_once()
        call_args = mock_get_synapses.call_args[1]
        assert call_args["threshold"] == 2

    @patch("crantpy.queries.connections.get_synapses")
    def test_get_adjacency_with_materialization(self, mock_get_synapses):
        """Test get_adjacency with materialization parameter."""
        mock_synapses = pd.DataFrame(
            {
                "pre_pt_root_id": [1],
                "post_pt_root_id": [2],
            }
        )
        mock_get_synapses.return_value = mock_synapses

        result = get_adjacency(pre_ids=[1], post_ids=[2], materialization="live")

        # Verify get_synapses was called with materialization parameter
        mock_get_synapses.assert_called_once()
        call_args = mock_get_synapses.call_args[1]
        assert call_args["materialization"] == "live"

    @patch("crantpy.queries.connections.get_synapses")
    def test_get_adjacency_with_min_size(self, mock_get_synapses):
        """Test get_adjacency with min_size parameter."""
        mock_synapses = pd.DataFrame(
            {
                "pre_pt_root_id": [1],
                "post_pt_root_id": [2],
            }
        )
        mock_get_synapses.return_value = mock_synapses

        result = get_adjacency(pre_ids=[1], post_ids=[2], min_size=10)

        # Verify get_synapses was called with min_size parameter
        mock_get_synapses.assert_called_once()
        call_args = mock_get_synapses.call_args[1]
        assert call_args["min_size"] == 10

    @patch("crantpy.queries.connections.get_synapses")
    def test_get_adjacency_empty_input(self, mock_get_synapses):
        """Test get_adjacency with empty inputs."""
        mock_get_synapses.return_value = pd.DataFrame()

        result = get_adjacency(pre_ids=[], post_ids=[])

        assert isinstance(result, pd.DataFrame)
        # Should return empty matrix
        assert result.shape == (0, 0)

    @patch("crantpy.queries.connections.get_synapses")
    def test_get_adjacency_single_connection(self, mock_get_synapses):
        """Test get_adjacency with single connection."""
        mock_synapses = pd.DataFrame(
            {
                "pre_pt_root_id": [1],
                "post_pt_root_id": [2],
            }
        )
        mock_get_synapses.return_value = mock_synapses

        result = get_adjacency(pre_ids=[1], post_ids=[2])

        assert result.shape == (1, 1)
        assert result.loc[1, 2] == 1

    @patch("crantpy.queries.connections.get_synapses")
    def test_get_adjacency_mixed_types(self, mock_get_synapses):
        """Test get_adjacency with mixed int/string types."""
        mock_synapses = pd.DataFrame(
            {
                "pre_pt_root_id": [1, 2, 3],
                "post_pt_root_id": [2, 3, 1],
            }
        )
        mock_get_synapses.return_value = mock_synapses

        result = get_adjacency(pre_ids=[1, "2", 3], post_ids=["2", 3, 1])

        assert isinstance(result, pd.DataFrame)
        # Should handle mixed types correctly
        assert result.shape == (3, 3)

    @patch("crantpy.queries.connections.get_synapses")
    def test_get_adjacency_zero_entries(self, mock_get_synapses):
        """Test that get_adjacency correctly sets zero entries."""
        # Mock no connections between specified pre and post
        mock_get_synapses.return_value = pd.DataFrame()

        result = get_adjacency(pre_ids=[1, 2], post_ids=[3, 4])

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)
        # All entries should be 0 since no connections
        assert (result == 0).all().all()


class TestGetConnectivity:
    """Test cases for get_connectivity function."""

    @patch("crantpy.queries.connections.get_synapses")
    def test_get_connectivity_basic(self, mock_get_synapses):
        """Test basic functionality of get_connectivity."""
        # Mock synapses data
        mock_synapses = pd.DataFrame(
            {
                "pre_pt_root_id": [1, 1, 2, 3],
                "post_pt_root_id": [2, 3, 1, 1],
                "id": [101, 102, 103, 104],
            }
        )
        mock_get_synapses.return_value = mock_synapses

        result = get_connectivity(neuron_ids=[1])

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["pre", "post", "weight"]
        # Should be sorted by weight descending
        assert result["weight"].is_monotonic_decreasing

        # Verify get_synapses was called twice (upstream and downstream)
        assert mock_get_synapses.call_count == 2

    @patch("crantpy.queries.connections.get_synapses")
    def test_get_connectivity_upstream_only(self, mock_get_synapses):
        """Test get_connectivity with upstream=True, downstream=False."""
        mock_synapses = pd.DataFrame(
            {"pre_pt_root_id": [2, 3], "post_pt_root_id": [1, 1], "id": [101, 102]}
        )
        mock_get_synapses.return_value = mock_synapses

        result = get_connectivity(neuron_ids=[1], upstream=True, downstream=False)

        assert isinstance(result, pd.DataFrame)
        # Should only call get_synapses once (for upstream)
        mock_get_synapses.assert_called_once()
        call_args = mock_get_synapses.call_args[1]
        assert call_args["pre_ids"] is None
        assert call_args["post_ids"] == ["1"]

    @patch("crantpy.queries.connections.get_synapses")
    def test_get_connectivity_downstream_only(self, mock_get_synapses):
        """Test get_connectivity with upstream=False, downstream=True."""
        mock_synapses = pd.DataFrame(
            {"pre_pt_root_id": [1, 1], "post_pt_root_id": [2, 3], "id": [101, 102]}
        )
        mock_get_synapses.return_value = mock_synapses

        result = get_connectivity(neuron_ids=[1], upstream=False, downstream=True)

        assert isinstance(result, pd.DataFrame)
        # Should only call get_synapses once (for downstream)
        mock_get_synapses.assert_called_once()
        call_args = mock_get_synapses.call_args[1]
        assert call_args["pre_ids"] == ["1"]
        assert call_args["post_ids"] is None

    def test_get_connectivity_invalid_parameters(self):
        """Test get_connectivity with invalid parameters."""
        with pytest.raises(
            ValueError, match="Both `upstream` and `downstream` cannot be False"
        ):
            get_connectivity(neuron_ids=[1], upstream=False, downstream=False)

    @patch("crantpy.queries.connections.get_synapses")
    def test_get_connectivity_empty_result(self, mock_get_synapses):
        """Test get_connectivity when no synapses are found."""
        mock_get_synapses.return_value = pd.DataFrame()

        result = get_connectivity(neuron_ids=[1])

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["pre", "post", "weight"]
        assert len(result) == 0

    @patch("crantpy.queries.connections.get_synapses")
    def test_get_connectivity_with_threshold(self, mock_get_synapses):
        """Test get_connectivity with threshold parameter."""
        # Create data where some pairs have multiple synapses
        mock_synapses = pd.DataFrame(
            {
                "pre_pt_root_id": [1, 1, 1, 2, 2],
                "post_pt_root_id": [2, 2, 3, 3, 3],
                "id": [101, 102, 103, 104, 105],
            }
        )
        mock_get_synapses.return_value = mock_synapses

        result = get_connectivity(neuron_ids=[1], threshold=2)

        assert isinstance(result, pd.DataFrame)
        # Only connection 1->2 (2 synapses) and 2->3 (2 synapses) should remain
        # Connection 1->3 (1 synapse) should be filtered out
        assert len(result) == 2
        assert all(result["weight"] >= 2)

        # Verify get_synapses was called with threshold=1 (applied later)
        for call in mock_get_synapses.call_args_list:
            assert call[1]["threshold"] == 1

    @patch("crantpy.queries.connections.get_synapses")
    def test_get_connectivity_with_clean(self, mock_get_synapses):
        """Test get_connectivity with clean=True to remove autapses and background."""
        # Mock data should represent what get_synapses returns after cleaning
        # (autapses and background connections already removed)
        mock_synapses = pd.DataFrame(
            {
                "pre_pt_root_id": [1],  # Only clean connection remains
                "post_pt_root_id": [2],
                "id": [101],
            }
        )
        mock_get_synapses.return_value = mock_synapses

        result = get_connectivity(neuron_ids=[1], clean=True)

        assert isinstance(result, pd.DataFrame)
        # Should only have 1->2 connection (autapse and background removed)
        assert len(result) == 1
        assert result.iloc[0]["pre"] == 1
        assert result.iloc[0]["post"] == 2
        assert result.iloc[0]["weight"] == 1

    @patch("crantpy.queries.connections.get_synapses")
    def test_get_connectivity_no_clean(self, mock_get_synapses):
        """Test get_connectivity with clean=False to keep autapses."""
        mock_synapses = pd.DataFrame(
            {
                "pre_pt_root_id": [1, 1],
                "post_pt_root_id": [2, 1],  # includes autapse (1->1)
                "id": [101, 102],
            }
        )
        mock_get_synapses.return_value = mock_synapses

        result = get_connectivity(neuron_ids=[1], clean=False)

        assert isinstance(result, pd.DataFrame)
        # Should include autapse
        assert len(result) == 2
        assert any((result["pre"] == 1) & (result["post"] == 1))  # autapse present
        assert any((result["pre"] == 1) & (result["post"] == 2))  # regular connection

    @patch("crantpy.queries.connections.get_synapses")
    def test_get_connectivity_multiple_neurons(self, mock_get_synapses):
        """Test get_connectivity with multiple query neurons."""
        mock_synapses = pd.DataFrame(
            {
                "pre_pt_root_id": [1, 2, 3, 4],
                "post_pt_root_id": [2, 1, 1, 2],
                "id": [101, 102, 103, 104],
            }
        )
        mock_get_synapses.return_value = mock_synapses

        result = get_connectivity(neuron_ids=[1, 2])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4  # All 4 connections

        # Verify get_synapses was called with multiple IDs
        for call in mock_get_synapses.call_args_list:
            call_args = call[1]
            # One call should have post_ids=['1', '2'], the other pre_ids=['1', '2']
            if call_args["pre_ids"] is None:
                assert set(call_args["post_ids"]) == {"1", "2"}
            else:
                assert set(call_args["pre_ids"]) == {"1", "2"}

    @patch("crantpy.queries.connections.get_synapses")
    def test_get_connectivity_duplicate_removal(self, mock_get_synapses):
        """Test that get_connectivity removes duplicate synapses correctly."""
        # Create data that would have duplicates if upstream and downstream overlap
        upstream_syns = pd.DataFrame(
            {"pre_pt_root_id": [2, 3], "post_pt_root_id": [1, 1], "id": [101, 102]}
        )
        downstream_syns = pd.DataFrame(
            {
                "pre_pt_root_id": [
                    1,
                    2,
                ],  # Note: 2->1 would be duplicate if it appeared in upstream
                "post_pt_root_id": [2, 1],
                "id": [103, 101],  # 101 is duplicate
            }
        )

        mock_get_synapses.side_effect = [upstream_syns, downstream_syns]

        result = get_connectivity(neuron_ids=[1])

        assert isinstance(result, pd.DataFrame)
        # Should have 3 unique connections, not 4
        assert len(result) == 3

        # Check that connections are correctly aggregated
        expected_connections = {(1, 2), (2, 1), (3, 1)}
        actual_connections = set(zip(result["pre"], result["post"]))
        assert actual_connections == expected_connections

    @patch("crantpy.queries.connections.get_synapses")
    def test_get_connectivity_with_materialization(self, mock_get_synapses):
        """Test get_connectivity passes materialization parameter correctly."""
        mock_get_synapses.return_value = pd.DataFrame()

        get_connectivity(neuron_ids=[1], materialization="live")

        # Verify all calls used materialization='live'
        for call in mock_get_synapses.call_args_list:
            assert call[1]["materialization"] == "live"

    @patch("crantpy.queries.connections.get_synapses")
    def test_get_connectivity_with_min_size(self, mock_get_synapses):
        """Test get_connectivity passes min_size parameter correctly."""
        mock_get_synapses.return_value = pd.DataFrame()

        get_connectivity(neuron_ids=[1], min_size=10)

        # Verify all calls used min_size=10
        for call in mock_get_synapses.call_args_list:
            assert call[1]["min_size"] == 10


class TestGetSynapseCounts:
    """Test cases for get_synapse_counts function."""

    @patch("crantpy.queries.connections.get_connectivity")
    def test_get_synapse_counts_single_neuron(self, mock_get_connectivity):
        """Test get_synapse_counts with a single neuron ID."""
        # Mock connectivity data showing neuron 1 has 2 pre and 3 post connections
        mock_connectivity_data = pd.DataFrame(
            {"pre": [1, 1, 2, 3, 4], "post": [2, 3, 1, 1, 1], "weight": [5, 3, 4, 2, 1]}
        )
        mock_get_connectivity.return_value = mock_connectivity_data

        result = get_synapse_counts(1)

        # Check that get_connectivity was called with correct parameters
        mock_get_connectivity.assert_called_once_with(
            neuron_ids=1,
            upstream=True,
            downstream=True,
            threshold=1,
            min_size=None,
            materialization="latest",
            clean=True,
            update_ids=True,
            dataset="latest",
        )

        # Check result structure
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["pre", "post"]
        assert result.index.name == "neuron_id"
        assert 1 in result.index

        # Check counts: neuron 1 appears as pre 2 times, as post 3 times
        assert result.loc[1, "pre"] == 2  # 1->2, 1->3
        assert result.loc[1, "post"] == 3  # 2->1, 3->1, 4->1

    @patch("crantpy.queries.connections.get_connectivity")
    def test_get_synapse_counts_multiple_neurons(self, mock_get_connectivity):
        """Test get_synapse_counts with multiple neuron IDs."""
        mock_connectivity_data = pd.DataFrame(
            {"pre": [1, 1, 2, 3, 4], "post": [2, 3, 1, 1, 5], "weight": [5, 3, 4, 2, 1]}
        )
        mock_get_connectivity.return_value = mock_connectivity_data

        result = get_synapse_counts([1, 2, 5])

        # Check result structure
        assert len(result) == 3
        assert all(nid in result.index for nid in [1, 2, 5])

        # Check counts
        assert result.loc[1, "pre"] == 2  # 1->2, 1->3
        assert result.loc[1, "post"] == 2  # 2->1, 3->1
        assert result.loc[2, "pre"] == 1  # 2->1
        assert result.loc[2, "post"] == 1  # 1->2
        assert result.loc[5, "pre"] == 0  # No outgoing
        assert result.loc[5, "post"] == 1  # 4->5

    @patch("crantpy.queries.connections.get_connectivity")
    def test_get_synapse_counts_empty_connectivity(self, mock_get_connectivity):
        """Test get_synapse_counts when no connectivity data is found."""
        mock_get_connectivity.return_value = pd.DataFrame(
            columns=["pre", "post", "weight"]
        )

        result = get_synapse_counts([1, 2])

        # Check result structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert all(result.values.flatten() == 0)  # All counts should be zero
        assert result.loc[1, "pre"] == 0
        assert result.loc[1, "post"] == 0
        assert result.loc[2, "pre"] == 0
        assert result.loc[2, "post"] == 0

    @patch("crantpy.queries.connections.get_connectivity")
    def test_get_synapse_counts_with_threshold(self, mock_get_connectivity):
        """Test get_synapse_counts with threshold parameter."""
        mock_connectivity_data = pd.DataFrame(
            {"pre": [1, 1, 2], "post": [2, 3, 1], "weight": [5, 3, 4]}
        )
        mock_get_connectivity.return_value = mock_connectivity_data

        result = get_synapse_counts(1, threshold=4)

        # Check that threshold was passed to get_connectivity
        mock_get_connectivity.assert_called_once_with(
            neuron_ids=1,
            upstream=True,
            downstream=True,
            threshold=4,
            min_size=None,
            materialization="latest",
            clean=True,
            update_ids=True,
            dataset="latest",
        )

    @patch("crantpy.queries.connections.get_connectivity")
    def test_get_synapse_counts_with_min_size(self, mock_get_connectivity):
        """Test get_synapse_counts with min_size parameter."""
        mock_connectivity_data = pd.DataFrame(
            {"pre": [1, 2], "post": [2, 1], "weight": [5, 4]}
        )
        mock_get_connectivity.return_value = mock_connectivity_data

        result = get_synapse_counts(1, min_size=10)

        # Check that min_size was passed to get_connectivity
        mock_get_connectivity.assert_called_once_with(
            neuron_ids=1,
            upstream=True,
            downstream=True,
            threshold=1,
            min_size=10,
            materialization="latest",
            clean=True,
            update_ids=True,
            dataset="latest",
        )

    @patch("crantpy.queries.connections.get_connectivity")
    def test_get_synapse_counts_live_materialization(self, mock_get_connectivity):
        """Test get_synapse_counts with live materialization."""
        mock_connectivity_data = pd.DataFrame(
            {"pre": [1, 2], "post": [2, 1], "weight": [5, 4]}
        )
        mock_get_connectivity.return_value = mock_connectivity_data

        result = get_synapse_counts(1, materialization="live")

        # Check that materialization was passed correctly
        mock_get_connectivity.assert_called_once_with(
            neuron_ids=1,
            upstream=True,
            downstream=True,
            threshold=1,
            min_size=None,
            materialization="live",
            clean=True,
            update_ids=True,
            dataset="latest",
        )

    @patch("crantpy.queries.connections.get_connectivity")
    def test_get_synapse_counts_with_dataset(self, mock_get_connectivity):
        """Test get_synapse_counts with specific dataset."""
        mock_connectivity_data = pd.DataFrame(
            {"pre": [1, 2], "post": [2, 1], "weight": [5, 4]}
        )
        mock_get_connectivity.return_value = mock_connectivity_data

        result = get_synapse_counts(1, dataset="sandbox")

        # Check that dataset was passed correctly
        mock_get_connectivity.assert_called_once_with(
            neuron_ids=1,
            upstream=True,
            downstream=True,
            threshold=1,
            min_size=None,
            materialization="latest",
            clean=True,
            update_ids=True,
            dataset="sandbox",
        )

    @patch("crantpy.queries.connections.get_connectivity")
    def test_get_synapse_counts_neuron_criteria(self, mock_get_connectivity):
        """Test get_synapse_counts with NeuronCriteria input."""
        mock_connectivity_data = pd.DataFrame(
            {"pre": [1, 2], "post": [2, 1], "weight": [5, 4]}
        )
        mock_get_connectivity.return_value = mock_connectivity_data

        # Mock NeuronCriteria that resolves to [1, 2]
        with patch("crantpy.queries.connections.parse_root_ids") as mock_parse:
            mock_parse.return_value = ["1", "2"]

            result = get_synapse_counts("mock_criteria")

            # Check that parse_root_ids was called
            mock_parse.assert_called_once_with("mock_criteria")

            # Check result structure
            assert len(result) == 2
            assert 1 in result.index and 2 in result.index

    @patch("crantpy.queries.connections.get_connectivity")
    def test_get_synapse_counts_integer_conversion(self, mock_get_connectivity):
        """Test that neuron IDs are properly converted to integers."""
        mock_connectivity_data = pd.DataFrame(
            {"pre": [123, 456], "post": [456, 123], "weight": [5, 4]}
        )
        mock_get_connectivity.return_value = mock_connectivity_data

        with patch("crantpy.queries.connections.parse_root_ids") as mock_parse:
            mock_parse.return_value = ["123", "456"]  # Return as strings

            result = get_synapse_counts(["123", "456"])

            # Check that IDs were converted to integers in the index
            assert result.index.dtype == int
            assert 123 in result.index and 456 in result.index

    @patch("crantpy.queries.connections.get_connectivity")
    def test_get_synapse_counts_no_self_connections(self, mock_get_connectivity):
        """Test that autapses are properly excluded via clean=True."""
        # Connectivity data without autapses (clean=True should remove them)
        mock_connectivity_data = pd.DataFrame(
            {"pre": [1, 2], "post": [2, 1], "weight": [5, 4]}
        )
        mock_get_connectivity.return_value = mock_connectivity_data

        result = get_synapse_counts([1, 2])

        # Verify clean=True was passed to get_connectivity
        mock_get_connectivity.assert_called_once_with(
            neuron_ids=[1, 2],
            upstream=True,
            downstream=True,
            threshold=1,
            min_size=None,
            materialization="latest",
            clean=True,
            update_ids=True,
            dataset="latest",
        )


class TestAutomaticIDUpdates:
    """Test cases for automatic ID updates in connectivity functions."""

    @patch("crantpy.queries.connections.get_cave_client")
    @patch("crantpy.utils.cave.segmentation.update_ids")
    def test_get_synapses_updates_ids_by_default(
        self, mock_update_ids, mock_get_client
    ):
        """Test that get_synapses updates IDs by default."""
        mock_get_client.return_value = create_mock_cave_client()

        # Mock update_ids to return updated IDs
        mock_update_ids.return_value = pd.DataFrame(
            {
                "old_id": [1, 2],
                "new_id": [10, 20],
                "confidence": [1, 1],
                "changed": [True, True],
            }
        )

        # Call get_synapses with update_ids=True (default)
        result = get_synapses(pre_ids=[1, 2], dataset="latest")

        # Verify update_ids was called for pre_ids
        assert mock_update_ids.called
        call_args = mock_update_ids.call_args[0]
        assert set(call_args[0]) == {1, 2}  # Check that IDs were passed

    @patch("crantpy.queries.connections.get_cave_client")
    @patch("crantpy.utils.cave.segmentation.update_ids")
    def test_get_synapses_skips_updates_when_disabled(
        self, mock_update_ids, mock_get_client
    ):
        """Test that get_synapses skips ID updates when update_ids=False."""
        mock_get_client.return_value = create_mock_cave_client()

        # Call get_synapses with update_ids=False
        result = get_synapses(pre_ids=[1, 2], update_ids=False, dataset="latest")

        # Verify update_ids was NOT called
        assert not mock_update_ids.called

    @patch("crantpy.queries.connections.get_synapses")
    def test_get_adjacency_passes_update_ids_parameter(self, mock_get_synapses):
        """Test that get_adjacency passes update_ids to get_synapses."""
        mock_get_synapses.return_value = pd.DataFrame(
            columns=["pre_pt_root_id", "post_pt_root_id"]
        )

        # Test with update_ids=True (default)
        get_adjacency(pre_ids=[1], post_ids=[2], dataset="latest")
        assert mock_get_synapses.call_args[1]["update_ids"] == True

        # Test with update_ids=False
        get_adjacency(pre_ids=[1], post_ids=[2], update_ids=False, dataset="latest")
        assert mock_get_synapses.call_args[1]["update_ids"] == False

    @patch("crantpy.queries.connections.get_synapses")
    def test_get_connectivity_passes_update_ids_parameter(self, mock_get_synapses):
        """Test that get_connectivity passes update_ids to get_synapses."""
        mock_get_synapses.return_value = pd.DataFrame(
            columns=["pre_pt_root_id", "post_pt_root_id", "id"]
        )

        # Test with update_ids=True (default)
        get_connectivity(neuron_ids=[1], dataset="latest")
        assert mock_get_synapses.call_args[1]["update_ids"] == True

        # Test with update_ids=False
        get_connectivity(neuron_ids=[1], update_ids=False, dataset="latest")
        assert mock_get_synapses.call_args[1]["update_ids"] == False

    @patch("crantpy.queries.connections.get_connectivity")
    def test_get_synapse_counts_passes_update_ids_parameter(
        self, mock_get_connectivity
    ):
        """Test that get_synapse_counts passes update_ids to get_connectivity."""
        mock_get_connectivity.return_value = pd.DataFrame(
            columns=["pre", "post", "weight"]
        )

        # Test with update_ids=True (default)
        get_synapse_counts(neuron_ids=[1], dataset="latest")
        assert mock_get_connectivity.call_args[1]["update_ids"] == True

        # Test with update_ids=False
        get_synapse_counts(neuron_ids=[1], update_ids=False, dataset="latest")
        assert mock_get_connectivity.call_args[1]["update_ids"] == False

    @patch("crantpy.queries.connections.get_cave_client")
    @patch("crantpy.utils.cave.segmentation.update_ids")
    def test_id_update_with_failed_updates(self, mock_update_ids, mock_get_client):
        """Test that get_synapses handles failed ID updates gracefully."""
        mock_get_client.return_value = create_mock_cave_client()

        # Mock update_ids to return some failed updates (confidence=0)
        mock_update_ids.return_value = pd.DataFrame(
            {
                "old_id": [1, 2, 3],
                "new_id": [10, 2, 30],  # ID 2 unchanged
                "confidence": [1, 0, 1],  # ID 2 failed to update
                "changed": [True, False, True],
            }
        )

        # Call get_synapses - should still work but log warning
        result = get_synapses(pre_ids=[1, 2, 3], dataset="latest")

        # Verify update_ids was called
        assert mock_update_ids.called


if __name__ == "__main__":
    pytest.main([__file__])


if __name__ == "__main__":
    pytest.main([__file__])
