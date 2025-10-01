# -*- coding: utf-8 -*-
"""Tests for the connections module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from crantpy.queries.connections import get_synapses, get_adjacency


# Mock data for testing
def create_mock_synapses_data():
    """Create mock synapse data for testing."""
    return pd.DataFrame({
        'pre_pt_root_id': [1, 1, 2, 2, 3, 3, 1, 2],
        'post_pt_root_id': [2, 3, 3, 1, 1, 2, 2, 3],
        'pre_pt_supervoxel_id': [10, 11, 20, 21, 30, 31, 12, 22],
        'post_pt_supervoxel_id': [25, 35, 36, 15, 16, 26, 25, 36],
        'cleft_score': [0.8, 0.9, 0.7, 0.85, 0.6, 0.75, 0.8, 0.7],
        'x': [100, 110, 200, 210, 300, 310, 105, 205],
        'y': [1000, 1100, 2000, 2100, 3000, 3100, 1050, 2050],
        'z': [5000, 5100, 6000, 6100, 7000, 7100, 5050, 6050]
    })


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
    
    @patch('crantpy.queries.connections.get_cave_client')
    def test_get_synapses_with_pre_ids_only(self, mock_get_client):
        """Test get_synapses with only pre_ids specified."""
        mock_get_client.return_value = create_mock_cave_client()
        
        result = get_synapses(pre_ids=[1, 2])
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        mock_get_client.assert_called_once()
    
    @patch('crantpy.queries.connections.get_cave_client')
    def test_get_synapses_with_post_ids_only(self, mock_get_client):
        """Test get_synapses with only post_ids specified.""" 
        mock_get_client.return_value = create_mock_cave_client()
        
        result = get_synapses(post_ids=[2, 3])
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        mock_get_client.assert_called_once()
    
    @patch('crantpy.queries.connections.get_cave_client')
    def test_get_synapses_with_both_ids(self, mock_get_client):
        """Test get_synapses with both pre_ids and post_ids specified."""
        mock_get_client.return_value = create_mock_cave_client()
        
        result = get_synapses(pre_ids=[1], post_ids=[2, 3])
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        mock_get_client.assert_called_once()
    
    @patch('crantpy.queries.connections.get_cave_client')
    def test_get_synapses_single_int_input(self, mock_get_client):
        """Test get_synapses with single integer inputs."""
        mock_get_client.return_value = create_mock_cave_client()
        
        result = get_synapses(pre_ids=1, post_ids=2)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        mock_get_client.assert_called_once()
    
    @patch('crantpy.queries.connections.get_cave_client')
    def test_get_synapses_string_input(self, mock_get_client):
        """Test get_synapses with string inputs."""
        mock_get_client.return_value = create_mock_cave_client()
        
        result = get_synapses(pre_ids="1", post_ids=["2", "3"])
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        mock_get_client.assert_called_once()
    
    def test_get_synapses_no_ids_raises_error(self):
        """Test that get_synapses raises ValueError when no IDs are provided."""
        with pytest.raises(ValueError, match="You must provide at least one of pre_ids or post_ids"):
            get_synapses()
    
    @patch('crantpy.queries.connections.get_cave_client')
    def test_get_synapses_with_threshold(self, mock_get_client):
        """Test get_synapses with threshold parameter."""
        mock_client = create_mock_cave_client()
        mock_get_client.return_value = mock_client
        
        result = get_synapses(pre_ids=[1], threshold=2)
        
        assert isinstance(result, pd.DataFrame)
        mock_get_client.assert_called_once()
    
    @patch('crantpy.queries.connections.get_cave_client')
    def test_get_synapses_with_min_score(self, mock_get_client):
        """Test get_synapses with min_score parameter."""
        mock_client = create_mock_cave_client()
        mock_get_client.return_value = mock_client
        
        # Create mock data with cleft_score column
        mock_data = create_mock_synapses_data()
        mock_client.materialize.query_table.return_value = mock_data
        
        result = get_synapses(pre_ids=[1], min_score=0.8)
        
        assert isinstance(result, pd.DataFrame)
        # Should filter out rows with cleft_score < 0.8
        assert all(result['cleft_score'] >= 0.8)
    
    @patch('crantpy.queries.connections.get_cave_client')
    def test_get_synapses_drop_duplicates(self, mock_get_client):
        """Test get_synapses with drop_duplicates parameter."""
        mock_client = create_mock_cave_client()
        mock_get_client.return_value = mock_client
        
        # Create data with duplicate supervoxel pairs
        duplicate_data = pd.DataFrame({
            'pre_pt_root_id': [1, 1, 1],
            'post_pt_root_id': [2, 2, 3],
            'pre_pt_supervoxel_id': [10, 10, 11],
            'post_pt_supervoxel_id': [20, 20, 30],
            'cleft_score': [0.8, 0.9, 0.7]
        })
        mock_client.materialize.query_table.return_value = duplicate_data
        
        result = get_synapses(pre_ids=[1], drop_duplicates=True)
        
        assert isinstance(result, pd.DataFrame)
        # Should remove duplicate supervoxel pairs
        unique_pairs = result[['pre_pt_supervoxel_id', 'post_pt_supervoxel_id']].drop_duplicates()
        assert len(result) == len(unique_pairs)
    
    @patch('crantpy.queries.connections.get_cave_client')
    def test_get_synapses_live_query(self, mock_get_client):
        """Test get_synapses with live=True parameter."""
        mock_client = create_mock_cave_client()
        mock_get_client.return_value = mock_client
        
        result = get_synapses(pre_ids=[1], live=True)
        
        assert isinstance(result, pd.DataFrame)
        # Should call live_query instead of query_table
        mock_client.materialize.live_query.assert_called_once()
        mock_client.materialize.query_table.assert_not_called()
    
    @patch('crantpy.queries.connections.get_cave_client')
    def test_get_synapses_specific_materialization(self, mock_get_client):
        """Test get_synapses with specific materialization version."""
        mock_client = create_mock_cave_client()
        mock_get_client.return_value = mock_client
        
        result = get_synapses(pre_ids=[1], materialization=100)
        
        assert isinstance(result, pd.DataFrame)
        # Should use the specified materialization version
        mock_client.materialize.query_table.assert_called_once()
        call_kwargs = mock_client.materialize.query_table.call_args[1]
        assert call_kwargs['materialization_version'] == 100
    
    @patch('crantpy.queries.connections.get_cave_client')
    def test_get_synapses_empty_result(self, mock_get_client):
        """Test get_synapses when query returns empty DataFrame."""
        mock_client = create_mock_cave_client()
        mock_client.materialize.query_table.return_value = pd.DataFrame()
        mock_get_client.return_value = mock_client
        
        result = get_synapses(pre_ids=[999])  # Non-existent ID
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    @patch('crantpy.queries.connections.get_cave_client')
    def test_get_synapses_with_dataset(self, mock_get_client):
        """Test get_synapses with dataset parameter."""
        mock_client = create_mock_cave_client()
        mock_get_client.return_value = mock_client
        
        result = get_synapses(pre_ids=[1], dataset='latest')
        
        assert isinstance(result, pd.DataFrame)
        # Should pass dataset to get_cave_client
        mock_get_client.assert_called_with(dataset='latest')


class TestGetAdjacency:
    """Test cases for get_adjacency function."""
    
    def test_get_adjacency_basic(self):
        """Test basic functionality of get_adjacency."""
        pre_ids = [1, 1, 2, 2, 3]
        post_ids = [2, 3, 3, 1, 1]
        
        result = get_adjacency(pre_ids, post_ids)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 3)  # 3 unique pre_ids, 3 unique post_ids
        assert result.loc[1, 2] == 1  # 1 synapse from neuron 1 to neuron 2
        assert result.loc[1, 3] == 1  # 1 synapse from neuron 1 to neuron 3
        assert result.loc[2, 3] == 1  # 1 synapse from neuron 2 to neuron 3
        assert result.loc[2, 1] == 1  # 1 synapse from neuron 2 to neuron 1
        assert result.loc[3, 1] == 1  # 1 synapse from neuron 3 to neuron 1
    
    def test_get_adjacency_multiple_synapses(self):
        """Test get_adjacency with multiple synapses between same pair."""
        pre_ids = [1, 1, 1, 2, 2]
        post_ids = [2, 2, 3, 3, 3]
        
        result = get_adjacency(pre_ids, post_ids)
        
        assert result.loc[1, 2] == 2  # 2 synapses from neuron 1 to neuron 2
        assert result.loc[1, 3] == 1  # 1 synapse from neuron 1 to neuron 3
        assert result.loc[2, 3] == 2  # 2 synapses from neuron 2 to neuron 3
    
    def test_get_adjacency_symmetric(self):
        """Test get_adjacency with symmetric=True."""
        pre_ids = [1, 2, 3, 4]
        post_ids = [2, 3, 1, 5]  # Note: 4 and 5 are not in both lists
        
        result = get_adjacency(pre_ids, post_ids, symmetric=True)
        
        # Only neurons 1, 2, 3 appear in both pre and post
        expected_neurons = [1, 2, 3]
        assert list(result.index) == expected_neurons
        assert list(result.columns) == expected_neurons
        assert result.shape == (3, 3)
    
    def test_get_adjacency_asymmetric(self):
        """Test get_adjacency with symmetric=False (default)."""
        pre_ids = [1, 2, 3]
        post_ids = [2, 3, 4]
        
        result = get_adjacency(pre_ids, post_ids, symmetric=False)
        
        # Rows should be unique pre_ids, columns should be unique post_ids
        assert set(result.index) == {1, 2, 3}
        assert set(result.columns) == {2, 3, 4}
        assert result.shape == (3, 3)
    
    def test_get_adjacency_empty_input(self):
        """Test get_adjacency with empty inputs."""
        result = get_adjacency([], [])
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_get_adjacency_single_connection(self):
        """Test get_adjacency with single connection."""
        pre_ids = [1]
        post_ids = [2]
        
        result = get_adjacency(pre_ids, post_ids)
        
        assert result.shape == (1, 1)
        assert result.loc[1, 2] == 1
    
    def test_get_adjacency_numpy_arrays(self):
        """Test get_adjacency with numpy array inputs."""
        pre_ids = np.array([1, 1, 2])
        post_ids = np.array([2, 3, 3])
        
        result = get_adjacency(pre_ids, post_ids)
        
        assert isinstance(result, pd.DataFrame)
        assert result.loc[1, 2] == 1
        assert result.loc[1, 3] == 1
        assert result.loc[2, 3] == 1
    
    def test_get_adjacency_mixed_types(self):
        """Test get_adjacency with mixed int/string types."""
        pre_ids = [1, "2", 3]
        post_ids = ["2", 3, 1]
        
        result = get_adjacency(pre_ids, post_ids)
        
        assert isinstance(result, pd.DataFrame)
        # Should handle mixed types correctly
        assert result.shape[0] == 3  # 3 unique pre_ids
        assert result.shape[1] == 3  # 3 unique post_ids
    
    def test_get_adjacency_zero_entries(self):
        """Test that get_adjacency correctly sets zero entries."""
        pre_ids = [1, 2]
        post_ids = [3, 4]  # No connections between pre and post
        
        result = get_adjacency(pre_ids, post_ids)
        
        # Should have zeros for non-existing connections
        assert result.loc[1, 3] == 1
        assert result.loc[2, 4] == 1
        # These connections don't exist in the data, so should remain 0
        assert result.loc[1, 4] == 0
        assert result.loc[2, 3] == 0


if __name__ == '__main__':
    pytest.main([__file__])
