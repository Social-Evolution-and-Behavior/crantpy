from typing import Any, Dict, List, Optional, Set, Tuple, Union
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from crantpy.queries.fetch_neurons import NeuronCriteria


class DummyNoMatchesError(Exception):
    """Dummy exception class for testing."""
    pass


def dummy_get_all_seatable_annotations(clear_cache: bool = False, dataset: Optional[str] = None, 
                                      proofread_only: bool = False) -> pd.DataFrame:
    """Mock function for get_all_seatable_annotations.
    
    Parameters
    ----------
    clear_cache : bool, default False
        Dummy parameter for compatibility.
    dataset : str, optional
        Dummy parameter for compatibility.
    proofread_only : bool, default False
        Dummy parameter for compatibility.
    
    Returns
    -------
    pd.DataFrame
        Mock annotation data.
    """
    return pd.DataFrame({
        'root_id': [1, 2, 3, 4],
        'cell_class': ['foo', 'bar', 'foobar', 'baz'],
        'status': [['A', 'B'], ['B'], ['A'], ['C']],
        'side': ['L', 'R', 'L', 'R']
    })

@patch('crantpy.queries.fetch_neurons.get_all_seatable_annotations', side_effect=dummy_get_all_seatable_annotations)
def test_neuroncriteria_exact(mock_ann: MagicMock) -> None:
    """Test exact matching in NeuronCriteria."""
    nc = NeuronCriteria(cell_class='foo')
    roots = nc.get_roots()
    assert set(roots) == {1}


@patch('crantpy.queries.fetch_neurons.get_all_seatable_annotations', side_effect=dummy_get_all_seatable_annotations)
def test_neuroncriteria_substring(mock_ann: MagicMock) -> None:
    """Test substring matching in NeuronCriteria."""
    nc = NeuronCriteria(cell_class='foo', exact=False)
    roots = nc.get_roots()
    assert set(roots) == {1, 3}


@patch('crantpy.queries.fetch_neurons.get_all_seatable_annotations', side_effect=dummy_get_all_seatable_annotations)
def test_neuroncriteria_list_column_any(mock_ann: MagicMock) -> None:
    """Test list column filtering with any matching in NeuronCriteria."""
    nc = NeuronCriteria(status='A')
    roots = nc.get_roots()
    assert set(roots) == {1, 3}

@patch('crantpy.queries.fetch_neurons.get_all_seatable_annotations', side_effect=dummy_get_all_seatable_annotations)
def test_neuroncriteria_list_column_match_all(mock_ann: MagicMock) -> None:
    """Test list column filtering with all matching in NeuronCriteria."""
    nc = NeuronCriteria(status=['A', 'B'], match_all=True)
    roots = nc.get_roots()
    assert set(roots) == {1}


@patch('crantpy.queries.fetch_neurons.get_all_seatable_annotations', side_effect=dummy_get_all_seatable_annotations)
def test_neuroncriteria_side_and_status(mock_ann: MagicMock) -> None:
    """Test multiple criteria filtering in NeuronCriteria."""
    nc = NeuronCriteria(side='L', status='A')
    roots = nc.get_roots()
    assert set(roots) == {1, 3}


@patch('crantpy.queries.fetch_neurons.get_all_seatable_annotations', side_effect=dummy_get_all_seatable_annotations)
def test_neuroncriteria_no_criteria(mock_ann: MagicMock) -> None:
    """Test NeuronCriteria with no criteria specified."""
    nc = NeuronCriteria()
    roots = nc.get_roots()
    assert set(roots) == {1, 2, 3, 4}
