import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from crantpy.queries.fetch_neurons import NeuronCriteria

class DummyNoMatchesError(Exception):
    pass

def dummy_get_seatable_annotations(clear_cache=False):
    return pd.DataFrame({
        'root_id': [1, 2, 3, 4],
        'cell_class': ['foo', 'bar', 'foobar', 'baz'],
        'status': [['A', 'B'], ['B'], ['A'], ['C']],
        'side': ['L', 'R', 'L', 'R']
    })

@patch('crantpy.queries.fetch_neurons.get_seatable_annotations', side_effect=dummy_get_seatable_annotations)
def test_neuroncriteria_exact(mock_ann):
    nc = NeuronCriteria(cell_class='foo')
    roots = nc.get_roots()
    assert set(roots) == {1}

@patch('crantpy.queries.fetch_neurons.get_seatable_annotations', side_effect=dummy_get_seatable_annotations)
def test_neuroncriteria_substring(mock_ann):
    nc = NeuronCriteria(cell_class='foo', exact=False)
    roots = nc.get_roots()
    assert set(roots) == {1, 3}

@patch('crantpy.queries.fetch_neurons.get_seatable_annotations', side_effect=dummy_get_seatable_annotations)
def test_neuroncriteria_list_column_any(mock_ann):
    nc = NeuronCriteria(status='A')
    roots = nc.get_roots()
    assert set(roots) == {1, 3}

@patch('crantpy.queries.fetch_neurons.get_seatable_annotations', side_effect=dummy_get_seatable_annotations)
def test_neuroncriteria_list_column_match_all(mock_ann):
    nc = NeuronCriteria(status=['A', 'B'], match_all=True)
    roots = nc.get_roots()
    assert set(roots) == {1}

@patch('crantpy.queries.fetch_neurons.get_seatable_annotations', side_effect=dummy_get_seatable_annotations)
def test_neuroncriteria_side_and_status(mock_ann):
    nc = NeuronCriteria(side='L', status='A')
    roots = nc.get_roots()
    assert set(roots) == {1, 3}

@patch('crantpy.queries.fetch_neurons.get_seatable_annotations', side_effect=dummy_get_seatable_annotations)
def test_neuroncriteria_no_criteria(mock_ann):
    nc = NeuronCriteria()
    roots = nc.get_roots()
    assert set(roots) == {1, 2, 3, 4}
