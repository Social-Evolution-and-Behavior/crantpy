import pytest
import pandas as pd
from crantpy.utils.utils import filter_df, match_dtype

def test_match_dtype_int():
    assert match_dtype('5', 'int64') == 5

def test_match_dtype_float():
    assert match_dtype('3.14', 'float64') == 3.14

def test_match_dtype_bool():
    assert match_dtype('True', 'bool') is True

def test_match_dtype_str():
    assert match_dtype(123, 'object') == '123'

def test_filter_df_exact_string():
    df = pd.DataFrame({'col': ['foo', 'bar', 'baz']})
    result = filter_df(df, 'col', 'foo')
    assert len(result) == 1 and result.iloc[0]['col'] == 'foo'

def test_filter_df_substring():
    df = pd.DataFrame({'col': ['foo', 'foobar', 'baz']})
    result = filter_df(df, 'col', 'foo', exact=False)
    assert set(result['col']) == {'foo', 'foobar'}

def test_filter_df_list_column():
    df = pd.DataFrame({'col': [['a', 'b'], ['b', 'c'], ['c']]})
    result = filter_df(df, 'col', 'a')
    assert len(result) == 1 and result.iloc[0]['col'] == ['a', 'b']

def test_filter_df_list_column_match_all():
    df = pd.DataFrame({'col': [['a', 'b'], ['b', 'c'], ['a', 'b', 'c']]})
    result = filter_df(df, 'col', ['a', 'b'], match_all=True)
    assert set(tuple(x) for x in result['col']) == {('a', 'b'), ('a', 'b', 'c')}
