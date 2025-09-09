from typing import Any, Dict, List, Set, Tuple

import pandas as pd
import pytest

from crantpy.utils.helpers import create_sql_query, filter_df, match_dtype


def test_match_dtype_int() -> None:
    """Test matching to int dtype.
    
    This test verifies that the match_dtype function correctly converts
    a string representation of an integer to an actual integer when the
    target dtype is 'int64'.
    """
    assert match_dtype('5', 'int64') == 5


def test_match_dtype_float() -> None:
    """Test matching to float dtype.
    
    This test verifies that the match_dtype function correctly converts
    a string representation of a float to an actual float when the
    target dtype is 'float64'.
    """
    assert match_dtype('3.14', 'float64') == 3.14


def test_match_dtype_bool() -> None:
    """Test matching to bool dtype.
    
    This test verifies that the match_dtype function correctly converts
    a string representation of a boolean to an actual boolean when the
    target dtype is 'bool'. The function should return True for 'True'.
    """
    assert match_dtype('True', 'bool') is True


def test_match_dtype_str() -> None:
    """Test matching to string/object dtype.
    
    This test verifies that the match_dtype function correctly converts
    a numeric value to a string when the target dtype is 'object', which
    is often used to represent strings in pandas.
    """
    assert match_dtype(123, 'object') == '123'


def test_match_dtype_unsupported() -> None:
    """Test matching to unsupported dtype.
    
    This test verifies that the match_dtype function raises a ValueError
    when given a dtype that it doesn't support. This is important for
    error handling so that users receive clear feedback when they try
    to use an unsupported data type.
    """
    with pytest.raises(ValueError) as excinfo:
        match_dtype('value', 'unsupported_dtype')
    assert "Unsupported dtype" in str(excinfo.value)


def test_filter_df_exact_string() -> None:
    """Test filtering dataframe with exact string match.
    
    This test verifies that the filter_df function correctly filters
    a DataFrame to only include rows where a column exactly matches
    a specified string value. This is the default behavior of filter_df
    when exact=True.
    """
    df = pd.DataFrame({'col': ['foo', 'bar', 'baz']})
    result = filter_df(df, 'col', 'foo')
    assert len(result) == 1 and result.iloc[0]['col'] == 'foo'


def test_filter_df_substring() -> None:
    """Test filtering dataframe with substring match.
    
    This test verifies that the filter_df function correctly filters
    a DataFrame to include rows where a column contains a specified
    substring when exact=False. This is useful for partial matching
    when you don't need an exact match on the entire field.
    """
    df = pd.DataFrame({'col': ['foo', 'foobar', 'baz']})
    result = filter_df(df, 'col', 'foo', exact=False)
    assert set(result['col']) == {'foo', 'foobar'}


def test_filter_df_case_insensitive() -> None:
    """Test filtering dataframe with case-insensitive match.
    
    This test verifies that the filter_df function correctly filters
    a DataFrame to include rows where a column matches a specified string
    value ignoring case differences when case=False. This allows matching
    'Foo', 'fOO', 'FOO', etc. when searching for 'foo'.
    """
    df = pd.DataFrame({'col': ['Foo', 'fOO', 'BAZ']})
    result = filter_df(df, 'col', 'foo', case=False)
    assert set(result['col']) == {'Foo', 'fOO'}


def test_filter_df_regex() -> None:
    """Test filtering dataframe with regex.
    
    This test verifies that the filter_df function correctly filters
    a DataFrame to include rows where a column matches a specified
    regular expression pattern when regex=True. This provides powerful
    pattern matching capabilities beyond simple exact or substring matching.
    """
    df = pd.DataFrame({'col': ['foo1', 'foo2', 'bar1']})
    result = filter_df(df, 'col', 'foo\\d', regex=True)
    assert set(result['col']) == {'foo1', 'foo2'}


def test_filter_df_list_column() -> None:
    """Test filtering dataframe where column contains lists.
    
    This test verifies that the filter_df function correctly filters
    a DataFrame where some column values are lists (e.g., ['a', 'b']).
    It should match rows where the specified value is found in the list
    column. This is particularly important for the annotations table where
    some columns contain lists of values.
    """
    df = pd.DataFrame({'col': [['a', 'b'], ['b', 'c'], ['c']]})
    result = filter_df(df, 'col', 'a')
    assert len(result) == 1 and result.iloc[0]['col'] == ['a', 'b']


def test_filter_df_list_column_match_all() -> None:
    """Test filtering dataframe where column contains lists with match_all=True.
    
    This test verifies that the filter_df function correctly filters a DataFrame
    where some column values are lists (e.g., ['a', 'b', 'c']) and match_all=True.
    In this case, it should only match rows where ALL specified values are found
    in the column's list, not just any of them. This is crucial for queries where
    we need items that satisfy multiple criteria simultaneously.
    """
    df = pd.DataFrame({'col': [['a', 'b'], ['b', 'c'], ['a', 'b', 'c']]})
    result = filter_df(df, 'col', ['a', 'b'], match_all=True)
    assert set(tuple(x) for x in result['col']) == {('a', 'b'), ('a', 'b', 'c')}


def test_filter_df_numeric() -> None:
    """Test filtering dataframe with numeric values.
    
    This test verifies that the filter_df function correctly filters
    a DataFrame to include rows where a numeric column exactly matches
    a specified numeric value. It checks that type conversion is handled
    correctly for numeric columns.
    """
    df = pd.DataFrame({'col': [1, 2, 3]})
    result = filter_df(df, 'col', 2)
    assert len(result) == 1 and result.iloc[0]['col'] == 2


def test_filter_df_numeric_list() -> None:
    """Test filtering dataframe with a list of numeric values.
    
    This test verifies that the filter_df function correctly filters
    a DataFrame to include rows where a numeric column matches any value
    from a list of specified numeric values. This is equivalent to an SQL
    'IN' clause and useful for filtering on multiple possible values.
    """
    df = pd.DataFrame({'col': [1, 2, 3, 4, 5]})
    result = filter_df(df, 'col', [2, 4])
    assert set(result['col']) == {2, 4}


def test_filter_df_nonexistent_column() -> None:
    """Test filtering dataframe with a non-existent column.
    
    This test verifies that the filter_df function raises a KeyError
    with an appropriate error message when a non-existent column name
    is specified. This helps users identify when they're trying to filter
    on a column that doesn't exist in the DataFrame.
    """
    df = pd.DataFrame({'col': [1, 2, 3]})
    with pytest.raises(KeyError) as excinfo:
        filter_df(df, 'nonexistent', 'value')
    assert "not found in DataFrame" in str(excinfo.value)


def test_filter_df_empty_result() -> None:
    """Test filtering dataframe resulting in empty dataframe.
    
    This test verifies that the filter_df function correctly returns an
    empty DataFrame when no rows match the filter criteria. This is important
    to ensure the function handles edge cases gracefully and doesn't raise
    exceptions when nothing matches.
    """
    df = pd.DataFrame({'col': [1, 2, 3]})
    result = filter_df(df, 'col', 999)
    assert result.empty


def test_create_sql_query_basic() -> None:
    """Test creating basic SQL query.
    
    This test verifies that the create_sql_query function correctly generates
    a basic SQL SELECT query with the specified table and fields. The basic
    form should be 'SELECT field1, field2 FROM table'.
    """
    sql = create_sql_query("table", ["field1", "field2"])
    assert sql == "SELECT field1, field2 FROM table"


def test_create_sql_query_with_condition() -> None:
    """Test creating SQL query with a condition.
    
    This test verifies that the create_sql_query function correctly generates
    a SQL query with a WHERE clause when a condition is provided. The resulting
    query should be in the form 'SELECT field1 FROM table WHERE field1 = 'value''.
    This is used for filtering data at the database level.
    """
    sql = create_sql_query("table", ["field1"], "field1 = 'value'")
    assert sql == "SELECT field1 FROM table WHERE field1 = 'value'"


def test_create_sql_query_with_limit() -> None:
    """Test creating SQL query with a limit.
    
    This test verifies that the create_sql_query function correctly generates
    a SQL query with a LIMIT clause when a limit parameter is provided. This
    is important for pagination and avoiding large result sets that could
    impact performance.
    """
    sql = create_sql_query("table", ["field1"], limit=10)
    assert sql == "SELECT field1 FROM table LIMIT 10"


def test_create_sql_query_with_offset() -> None:
    """Test creating SQL query with an offset.
    
    This test verifies that the create_sql_query function correctly generates
    a SQL query with an OFFSET clause when a start parameter is provided. This
    is used for pagination in combination with LIMIT to fetch a specific
    "page" of results.
    """
    sql = create_sql_query("table", ["field1"], start=20)
    assert sql == "SELECT field1 FROM table OFFSET 20"


def test_create_sql_query_full() -> None:
    """Test creating SQL query with all parameters.
    
    This test verifies that the create_sql_query function correctly generates
    a complex SQL query with all possible parameters: table name, fields,
    WHERE condition, LIMIT, and OFFSET. This is important to ensure all
    parts of the query are properly combined in the correct order.
    """
    sql = create_sql_query("table", ["field1", "field2"], "field1 = 'value'", 10, 20)
    assert sql == "SELECT field1, field2 FROM table WHERE field1 = 'value' LIMIT 10 OFFSET 20"
