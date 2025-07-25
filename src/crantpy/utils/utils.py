# -*- coding: utf-8 -*-
"""
This module contains utility functions for crantpy.
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

import navis
import pandas as pd
import requests
from pandas.api.types import pandas_dtype

T = TypeVar('T')

# set up logging and options to change logging level
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
def set_logging_level(level: str) -> None:
    """
    Sets the logging level for the logger.
    
    Parameters
    ----------
    level : str
        The logging level to set. Options are 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
    """
    logging_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    if level in logging_levels:
        logging.getLogger().setLevel(logging_levels[level])
    else:
        raise ValueError(f"Invalid logging level: {level}. Choose from {list(logging_levels.keys())}.")


# Custom functions
def create_sql_query(table_name: str, fields: List[str], condition: Optional[str] = None, 
                    limit: Optional[int] = None, start: Optional[int] = None) -> str:
    """
    Creates a SQL query to get the specified fields from the specified table.
    
    Parameters
    ----------
    table_name : str
        The name of the table to query.
    fields : List[str]
        The list of field names to include in the query.
    condition : str, optional
        The WHERE clause of the query.
    limit : int, optional
        The maximum number of rows to return.
    start : int, optional
        The number of rows to skip (OFFSET).
        
    Returns
    -------
    str
        The constructed SQL query string.
    """
    # Create the SQL query
    sql_query = f"SELECT {', '.join(fields)} FROM {table_name}"
    if condition:
        sql_query += f" WHERE {condition}"
    if limit is not None:
        sql_query += f" LIMIT {limit}"
    if start is not None:
        sql_query += f" OFFSET {start}"
    return sql_query


def match_dtype(value: Any, dtype: Union[str, type]) -> Any:
    """
    Match the dtype of a value to a given dtype.
    
    Parameters
    ----------
    value : Any
        The value to convert.
    dtype : str or type
        The target dtype to convert to.
        
    Returns
    -------
    Any
        The converted value.
        
    Raises
    ------
    ValueError
        If the dtype is not supported.
    """
    if pd.api.types.is_integer_dtype(dtype):
        return int(value)
    elif pd.api.types.is_float_dtype(dtype):
        return float(value)
    elif pd.api.types.is_bool_dtype(dtype):
        return bool(value)
    elif pd.api.types.is_string_dtype(dtype):
        return str(value)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")



def _check_list_membership(cell_value: Any, filter_value: Any, is_filter_list: bool) -> bool:
    """Helper function to check membership within a list cell.
    
    Parameters
    ----------
    cell_value : Any
        The value of the cell to check.
    filter_value : Any
        The value or values to filter by.
    is_filter_list : bool
        Whether filter_value is a list.
        
    Returns
    -------
    bool
        True if the filter value(s) are found in the cell value, False otherwise.
    """
    if not isinstance(cell_value, list):
        return False # Cell itself is not a list
    if not is_filter_list:
        # Filter value is a single item, check if it's in the cell's list
        return filter_value in cell_value
    else:
        # Filter value is a list, check if *any* item from filter_value is in the cell's list
        return any(item in cell_value for item in filter_value)

def filter_df(df: pd.DataFrame, column: str, value: Any, regex: bool = False, case: bool = False, match_all: bool = False, exact: bool = True) -> pd.DataFrame:
    """
    This function filters a df based on a column and a value.
    It can handle string, numeric, and list-containing columns.

    Args:
        df (pd.DataFrame): The df to filter.
        column (str): The column to filter on.
        value (Any): The value(s) to filter by (can be single item or list).
        regex (bool): Whether to use regex for string columns (not applicable for list columns).
        case (bool): Whether to ignore case for string columns (not applicable for list columns).
        match_all (bool): For list-containing columns and list filter values,
                          if True, requires all filter values to be present in the cell's list.
                          If False, requires at least one filter value to be present.
                          Defaults to False.
        exact (bool): For string columns, if False, use substring (contains) matching instead of exact match.

    Returns:
        pd.DataFrame: The filtered df.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")

    # Drop rows where the target column is NaN to avoid errors in comparisons/iterations
    df_filtered = df.dropna(subset=[column]).copy()
    if df_filtered.empty:
        return df_filtered # Return empty if all were NaN

    # Determine if the column likely contains lists (check first non-NA value)
    first_val = df_filtered[column].iloc[0]
    col_contains_lists = isinstance(first_val, list)

    if col_contains_lists:
        if regex:
            logging.warning(f"Regex filtering is not supported for list-containing column '{column}'. Ignoring regex flag.")
        is_filter_list = navis.utils.is_iterable(value) and not isinstance(value, str) # Check if filter value is a list/iterable

        if is_filter_list and match_all:
            # Apply "AND" logic: check if *all* items in the filter list `value` are present in the cell's list
            mask = df_filtered[column].apply(lambda cell_list: isinstance(cell_list, list) and all(item in cell_list for item in value))
        else:
            # Apply "OR" logic (or single item check) using the helper function
            mask = df_filtered[column].apply(_check_list_membership, args=(value, is_filter_list))

        return df_filtered[mask]
    else:
        # --- Original logic for non-list columns ---
        dt = df_filtered[column].dtype
        is_str_col = pd.api.types.is_string_dtype(dt) or dt == 'object' # Broader check for string-like

        # Ensure filter value(s) match column type if not regex on string
        if not (regex and is_str_col):
             try:
                 if navis.utils.is_iterable(value) and not isinstance(value, str):
                     # Convert each item in the list value
                     value = [match_dtype(v, dt) for v in value]
                 else:
                     # Convert single value
                     value = match_dtype(value, dt)
             except (ValueError, TypeError) as e:
                 raise ValueError(f'Unable to convert filter value for column "{column}" to type {dt}: {e}')

        # Apply filtering
        is_filter_list = navis.utils.is_iterable(value) and not isinstance(value, str)
        if is_filter_list:
            if regex and is_str_col:
                 # Combine patterns for regex matching against list of values
                 pattern = '|'.join(map(str, value))
                 df_filtered = df_filtered[df_filtered[column].astype(str).str.contains(pattern, na=False, case=case, regex=True)]
            elif is_str_col and not case:
                 # Case-insensitive isin for string columns
                 lower_value = {str(v).lower() for v in value}
                 df_filtered = df_filtered[df_filtered[column].astype(str).str.lower().isin(lower_value)]
            elif is_str_col and not exact:
                 # Substring (contains) match for each value in list
                 mask = df_filtered[column].astype(str).apply(lambda x: any(str(v).lower() in x.lower() for v in value))
                 df_filtered = df_filtered[mask]
            else:
                 # Standard isin for exact matches or non-string types
                 df_filtered = df_filtered[df_filtered[column].isin(value)]
        else: # Single filter value
            if regex and is_str_col:
                 df_filtered = df_filtered[df_filtered[column].astype(str).str.contains(str(value), na=False, case=case, regex=True)]
            elif is_str_col and not case:
                 # Case-insensitive comparison for single string value
                 if not exact:
                     # Substring (contains) match, case-insensitive
                     mask = df_filtered[column].astype(str).str.lower().str.contains(str(value).lower(), na=False)
                     df_filtered = df_filtered[mask]
                 else:
                     df_filtered = df_filtered[df_filtered[column].astype(str).str.lower() == str(value).lower()]
            elif is_str_col and not exact:
                 # Substring (contains) match, case-sensitive
                 mask = df_filtered[column].astype(str).str.contains(str(value), na=False)
                 df_filtered = df_filtered[mask]
            else:
                 # Standard equality check
                 df_filtered = df_filtered[df_filtered[column] == value]
        return df_filtered
