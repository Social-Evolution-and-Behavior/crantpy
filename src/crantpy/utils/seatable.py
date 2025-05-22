# -*- coding: utf-8 -*-
"""
This module contains functions to interact with the Seatable API for CRANTpy.
It provides functionality to fetch annotations, handle pagination,
and manage caching of results.
It also includes a function to create SQL queries for Seatable.
"""

import datetime as dt
import json
import logging
import os
from typing import Any, Dict, List, Optional, TypeVar, Union, cast

import numpy as np
import pandas as pd
import pytz
import requests
import seatable_api
from seatable_api import Base

from crantpy.utils.config import (ALL_ANNOTATION_FIELDS,
                                  CRANT_SEATABLE_ANNOTATIONS_TABLES,
                                  CRANT_SEATABLE_BASENAME,
                                  CRANT_SEATABLE_SERVER_URL,
                                  CRANT_SEATABLE_WORKSPACE_ID,
                                  CRANT_VALID_DATASETS, MAXIMUM_CACHE_DURATION,
                                  SEARCH_EXCLUDED_ANNOTATION_FIELDS)
from crantpy.utils.decorators import cached_result, inject_dataset, parse_neuroncriteria
from crantpy.utils.utils import create_sql_query

# get API TOKEN from environment variable
CRANT_SEATABLE_API_TOKEN = os.getenv('CRANTTABLE_TOKEN')
if CRANT_SEATABLE_API_TOKEN is None:
    raise ValueError("CRANTTABLE_TOKEN environment variable not set. Please set it to your Seatable API token.")

def get_seatable_base_object() -> Base:
    """
    Uses Seatable API to get the base object for the CRANTpy project.
    
    Returns
    -------
    seatable_api.Base
        The authenticated Seatable Base object.
    """
    # Login to seatable
    ac = seatable_api.Account(login_name=[], password=[], server_url=CRANT_SEATABLE_SERVER_URL)
    ac.token = CRANT_SEATABLE_API_TOKEN

    # Initialize the Base object
    base = ac.get_base(workspace_id=CRANT_SEATABLE_WORKSPACE_ID, base_name=CRANT_SEATABLE_BASENAME)
    base.auth()
    return base

def get_seatable_cache_name(*args, **kwargs) -> str:
    """
    Returns the name of the Seatable cache based on the dataset and proofread_only status.
    """
    dataset = args[0] if args else kwargs['dataset']
    proofread_only = args[1] if len(args) > 1 else kwargs['proofread_only']
    return f"{dataset}{'_proofread' if proofread_only else ''}"

@inject_dataset(allowed=CRANT_VALID_DATASETS)
@cached_result(
    cache_name="seatable_annotations", 
    key_fn=get_seatable_cache_name,
)
def get_all_seatable_annotations(
        dataset: Optional[str] = None,
        proofread_only: bool = False, 
        clear_cache: bool = False,
        check_stale: bool = True,
    ) -> pd.DataFrame:
    """
    Uses Seatable API to get the table object for the CRANTb project.
    Handles pagination to retrieve all rows even if exceeding the API limit.
    Caches the result to avoid redundant API calls.

    Parameters
    ----------
    proofread_only : bool, default False
        If True, only retrieve annotations marked as proofread.
    clear_cache : bool, default False
        If True, bypass the cache and fetch fresh data from Seatable.
    check_stale : bool, default True
        If True, check if the cache is stale before using it based on the maximum cache duration.
    dataset : str, optional
        The dataset to use. If not provided, uses the default dataset from the environment variable.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the annotations.
    """
    base = get_seatable_base_object()
    all_results = []
    start = 0
    limit = 10000  # Seatable API limit per query

    while True:
        logging.info(f"Fetching rows from {start} to {start + limit}...")
        sql_query = create_sql_query(
            table_name=CRANT_SEATABLE_ANNOTATIONS_TABLES[dataset],
            fields=ALL_ANNOTATION_FIELDS,
            start=start,
            limit=limit
        )
        try:
            results = base.query(sql_query)
        except Exception as e:
                logging.error(f"Error querying Seatable: {e}")
                break

        if not results:
            logging.info("No more results found.")
            break # Exit loop if no results are returned

        all_results.extend(results)
        logging.info(f"Retrieved {len(results)} rows in this batch.")

        if len(results) < limit:
            logging.info("Fetched all rows.")
            break # Exit loop if fewer than 'limit' rows were returned

        start += limit # Prepare for the next batch

    logging.info(f"Retrieved a total of {len(all_results)} rows.")
    # Convert the results to a pandas DataFrame
    df = pd.DataFrame(all_results) if all_results else pd.DataFrame(columns=ALL_ANNOTATION_FIELDS)

    # Apply proofread filter if requested (post-fetch)
    if proofread_only and 'proofread' in df.columns:
        try:
            return df[df['proofread'].astype(bool) == True]
        except Exception as e:
            logging.warning(f"Could not apply 'proofread' filter due to data type issue: {e}")
    
    return df