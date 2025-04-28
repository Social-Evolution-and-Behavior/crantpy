# -*- coding: utf-8 -*-
"""
This module contains functions to interact with the Seatable API for CRANTpy.
It provides functionality to fetch annotations, handle pagination,
and manage caching of results.
It also includes a function to create SQL queries for Seatable.
"""

import os
import json
import requests
import logging
import pandas as pd
import numpy as np
import seatable_api
from typing import List, Optional, Any
from crantpy.utils.utils import create_sql_query

CRANT_SEATABLE_SERVER_URL = "https://cloud.seatable.io/"
CRANT_SEATABLE_WORKSPACE_ID = "62919"
CRANT_SEATABLE_BASENAME = "CRANTb"
CRANT_SEATABLE_ANNOTATIONS_TABLE = "CRANTb_meta"

# get API TOKEN from environment variable
CRANT_SEATABLE_API_TOKEN = os.getenv('CRANTTABLE_TOKEN')
if CRANT_SEATABLE_API_TOKEN is None:
    raise ValueError("CRANTTABLE_TOKEN environment variable not set. Please set it to your Seatable API token.")

ALL_FIELDS = [
    "root_id",
    "root_id_processed",
    "supervoxel_id",
    "position",
    "nucleus_id",
    "nucleus_position",
    "root_position",
    "cave_table",
    "proofread",
    "status",
    "region",
    "proofreader_notes",
    "side",
    "nerve",
    "tract",
    "hemilineage",
    "flow",
    "super_class",
    "cell_class",
    "cell_type",
    "cell_subtype",
    "cell_instance",
    "known_nt",
    "known_nt_source",
    "alternative_names",
    "annotator_notes",
    "user_annotator",
    "user_proofreader",
    "ngl_link",
    "date_proofread",
]

SEARCH_EXCLUDED_FIELDS = [
    "root_id_processed",
    "supervoxel_id",
    "position",
    "nucleus_position",
    "root_position",
]

# Cache for annotations to avoid repeated API calls within the same session
_CACHED_ANNOTATIONS = None

def get_seatable_base_object():
    """
    Uses Seatable API to get the base object for the CRANTpy project.
    """
    # Login to seatable
    ac = seatable_api.Account(login_name=[], password=[], server_url=CRANT_SEATABLE_SERVER_URL)
    ac.token = CRANT_SEATABLE_API_TOKEN

    # Initialize the Base object
    base = ac.get_base(workspace_id=CRANT_SEATABLE_WORKSPACE_ID, base_name=CRANT_SEATABLE_BASENAME)
    base.auth()
    return base

def get_seatable_annotations(proofread_only=False, clear_cache=False):
    """
    Uses Seatable API to get the table object for the CRANTb project.
    Handles pagination to retrieve all rows even if exceeding the API limit.
    Caches the result to avoid redundant API calls.

    Parameters
    ----------
    proofread_only : bool, optional
        If True, only retrieve annotations marked as proofread. Defaults to False.
    clear_cache : bool, optional
        If True, bypass the cache and fetch fresh data from Seatable. Defaults to False.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the annotations.
    """
    global _CACHED_ANNOTATIONS
    if not clear_cache and _CACHED_ANNOTATIONS is not None:
        logging.info("Using cached annotations.")
        df = _CACHED_ANNOTATIONS
    else:
        logging.info("Fetching annotations from Seatable...")
        base = get_seatable_base_object()
        all_results = []
        start = 0
        limit = 10000  # Seatable API limit per query

        while True:
            logging.info(f"Fetching rows from {start} to {start + limit}...")
            sql_query = create_sql_query(
                table_name=CRANT_SEATABLE_ANNOTATIONS_TABLE,
                fields=ALL_FIELDS,
                start=start,
                limit=limit
            )
            try:
                results = base.query(sql_query)
            except Exception as e:
                 logging.error(f"Error querying Seatable: {e}")
                 # Decide how to handle errors: break, retry, raise?
                 # For now, break the loop if an error occurs.
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
        if all_results:
            df = pd.DataFrame(all_results)
        else:
            # Return an empty DataFrame with expected columns if no results
            df = pd.DataFrame(columns=ALL_FIELDS)

        _CACHED_ANNOTATIONS = df # Cache the complete result

    # Apply proofread filter if requested (post-fetch)
    if proofread_only:
        if 'proofread' in df.columns:
            # Ensure boolean comparison works correctly, handle potential non-boolean values
            try:
                df = df[df['proofread'].astype(bool) == True]
            except Exception as e:
                 logging.warning(f"Could not apply 'proofread' filter due to data type issue: {e}")
        else:
            logging.warning("'proofread' column not found in annotations, cannot filter.")

    return df

