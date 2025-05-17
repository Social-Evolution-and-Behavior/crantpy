# -*- coding: utf-8 -*-
"""
This module provides utility functions for cave-related operations.
"""

import datetime as dt
import functools
import logging
import os
from getpass import getpass
from typing import Optional

import pytz
from caveclient import CAVEclient

from crantpy.utils.config import (CRANT_CAVE_DATASTACKS, CRANT_CAVE_SERVER_URL,
                                  CRANT_VALID_DATASETS, MAXIMUM_CACHE_DURATION,
                                  inject_dataset)

_CACHED_CAVE_CLIENTS = {}
_CACHED_CLOUDVOLUMES = {}

def get_current_cave_token() -> str:
    """
    Retrieves the current token from the CAVE client.
    
    Returns
    -------
    str
        The current CAVE token.
        
    Raises
    ------
    ValueError
        If no token is found.
    """
    # Create a CAVE client instance
    client = CAVEclient(server_address=CRANT_CAVE_SERVER_URL)
    # Get the current authentication 
    auth = client.auth 

    if auth.token:
        # If a token is already set, return it
        return auth.token
    else:
        raise ValueError("No token found. Please generate a new token using generate_cave_token().")

def set_cave_token(token: str) -> None:
    """
    Sets the CAVE token for the CAVE client.
    
    Parameters
    ----------
    token : str
        The CAVE token to set.
    """
    assert isinstance(token, str), "Token must be a string."

    # Create a CAVE client instance
    client = CAVEclient(server_address=CRANT_CAVE_SERVER_URL)
    # Get the current authentication
    auth = client.auth
    # Save the token
    auth.save_token(token, overwrite=True)

def generate_cave_token(save: bool = False) -> None:
    """
    Generates a token for the CAVE client.
    If save is True, the token will be saved (overwriting any existing token).
    
    Parameters
    ----------
    save : bool, default False
        Whether to save the token after generation.
    """
    # Create a CAVE client instance
    client = CAVEclient(server_address=CRANT_CAVE_SERVER_URL)
    # Get the current authentication
    auth = client.auth
    # Generate a new token
    auth.get_new_token(open=True)
    if save:
        token = getpass("Enter your CAVE token: ").strip()
        set_cave_token(token)
    else:
        logging.warning("Token generated but not saved. Use set_cave_token(<token>) to save it.")

@functools.lru_cache
def get_cave_datastacks() -> list:
    """Get available CAVE datastacks."""
    return CAVEclient(server_address=CRANT_CAVE_SERVER_URL).info.get_datastacks()

@functools.lru_cache
def get_datastack_segmentation_source(datastack) -> str:
    """Get segmentation source for given CAVE datastack."""
    return CAVEclient(server_address=CRANT_CAVE_SERVER_URL).info.get_datastack_info(datastack_name=datastack)['segmentation_source']

@inject_dataset(allowed=CRANT_VALID_DATASETS)
def get_cave_client(
    clear_cache: bool = False,
    check_stale: bool = True,
    dataset: Optional[str] = None,
    ) -> CAVEclient:
    """
    Returns a CAVE client instance.
    If a token is already set, it will be used for authentication.
    Otherwise, a new token will be generated.
    
    Parameters
    ----------
    clear_cache : bool, default False
        If True, bypasses the cache and fetches a new client.
    check_stale : bool, default True
        If True, checks if the cached client is stale based on materialization and maximum cache duration.
    dataset : str, optional
        The dataset to use. If not provided, uses the default dataset.
        
    Returns
    -------
    CAVEclient
        A CAVE client instance authenticated with the token.
        
    Raises
    ------
    ValueError
        If no token is found after attempting to generate one.
    """
    # verify the dataset
    if CRANT_CAVE_DATASTACKS[dataset] not in get_cave_datastacks():
        raise ValueError(f"Invalid dataset: {dataset}")

    global _CACHED_CAVE_CLIENTS
    # Check if the client is already cached
    if not clear_cache and _CACHED_CAVE_CLIENTS.get(dataset) is not None and hasattr(_CACHED_CAVE_CLIENTS[dataset], '_created_at'):
        
        # Check if the cached client is still valid
        if check_stale:

            current_time = pytz.UTC.localize(dt.datetime.utcnow())

            # Get the expired materialization versions
            client = _CACHED_CAVE_CLIENTS[dataset]
            mds = client.materialize.get_versions_metadata()
            expired = [version for version in mds if version['expires_on'] <= current_time]

            # Get the elapsed time since the client was created
            elapsed_time = (current_time - client._created_at).total_seconds()

            # The cache is valid if the elapsed time is less than the maximum cache duration and there are no expired versions
            if elapsed_time < MAXIMUM_CACHE_DURATION and not expired:
                # Cache is still valid
                logging.info("Using cached CAVE client.")
                return client
            else:
                logging.info("Cached CAVE client is stale.")
                # Cache is stale, remove it
                del _CACHED_CAVE_CLIENTS[dataset]
        
        else:
        
            logging.info("Using cached CAVE client.")
            return _CACHED_CAVE_CLIENTS[dataset]

    logging.info("Fetching new CAVE client...")

    # Create a CAVE client instance
    client = CAVEclient(
        datastack_name=CRANT_CAVE_DATASTACKS[dataset],
        server_address=CRANT_CAVE_SERVER_URL,
    )

    # Check if a token is already set
    if client.auth.token:
        # Cache the client
        _CACHED_CAVE_CLIENTS[dataset] = client
        # Set the created_at attribute to the current time
        _CACHED_CAVE_CLIENTS[dataset]._created_at = pytz.UTC.localize(dt.datetime.utcnow())
        return _CACHED_CAVE_CLIENTS[dataset]
    else:
        # TODO: get token from cloudvolume?
        generate_cave_token(save=True)
        # Regenerate the client with the new token
        client = CAVEclient(
            datastack_name=CRANT_CAVE_DATASTACKS[dataset],
            server_address=CRANT_CAVE_SERVER_URL,
        )
        # check if the token is set
        if not client.auth.token:
            raise ValueError("No token found. Please generate a new token using generate_cave_token().")
        
        # Cache the client
        _CACHED_CAVE_CLIENTS[dataset] = client
        # Set the created_at attribute to the current time
        _CACHED_CAVE_CLIENTS[dataset]._created_at = pytz.UTC.localize(dt.datetime.utcnow())
        return _CACHED_CAVE_CLIENTS[dataset]

def clear_cave_client_cache() -> None:
    """
    Clears the CAVE client cache.
    """
    global _CACHED_CAVE_CLIENTS
    _CACHED_CAVE_CLIENTS = {}
    logging.info("CAVE client cache cleared.")


@inject_dataset(allowed=CRANT_VALID_DATASETS)
def get_cloudvolume(
    clear_cache: bool = False,
    check_stale: bool = True,
    dataset: Optional[str] = None,
    **kwargs
    ) -> CAVEclient:
    """
    Returns a cloudvolume instance.
    """
    global _CACHED_CLOUDVOLUMES
    # Check if the client is already cached
    if not clear_cache and _CACHED_CLOUDVOLUMES.get(dataset) is not None and hasattr(_CACHED_CLOUDVOLUMES[dataset], '_created_at'):
        
        # Check if the cached client is still valid
        if check_stale:

            current_time = pytz.UTC.localize(dt.datetime.utcnow())

            # Get the elapsed time since the client was created
            elapsed_time = (current_time - _CACHED_CLOUDVOLUMES[dataset]._created_at).total_seconds()

            # The cache is valid if the elapsed time is less than the maximum cache duration and there are no expired versions
            if elapsed_time < MAXIMUM_CACHE_DURATION and not expired:
                # Cache is still valid
                logging.info("Using cached cloudvolume.")
                return _CACHED_CLOUDVOLUMES[dataset]
            else:
                logging.info("Cached cloudvolume is stale.")
                # Cache is stale, remove it
                del _CACHED_CLOUDVOLUMES[dataset]
        else:
            logging.info("Using cached cloudvolume.")
            return _CACHED_CLOUDVOLUMES[dataset]

    logging.info("Fetching new cloudvolume...")

    defaults = dict(mip=0,
                    fill_missing=True,
                    cache=False,
                    use_https=True,  # this way google secret is not needed
                    progress=False)
    defaults.update(kwargs)

    # Create a CAVE client instance
    client = get_cave_client(dataset=dataset)
    seg_source = client.info.segmentation_source()

    # Check if the segmentation source is valid
    if not seg_source:
        raise ValueError("Invalid segmentation source.")

    # get the cloudvolume
    vol = cv.CloudVolume(seg_source, **defaults)

    # Cache the volume
    _CACHED_CLOUDVOLUMES[dataset] = vol
    # Set the created_at attribute to the current time
    _CACHED_CLOUDVOLUMES[dataset]._created_at = pytz.UTC.localize(dt.datetime.utcnow())
    # Set the path attribute to the segmentation source
    _CACHED_CLOUDVOLUMES[dataset].path = seg_source
    return _CACHED_CLOUDVOLUMES[dataset]

def clear_cloudvolume_cache() -> None:
    """
    Clears the cloudvolume cache.
    """
    global _CACHED_CLOUDVOLUMES
    _CACHED_CLOUDVOLUMES = {}
    logging.info("Cloudvolume cache cleared.")
