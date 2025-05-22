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
from importlib import reload

import navis
import cloudvolume as cv
# monkeypatch cloudvolume to use the navis version of the cloudvolume
navis.patch_cloudvolume()

import pytz
from caveclient import CAVEclient

from crantpy.utils.config import (CRANT_CAVE_DATASTACKS, CRANT_CAVE_SERVER_URL,
                                  CRANT_VALID_DATASETS, MAXIMUM_CACHE_DURATION)
from crantpy.utils.decorators import cached_result, inject_dataset


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

    if client.auth.token:
        # If a token is already set, return it
        return client.auth.token

    # If no token is set, try to find it in the cloudvolume secrets
    token = cv.secrets.cave_credentials(CRANT_CAVE_SERVER_URL).get('token', None)

    if token:
        return token

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

    # reload cloudvolume secrets
    # This is a workaround to ensure the token is saved in the cloudvolume secrets
    # file. The cloudvolume library does not provide a direct way to do this.

    reload(cv.secrets)
    reload(cv)

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

# Define a validation function for CAVE client cache
def validate_cave_client(client, *args, **kwargs):
    """Validate if a cached CAVE client is still valid."""
    try:
        current_time = pytz.UTC.localize(dt.datetime.utcnow())
        mds = client.materialize.get_versions_metadata()
        expired = [version for version in mds if version['expires_on'] <= current_time]
        return len(expired) == 0
    except Exception as e:
        logging.warning(f"Error validating CAVE client: {e}")
        return False

@inject_dataset(allowed=CRANT_VALID_DATASETS)
@cached_result(
    cache_name="cave_clients",
    key_fn=lambda *args, **kwargs: args[0] if args else kwargs['dataset'],
    validate_cache_fn=validate_cave_client
)
def get_cave_client(
    dataset: Optional[str] = None,
    clear_cache: bool = False,
    check_stale: bool = True,
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
        raise ValueError(f"Invalid dataset: {CRANT_CAVE_DATASTACKS[dataset]}. Available datastacks: {get_cave_datastacks()}")

    # Create a CAVE client instance
    client = CAVEclient(
        datastack_name=CRANT_CAVE_DATASTACKS[dataset],
        server_address=CRANT_CAVE_SERVER_URL,
    )

    # Check if a token is already set
    if not client.auth.token:
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
    
    return client

def clear_cave_client_cache() -> None:
    """Clears the CAVE client cache."""
    get_cave_client.clear_cache()

@inject_dataset(allowed=CRANT_VALID_DATASETS)
@cached_result(
    cache_name="cloudvolumes",
    key_fn=lambda *args, **kwargs: args[0] if args else kwargs['dataset'],
)
def get_cloudvolume(
    dataset: Optional[str] = None,
    clear_cache: bool = False,
    check_stale: bool = True,
    **kwargs
    ) -> CAVEclient:
    """
    Returns a cloudvolume instance.
    """
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
    vol.path = seg_source
    return vol

def clear_cloudvolume_cache() -> None:
    """Clears the cloudvolume cache."""
    get_cloudvolume.clear_cache()
