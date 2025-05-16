# -*- coding: utf-8 -*-
"""
This module provides utility functions for cave-related operations.
"""

import logging
import os
from getpass import getpass
from typing import Optional

from caveclient import CAVEclient

from crantpy.utils.config import (CRANT_CAVE_DATASTACKS, CRANT_CAVE_SERVER_URL,
                                  CRANT_VALID_DATASETS, inject_dataset)


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

@inject_dataset(allowed=CRANT_VALID_DATASETS)
def get_cave_client(dataset: Optional[str] = None) -> CAVEclient:
    """
    Returns a CAVE client instance.
    If a token is already set, it will be used for authentication.
    Otherwise, a new token will be generated.
    
    Parameters
    ----------
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
    # Create a CAVE client instance
    client = CAVEclient(
        datastack_name=CRANT_CAVE_DATASTACKS[dataset],
        server_address=CRANT_CAVE_SERVER_URL,
    )
    # Check if a token is already set
    if client.auth.token:
        return client
    else:
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




    




