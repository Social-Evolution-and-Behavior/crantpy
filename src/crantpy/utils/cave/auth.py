# -*- coding: utf-8 -*-
"""
This module provides utility functions for cave-related authentication and authorization.
"""

import logging
from getpass import getpass
from importlib import reload

from caveclient import CAVEclient

import navis
import cloudvolume as cv

from crantpy.utils.config import CRANT_CAVE_SERVER_URL


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
