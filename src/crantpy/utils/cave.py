# -*- coding: utf-8 -*-
"""
This module provides utility functions for cave-related operations.
"""

import os
from caveclient import CAVEclient
from getpass import getpass
import logging

CRANT_CAVE_SERVER_URL = "https://proofreading.zetta.ai"
CRANT_DATASTACK = "kronauer_ant"

def get_current_cave_token():
    """
    Retrieves the current token from the CAVE client.
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

def set_cave_token(token):
    """
    Sets the CAVE token for the CAVE client.
    """
    assert isinstance(token, str), "Token must be a string."

    # Create a CAVE client instance
    client = CAVEclient(server_address=CRANT_CAVE_SERVER_URL)
    # Get the current authentication
    auth = client.auth
    # Save the token
    auth.save_token(token, overwrite=True)

def generate_cave_token(save=False):
    """
    Generates a token for the CAVE client.
    If save is True, the token will be saved (overwriting any existing token).
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

def get_cave_client():
    """
    Returns a CAVE client instance.
    If a token is already set, it will be used for authentication.
    Otherwise, a new token will be generated.
    """
    # Create a CAVE client instance
    client = CAVEclient(
        datastack_name=CRANT_DATASTACK,
        server_address=CRANT_CAVE_SERVER_URL,
    )
    # Check if a token is already set
    if client.auth.token:
        return client
    else:
        generate_cave_token(save=True)
        # Regenerate the client with the new token
        client = CAVEclient(
            datastack_name=CRANT_DATASTACK,
            server_address=CRANT_CAVE_SERVER_URL,
        )
        # check if the token is set
        if not client.auth.token:
            raise ValueError("No token found. Please generate a new token using generate_cave_token().")
        return client




    




