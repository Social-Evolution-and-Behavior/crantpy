# -*- coding: utf-8 -*-
"""
This module provides utility functions for cave-related loading and caching.
"""

import functools
import logging
import datetime as dt
from typing import Optional
import pytz

from caveclient import CAVEclient

import navis
import cloudvolume as cv

from crantpy.utils.config import (
    CRANT_CAVE_SERVER_URL,
    CRANT_CAVE_DATASTACKS,
    CRANT_VALID_DATASETS,
)
from crantpy.utils.decorators import (
    inject_dataset,
    cached_result,
)
from crantpy.utils.cave.auth import generate_cave_token


@functools.lru_cache
def get_cave_datastacks() -> list:
    """Get available CAVE datastacks."""
    return CAVEclient(server_address=CRANT_CAVE_SERVER_URL).info.get_datastacks()


@functools.lru_cache
def get_datastack_segmentation_source(datastack) -> str:
    """Get segmentation source for given CAVE datastack."""
    # ensure the datastack is valid
    if datastack not in get_cave_datastacks():
        raise ValueError(
            f"Invalid datastack: {datastack}. Available datastacks: {get_cave_datastacks()}"
        )
    return CAVEclient(server_address=CRANT_CAVE_SERVER_URL).info.get_datastack_info(
        datastack_name=datastack
    )["segmentation_source"]


@functools.lru_cache
@inject_dataset(allowed=CRANT_VALID_DATASETS)
def get_dataset_segmentation_source(dataset: str) -> str:
    """Get segmentation source for given dataset."""
    return get_datastack_segmentation_source(CRANT_CAVE_DATASTACKS[dataset])


# Define a validation function for CAVE client cache
def validate_cave_client(client, *args, **kwargs):
    """Validate if a cached CAVE client is still valid."""
    try:
        current_time = pytz.UTC.localize(dt.datetime.utcnow())
        mds = client.materialize.get_versions_metadata()
        unexpired = [version for version in mds if version["expires_on"] > current_time]
        return len(unexpired) > 0
    except Exception as e:
        logging.warning(f"Error validating CAVE client: {e}")
        return False


@inject_dataset(allowed=CRANT_VALID_DATASETS)
@cached_result(
    cache_name="cave_clients",
    key_fn=lambda *args, **kwargs: args[0] if args else kwargs["dataset"],
    validate_cache_fn=validate_cave_client,
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
        raise ValueError(
            f"Invalid dataset: {CRANT_CAVE_DATASTACKS[dataset]}. Available datastacks: {get_cave_datastacks()}"
        )

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
            raise ValueError(
                "No token found. Please generate a new token using generate_cave_token()."
            )

    return client


def clear_cave_client_cache() -> None:
    """Clears the CAVE client cache."""
    get_cave_client.clear_cache()


@inject_dataset(allowed=CRANT_VALID_DATASETS)
@cached_result(
    cache_name="cloudvolumes",
    key_fn=lambda *args, **kwargs: args[0] if args else kwargs["dataset"],
)
def get_cloudvolume(
    dataset: Optional[str] = None,
    clear_cache: bool = False,
    check_stale: bool = True,
    **kwargs,
) -> CAVEclient:
    """
    Returns a cloudvolume instance.
    """
    defaults = dict(
        mip=0,
        fill_missing=True,
        cache=False,
        use_https=True,  # this way google secret is not needed
        progress=False,
    )
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


def clear_all_caches() -> None:
    """Clears all caches."""
    clear_cave_client_cache()
    clear_cloudvolume_cache()
