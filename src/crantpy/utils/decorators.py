# -*- coding: utf-8 -*-
"""
This module provides decorators for caching and other utilities.
"""

import datetime as dt
import functools
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast

import warnings
import requests
import time
import numpy as np
import pytz

from crantpy.utils.config import CRANT_DEFAULT_DATASET, MAXIMUM_CACHE_DURATION
from crantpy.utils.exceptions import NoMatchesError
from crantpy.utils.types import T, F

_GLOBAL_CACHES: Dict[str, Dict[Any, Any]] = {}


def get_global_cache(cache_name: str) -> Dict[Any, Any]:
    """Get a named global cache dictionary.

    Parameters
    ----------
    cache_name : str
        Name of the cache to retrieve

    Returns
    -------
    dict
        The requested cache dictionary
    """
    global _GLOBAL_CACHES
    if cache_name not in _GLOBAL_CACHES:
        _GLOBAL_CACHES[cache_name] = {}
    return _GLOBAL_CACHES[cache_name]


def clear_global_cache(cache_name: str) -> None:
    """Clear a named global cache.

    Parameters
    ----------
    cache_name : str
        Name of the cache to clear
    """
    global _GLOBAL_CACHES
    if cache_name in _GLOBAL_CACHES:
        _GLOBAL_CACHES[cache_name] = {}
        logging.info(f"{cache_name} cache cleared.")


def _generate_default_cache_key(func, *args, **kwargs):
    """Generate a default cache key for a function call."""
    sig = inspect.signature(func)
    bound_args = sig.bind_partial(*args, **kwargs)
    bound_args.apply_defaults()

    # Try to get dataset from bound arguments
    if "dataset" in bound_args.arguments:
        return bound_args.arguments["dataset"]
    elif args:
        return args[0]
    else:
        return "latest"


def cached_result(
    cache_name: str,
    max_age: int = MAXIMUM_CACHE_DURATION,
    key_fn: Callable[..., Any] = None,
    should_cache_fn: Callable[..., bool] = None,
    validate_cache_fn: Callable[..., bool] = None,
) -> Callable[[F], F]:
    """Decorator for caching function results.

    WARNING: This decorator is not thread-safe. It is recommended to use
    threading.Lock() to ensure thread safety when using this decorator
    in a multi-threaded environment.

    This decorator provides a flexible caching mechanism for function results.
    It supports custom cache keys, validation, and conditional caching, making
    it suitable for a variety of use cases.

    The cache stores entries in a dictionary structure:
    {
        'result': original_function_result,
        'metadata': {
            '_created_at': timestamp
        }
    }

    This approach avoids modifying the original result objects directly,
    ensuring compatibility with immutable types.

    Parameters
    ----------
    cache_name : str
        Name of the global cache to use. This is used to group cached results
        under a specific namespace.
    max_age : int, default MAXIMUM_CACHE_DURATION
        Maximum age of cached result in seconds. Cached results older than
        this duration are considered stale and will be refreshed.
    key_fn : callable, optional
        Function to generate a unique cache key based on the function's
        arguments. Defaults to using the first positional argument or the
        'dataset' keyword argument. If the function returns None, an error
        will be raised.
    should_cache_fn : callable, optional
        Function to determine whether the result of the function should be
        cached. It takes the function result and arguments as input and
        returns a boolean.
    validate_cache_fn : callable, optional
        Function to validate if a cached result is still valid beyond the
        age check. It takes the cached result and the function arguments as
        input and returns a boolean.

    Returns
    -------
    callable
        The decorated function with caching capabilities.

    Examples
    --------
    >>> # Basic Caching:
    >>> @cached_result(cache_name="example_cache")
    >>> def expensive_function(x):
    >>>     return x ** 2

    >>> # Custom Cache Key:
    >>> @cached_result(cache_name="example_cache", key_fn=lambda x: f"key_{x}")
    >>> def expensive_function(x):
    >>>     return x ** 2

    >>> # Conditional Caching:
    >>> @cached_result(cache_name="example_cache", should_cache_fn=lambda result, *args: result > 10)
    >>> def expensive_function(x):
    >>>     return x ** 2

    >>> # Cache Validation:
    >>> def validate_cache(result, *args):
    >>>     return result is not None

    >>> @cached_result(cache_name="example_cache", validate_cache_fn=validate_cache)
    >>> def expensive_function(x):
    >>>     return x ** 2

    Notes
    -----
    - The decorated function gains a `clear_cache` method to manually clear
      the cache for the specified `cache_name`.
    - The `check_stale` parameter can be used to skip staleness checks when
      calling the decorated function.
    """

    def outer(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get the cache dictionary
            cache = get_global_cache(cache_name)

            # Handle clear_cache parameter if present
            clear_cache = kwargs.get("clear_cache", False)
            check_stale = kwargs.get("check_stale", True)

            # Generate cache key
            if key_fn is not None:
                try:
                    cache_key = key_fn(*args, **kwargs)
                    logging.debug(f"Generated cache key: {cache_key} for {cache_name}")
                except (KeyError, IndexError):
                    logging.debug(
                        "Key function failed, falling back to default cache key generation."
                    )
                    # If key_fn fails, fall back to default behavior
                    cache_key = _generate_default_cache_key(func, *args, **kwargs)
            else:
                cache_key = _generate_default_cache_key(func, *args, **kwargs)

            if cache_key is None:
                raise ValueError("Cache key function returned None.")

            # Check for cached result
            if not clear_cache and cache_key in cache:
                logging.debug(f"Cache hit for {cache_name} with key: {cache_key}")
                cached_entry = cache[cache_key]

                # Extract the actual result and metadata
                cached_result = cached_entry["result"]
                metadata = cached_entry["metadata"]

                # Skip staleness checks if check_stale is False
                if not check_stale:
                    logging.debug(f"Using cached {cache_name}.")
                    return cached_result

                # Get current time for age check
                current_time = dt.datetime.now(dt.timezone.utc)
                elapsed_time = (current_time - metadata["_created_at"]).total_seconds()

                # Check if cache is valid based on age and custom validation
                age_valid = elapsed_time < max_age
                validation_valid = True

                # Run custom validation if provided
                if validate_cache_fn:
                    validation_valid = validate_cache_fn(cached_result, *args, **kwargs)

                # If cache is valid, return it
                if age_valid and validation_valid:
                    logging.debug(f"Using cached {cache_name}.")
                    return cached_result
                else:
                    logging.info(f"Cached {cache_name} is stale.")
                    # Remove stale cache entry safely to avoid key errors
                    cache.pop(cache_key, None)

            # Cache miss or forced refresh
            logging.debug(f"Fetching new {cache_name}...")
            result = func(*args, **kwargs)

            # Determine if result should be cached
            should_cache = True
            if should_cache_fn:
                should_cache = should_cache_fn(result, *args, **kwargs)

            # Cache the result if needed
            if should_cache:
                cache[cache_key] = {
                    "result": result,
                    "metadata": {"_created_at": dt.datetime.now(dt.timezone.utc)},
                }

            return result

        # Add clear_cache method to the wrapped function
        setattr(wrapper, "clear_cache", lambda: clear_global_cache(cache_name))

        return cast(F, wrapper)

    return outer


# decorator to inject dataset
def inject_dataset(
    allowed: Optional[Union[List[str], str]] = None,
    disallowed: Optional[Union[List[str], str]] = None,
    param_name: str = "dataset",
) -> Callable[[F], F]:
    """Inject current default dataset.

    Parameters
    ----------
    allowed : List[str] or str, optional
        List of allowed datasets or a single allowed dataset.
    disallowed : List[str] or str, optional
        List of disallowed datasets or a single disallowed dataset.
    param_name : str, default 'dataset'
        Name of the parameter to inject the dataset into.

    Returns
    -------
    Callable
        Decorator function that injects the dataset.
    """
    if isinstance(allowed, str):
        allowed = [allowed]
    if isinstance(disallowed, str):
        disallowed = [disallowed]

    def outer(func: F) -> F:
        @functools.wraps(func)
        def inner(*args: Any, **kwargs: Any) -> Any:
            # Get function signature
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            # Check if dataset parameter exists in function signature
            if param_name not in param_names:
                # If parameter doesn't exist, just call the function
                return func(*args, **kwargs)

            param_idx = param_names.index(param_name)

            # Check if dataset is already provided
            if param_name not in kwargs and len(args) <= param_idx:
                # Inject default dataset
                kwargs[param_name] = CRANT_DEFAULT_DATASET

            # Get the dataset value for validation
            if param_name in kwargs:
                ds = kwargs[param_name]
            elif len(args) > param_idx:
                ds = args[param_idx]
            else:
                ds = CRANT_DEFAULT_DATASET

            # Validate dataset
            if allowed and ds not in allowed:
                raise ValueError(
                    f'Dataset "{ds}" not allowed for function {func.__name__}. '
                    f"Accepted datasets: {allowed}"
                )
            if disallowed and ds in disallowed:
                raise ValueError(
                    f'Dataset "{ds}" not allowed for function {func.__name__}.'
                )

            return func(*args, **kwargs)

        return cast(F, inner)

    return outer


def parse_neuroncriteria(allow_empty: bool = False) -> Callable[[F], F]:
    """Parse all NeuronCriteria arguments into an array of root IDs.

    This decorator automatically converts any NeuronCriteria objects in
    function arguments to arrays of root IDs. It uses type checking by class
    name to avoid circular imports.

    Parameters
    ----------
    allow_empty : bool, default False
        Whether to allow the NeuronCriteria to not match any neurons.

    Returns
    -------
    Callable
        Decorator function that processes NeuronCriteria arguments.

    Examples
    --------
    >>> @parse_neuroncriteria()
    >>> def process_neurons(neurons):
    >>>     # neurons will be an array of root IDs
    >>>     return neurons
    >>>
    >>> # Can be called with a NeuronCriteria object
    >>> result = process_neurons(NeuronCriteria(cell_class='example'))
    """

    def outer(func: F) -> F:
        @functools.wraps(func)
        def inner(*args: Any, **kwargs: Any) -> Any:
            # Search through *args for NeuronCriteria
            for i, nc in enumerate(args):
                # Check by class name to avoid circular imports
                if nc.__class__.__name__ == "NeuronCriteria":
                    # First check if we're allowed to query all neurons
                    if nc.is_empty and not allow_empty:
                        raise ValueError(
                            "NeuronCriteria must contain filter conditions."
                        )
                    args = list(args)
                    args[i] = nc.get_roots()

            # Search through **kwargs for NeuronCriteria
            for key, nc in kwargs.items():
                # Check by class name to avoid circular imports
                if (
                    hasattr(nc, "__class__")
                    and nc.__class__.__name__ == "NeuronCriteria"
                ):
                    # First check if we're allowed to query all neurons
                    if nc.is_empty and not allow_empty:
                        raise ValueError(
                            "NeuronCriteria must contain filter conditions."
                        )
                    kwargs[key] = nc.get_roots()

            try:
                return func(*args, **kwargs)
            except NoMatchesError as e:
                if allow_empty:
                    return np.array([], dtype=np.int64)
                else:
                    raise e

        return cast(F, inner)

    return outer


def retry_func(retries=5, cooldown=2):
    """
    Retry decorator for functions on HTTPError.
    This also suppresses UserWarnings (commonly raised by l2 cache requests)
    Parameters
    ----------
    cooldown :  int | float
                Cooldown period in seconds between attempts.
    retries :   int
                Number of retries before we give up. Every subsequent retry
                will delay by an additional `retry`.
    """

    def outer(func: F) -> F:
        @functools.wraps(func)
        def inner(*args: Any, **kwargs: Any) -> Any:
            for i in range(1, retries + 1):
                logging.debug(f"Attempt {i} of {retries} for {func.__name__}")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        return func(*args, **kwargs)
                    except KeyboardInterrupt:
                        raise
                    except requests.RequestException:
                        if i >= retries:
                            raise
                    except BaseException:
                        raise
                    time.sleep(cooldown * i)

        return cast(F, inner)

    return outer
