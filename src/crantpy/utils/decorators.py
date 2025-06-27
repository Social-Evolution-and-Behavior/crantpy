# -*- coding: utf-8 -*-
"""
This module provides decorators for caching and other utilities.
"""

import datetime as dt
import functools
import logging
from typing import (Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union,
                    cast)

import numpy as np
import pytz

from crantpy.utils.config import CRANT_DEFAULT_DATASET, MAXIMUM_CACHE_DURATION
from crantpy.utils.exceptions import NoMatchesError

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

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

def cached_result(
    cache_name: str,
    max_age: int = MAXIMUM_CACHE_DURATION,
    key_fn: Callable[..., Any] = None,
    should_cache_fn: Callable[..., bool] = None,
    validate_cache_fn: Callable[..., bool] = None
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

    Usage Examples
    --------------
    Basic Caching:
        @cached_result(cache_name="example_cache")
        def expensive_function(x):
            return x ** 2

    Custom Cache Key:
        @cached_result(cache_name="example_cache", key_fn=lambda x: f"key_{x}")
        def expensive_function(x):
            return x ** 2

    Conditional Caching:
        @cached_result(cache_name="example_cache", should_cache_fn=lambda result, *args: result > 10)
        def expensive_function(x):
            return x ** 2

    Cache Validation:
        def validate_cache(result, *args):
            return result is not None

        @cached_result(cache_name="example_cache", validate_cache_fn=validate_cache)
        def expensive_function(x):
            return x ** 2

    Notes
    -----
    - The decorated function gains a `clear_cache` method to manually clear
      the cache for the specified `cache_name`.
    - The `check_stale` parameter can be used to skip staleness checks when
      calling the decorated function.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get the cache dictionary
            cache = get_global_cache(cache_name)
            
            # Handle clear_cache parameter if present
            clear_cache = kwargs.get('clear_cache', False)
            check_stale = kwargs.get('check_stale', True)
            
            # Generate cache key
            if key_fn is not None:
                cache_key = key_fn(*args, **kwargs)
            else:
                # Default: use first arg or dataset kwarg
                cache_key = args[0] if args else kwargs['dataset']
            
            if cache_key is None:
                raise ValueError("Cache key function returned None.")
            
            # Check for cached result
            if not clear_cache and cache_key in cache:
                cached_entry = cache[cache_key]
                
                # Extract the actual result and metadata
                cached_result = cached_entry['result']
                metadata = cached_entry['metadata']
                
                # Skip staleness checks if check_stale is False
                if not check_stale:
                    logging.info(f"Using cached {cache_name}.")
                    return cached_result
                
                # Get current time for age check
                current_time = pytz.UTC.localize(dt.datetime.utcnow())
                elapsed_time = (current_time - metadata['_created_at']).total_seconds()
                
                # Check if cache is valid based on age and custom validation
                age_valid = elapsed_time < max_age
                validation_valid = True
                
                # Run custom validation if provided
                if validate_cache_fn:
                    validation_valid = validate_cache_fn(cached_result, *args, **kwargs)
                
                # If cache is valid, return it
                if age_valid and validation_valid:
                    logging.info(f"Using cached {cache_name}.")
                    return cached_result
                else:
                    logging.info(f"Cached {cache_name} is stale.")
                    # Remove stale cache entry
                    cache.pop(cache_key, None) 
            
            # Cache miss or forced refresh
            logging.info(f"Fetching new {cache_name}...")
            result = func(*args, **kwargs)
            
            # Determine if result should be cached
            should_cache = True
            if should_cache_fn:
                should_cache = should_cache_fn(result, *args, **kwargs)
            
            # Cache the result if needed
            if should_cache:
                cache[cache_key] = {
                    'result': result,
                    'metadata': {
                        '_created_at': pytz.UTC.localize(dt.datetime.utcnow())
                    }
                }
                
            return result
        
        # Add clear_cache method to the wrapped function
        setattr(wrapper, 'clear_cache', lambda: clear_global_cache(cache_name))
        
        return cast(F, wrapper)
    
    return decorator


# decorator to inject dataset
def inject_dataset(allowed: Optional[Union[List[str], str]] = None,
                   disallowed: Optional[Union[List[str], str]] = None) -> Callable[[F], F]:
    """Inject current default dataset.
    
    Parameters
    ----------
    allowed : List[str] or str, optional
        List of allowed datasets or a single allowed dataset.
    disallowed : List[str] or str, optional
        List of disallowed datasets or a single disallowed dataset.
        
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
            if kwargs.get('dataset', None) is None:
                kwargs['dataset'] = CRANT_DEFAULT_DATASET

            ds = kwargs['dataset']
            if allowed and ds not in allowed:
                raise ValueError(f'Dataset "{ds}" not allowed for function {func}. '
                                 f'Accepted datasets: {allowed}')
            if disallowed and ds in disallowed:
                raise ValueError(f'Dataset "{ds}" not allowed for function {func}.')
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
                if nc.__class__.__name__ == 'NeuronCriteria':
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
                if hasattr(nc, '__class__') and nc.__class__.__name__ == 'NeuronCriteria':
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


