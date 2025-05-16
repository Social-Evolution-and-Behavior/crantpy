# -*- coding: utf-8 -*-
"""
This module contains configuration settings for CRANTpy.
It includes the default dataset, CRANT data stacks, and Seatable server details.
It also provides a decorator to inject the current default dataset into functions.
"""
import functools
import os
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

F = TypeVar('F', bound=Callable[..., Any])

CRANT_VALID_DATASETS = ['latest', 'sandbox']
CRANT_DEFAULT_DATASET = os.environ.get('CRANT_DEFAULT_DATASET', 'latest')
if CRANT_DEFAULT_DATASET not in CRANT_VALID_DATASETS:
    raise ValueError(f"Invalid CRANT_DEFAULT_DATASET: {CRANT_DEFAULT_DATASET}. "
                     f"Accepted values are: {CRANT_VALID_DATASETS}")


CRANT_CAVE_SERVER_URL = "https://proofreading.zetta.ai"
CRANT_CAVE_DATASTACKS = {
    'latest': 'kronauer_ant',
    'sandbox': 'kronauer_ant_clone_x1',
}

CRANT_SEATABLE_SERVER_URL = "https://cloud.seatable.io/"
CRANT_SEATABLE_WORKSPACE_ID = "62919"
CRANT_SEATABLE_BASENAME = "CRANTb"

CRANT_SEATABLE_ANNOTATIONS_TABLES = {
    'latest': 'CRANTb_meta',
    'sandbox': 'CRANTb_meta',
}

# decorators to inject dataset and parse neuron criteria
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


