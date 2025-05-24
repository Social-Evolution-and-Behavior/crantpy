# -*- coding: utf-8 -*-
"""
This module provides utility functions for cave-related helpers.
"""
import numpy as np
from typing import Iterable
import navis

from crantpy.utils.decorators import parse_neuroncriteria
from crantpy.utils.helpers import make_iterable
from crantpy.utils.types import Neurons, IDs, Timestamp


@parse_neuroncriteria()
def parse_root_ids(x: Neurons) -> np.ndarray:
    """
    Parse root IDs from various input formats to a list of np.int64.

    Parameters
    ----------
    x : Neurons = str | int | np.int64 | navis.BaseNeuron | Iterables of previous types | navis.NeuronList | NeuronCriteria
        The input to parse. Can be a single ID, a list of IDs, or a navis neuron object.

    Returns
    -------
    np.ndarray
        A numpy array of root IDs as np.int64.
    """
    print(f"parse_root_ids: {x} {type(x)}")
    # process NeuronList
    if isinstance(x, navis.NeuronList):
        x = x.id

    # process neuron objects (single)
    if isinstance(x, navis.BaseNeuron):
        x = x.id
    elif isinstance(x, Iterable):
        # process neuron objects (iterable)
        x = [i.id if isinstance(i, navis.BaseNeuron) else i for i in x]

    # make iterable
    x = make_iterable(x, force_type=np.int64)

    return x.astype(np.int64) if len(x) > 0 else np.array([], dtype=np.int64)

def parse_timestamp(x: str) -> str:
    """
    Parse a timestamp string to a specific format.

    Parameters
    ----------
    x : str
        The input timestamp string.

    Returns
    -------
    str
        The parsed timestamp string in the format 'YYYY-MM-DD HH:MM:SS'.
    """
    return x.split('.')[0] if '.' in x else x