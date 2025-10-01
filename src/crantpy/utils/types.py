# -*- coding: utf-8 -*-
"""
This module contains type definitions and utility functions for CRANTpy.
It includes type aliases for various data structures used in the project,
such as Neurons, Function and other placeholders.
"""
from typing import TypeVar, Union, Iterable, Optional, Dict, Any, Tuple, List, Callable

import numpy as np
from datetime import datetime
import pandas as pd
import navis

# TypeVar is a generic type variable that can be used to specify a type
T = TypeVar('T')

# Function is a type alias for a callable function with any number of arguments and return type Any

F = TypeVar('F', bound=Callable[..., Any])

# IDs is a type alias for various int or iterable integer representations (e.g. string, int, np.int64, or iterable of these types)
IDs = Union[str, int, np.int64, Iterable[Union[str, int, np.int64]]]

# Neurons is a type alias for various neuron representations (e.g., string, int, navis.BaseNeuron, etc.)
# Neurons = Union[str, int, np.int64, navis.BaseNeuron, Iterable[Union[str, int, np.int64, navis.BaseNeuron]], navis.NeuronList, 'NeuronCriteria']
Neurons = Union[IDs, navis.BaseNeuron, Iterable[navis.BaseNeuron], navis.NeuronList, 'NeuronCriteria']

# Timestamp is a type alias for a datetime-like object
Timestamp = Union[str, int, np.int64, datetime, np.datetime64, pd.Timestamp]