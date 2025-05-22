# -*- coding: utf-8 -*-
"""
This module contains type definitions and utility functions for CRANTpy.
It includes type aliases for various data structures used in the project,
such as Neurons, Function and other placeholders.
"""
from typing import TypeVar, Union, Iterable, Optional, Dict, Any, Tuple, List, Callable

import numpy as np
import navis

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])
Neurons = Union[str, int, np.int64, navis.BaseNeuron, Iterable[Union[str, int, np.int64, navis.BaseNeuron]], navis.NeuronList, 'NeuronCriteria']