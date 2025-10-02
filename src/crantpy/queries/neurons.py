# -*- coding: utf-8 -*-
"""NeuronCriteria class for filtering neurons based on Seatable annotations.
This class allows filtering neurons based on various criteria defined in the
Seatable annotations table. Filtering for multiple criteria is done using a
logical AND, i.e. only neurons that match all criteria will be selected.
Not providing any criteria will return all neurons in the annotations table.
This class is part of the CRANTB project and is designed to work with
Seatable's API to fetch and filter neuron data.
"""

import functools
import logging
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from crantpy.utils.config import (
    ALL_ANNOTATION_FIELDS,
    CRANT_VALID_DATASETS,
    SEARCH_EXCLUDED_ANNOTATION_FIELDS,
)
from crantpy.utils.decorators import inject_dataset, parse_neuroncriteria
from crantpy.utils.exceptions import FilteringError, NoMatchesError
from crantpy.utils.seatable import get_all_seatable_annotations, get_proofread_neurons
from crantpy.utils.helpers import filter_df
from crantpy.utils.cave.helpers import parse_root_ids
from crantpy.utils.types import Neurons


class NeuronCriteria:
    """Parses filter queries into root IDs using Seatable.

    This class allows filtering neurons based on various criteria defined in the
    Seatable annotations table.

    Filtering for multiple criteria is done using a logical AND, i.e.
    only neurons that match all criteria will be selected.

    Not providing any criteria will return all neurons in the annotations table.

    Parameters
    ----------
    regex : bool, default False
        Whether to interpret string criteria as regex.
    case : bool, default False
        Whether to interpret string criteria as case sensitive (only applies
        when `regex=True`).
    verbose : bool, default False
        Whether to print information about the query process.
    clear_cache : bool, default False
        Whether to force reloading annotations from Seatable, bypassing the cache.
    match_all : bool, default False
        When filtering by a list of values on a column that contains lists
        (e.g., `status=['DAMAGED', 'TRACING_ISSUE']`), setting `match_all=True`
        will only return neurons where the column contains *all* the specified values.
        If `False` (default), it returns neurons where the column contains *any*
        of the specified values.
    exact : bool, default True
        Whether to match values exactly. If `False`, substring matching is enabled.
    dataset : str, optional
        The dataset to fetch annotations from.
    **criteria : dict
        Filtering criteria where keys are column names from `NeuronCriteria.available_fields()`
        and values are the desired values or lists of values to filter by.

    Examples
    --------
    >>> from crantbseg.crantb.annotations import NeuronCriteria as NC

    >>> # Check available fields
    >>> NC.available_fields()
    ['root_id', 'nucleus_id', ...]

    >>> # Get all neurons (will print a warning)
    >>> all_neurons = NC()

    >>> # Get neurons by cell class
    >>> ol_neurons = NC(cell_class='olfactory_projection_neuron')
    >>> # Get neurons by multiple cell types using regex
    >>> pns = NC(cell_type=['PN_.*', 'mPN_.*'], regex=True)

    >>> # Get neurons by side and status (any status in the list)
    >>> left_proofread = NC(side='L', status=['BACKBONE_PROOFREAD', 'PRELIM_PROOFREAD'])

    >>> # Get neurons by side and status (must have BOTH statuses)
    >>> left_damaged_and_tracing = NC(side='L', status=['DAMAGED', 'TRACING_ISSUE'], match_all=True)

    >>> # Use in functions expecting root IDs
    >>> neuron_data = some_function(NC(cell_class='picky_neuron'))
    """

    @inject_dataset(allowed=CRANT_VALID_DATASETS)
    def __init__(
        self,
        *,
        regex: bool = False,
        case: bool = False,
        verbose: bool = False,
        clear_cache: bool = False,
        match_all: bool = False,
        exact: bool = True,
        dataset: Optional[str] = None,
        **criteria: Union[str, int, List[Union[str, int]]],
    ) -> None:
        # If no criteria make sure this is intended
        if not len(criteria):
            logging.warning("No criteria specified. This will query all neurons!")

        # Make sure all criteria are valid fields
        valid_fields = self.available_fields()
        for field in criteria:
            if field not in valid_fields:
                # Basic suggestion without fuzzy matching for now
                suggestions = [f for f in valid_fields if field.lower() in f.lower()]
                suggestion_str = (
                    f" Did you mean one of: {suggestions}?" if suggestions else ""
                )
                raise ValueError(
                    f'"{field}" is not a searchable field.'
                    f" Available fields are: {valid_fields}.{suggestion_str}"
                )
        self.dataset = dataset
        self.criteria: Dict[str, Union[str, int, List[Union[str, int]]]] = criteria
        self.regex = regex
        self.case = case
        self.verbose = verbose
        self.clear_cache = clear_cache
        self.match_all = match_all  # Store match_all
        self.exact = exact  # Store exact
        self._annotations: Optional[pd.DataFrame] = None
        self._roots: Optional[NDArray] = None

    def __iter__(self) -> Iterator[Union[int, str]]:
        """Allows iterating over the root IDs matched by this criteria.

        Returns
        -------
        Iterator
            Iterator over the root IDs matched by this criteria.
        """
        if self._roots is None:
            self._roots = self.get_roots()
        return iter(self._roots)

    def __len__(self) -> int:
        """Returns the number of root IDs matched by this criteria.

        Returns
        -------
        int
            Number of root IDs matched by this criteria.
        """
        if self._roots is None:
            self._roots = self.get_roots()
        return len(self._roots)

    def __contains__(self, item: Union[int, str]) -> bool:
        """Checks if a root ID is matched by this criteria.

        Parameters
        ----------
        item : int or str
            Root ID to check.

        Returns
        -------
        bool
            True if the root ID is matched by this criteria, False otherwise.
        """
        if self._roots is None:
            self._roots = self.get_roots()
        # Ensure item is compared with the correct type (roots are likely int or str)
        try:
            if self._roots.dtype == np.int64 or self._roots.dtype == np.float64:
                item = int(item)
            elif self._roots.dtype == object:  # Often strings
                item = str(item)
        except (ValueError, TypeError):
            # If conversion fails, it's unlikely to be in the list
            return False
        return item in self._roots

    @classmethod
    def available_fields(cls) -> List[str]:
        """Return all available fields for selection.

        Returns
        -------
        List[str]
            List of field names that can be used for filtering.
        """
        # remove fields that are not searchable from the list
        searchable_fields = [
            f
            for f in ALL_ANNOTATION_FIELDS
            if f not in SEARCH_EXCLUDED_ANNOTATION_FIELDS
        ]
        # Return as a list
        return searchable_fields

    @property
    def annotations(self) -> pd.DataFrame:
        """Return annotations table (DataFrame), loading if necessary.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the annotations from Seatable.
        """
        if self._annotations is None:
            # Pass clear_cache to potentially bypass cache
            self._annotations = get_all_seatable_annotations(
                clear_cache=self.clear_cache, dataset=self.dataset, proofread_only=False
            )
        return self._annotations

    @property
    def is_empty(self) -> bool:
        """Returns True if no criteria are specified.

        Returns
        -------
        bool
            True if no criteria are specified, False otherwise.
        """
        return len(self.criteria) == 0

    def get_roots(self) -> NDArray:
        """Return all root IDs matching the given criteria.

        Returns
        -------
        numpy.ndarray
            Array of root IDs matching the given criteria.

        Raises
        ------
        NoMatchesError
            If no neurons are found matching the given criteria.
        """
        # Use the property to get annotations (handles loading/caching)
        ann = self.annotations.copy()

        # If no criteria at all, just return all unique root IDs
        if self.is_empty:
            roots = ann["root_id"].unique()
            if self.verbose:
                logging.warning(
                    f"Querying all {len(roots)} unique neurons based on 'root_id'!"
                )
            # Convert to numpy array for consistency
            return np.asarray(roots)

        # Apply our filters using filter_df
        for field, value in self.criteria.items():
            if not field in ann.columns:
                logging.warning(
                    f"Field '{field}' not found in the annotation table columns. Skipping filter."
                )
                continue

            # Use the imported filter_df function
            try:
                # Pass self.match_all and self.exact to filter_df
                ann = filter_df(
                    ann,
                    field,
                    value,
                    regex=self.regex,
                    case=self.case,
                    match_all=self.match_all,
                    exact=self.exact,
                )
            except ValueError as e:
                # Catch potential errors from filter_df (e.g., dtype mismatch)
                logging.error(
                    f"Error filtering field '{field}' with value '{value}': {e}"
                )
                # Raise a custom FilteringError with additional context
                raise FilteringError(
                    f"Error filtering field '{field}' with value '{value}': {e}"
                ) from e
            except KeyError:
                # This shouldn't happen due to the check above, but as a safeguard:
                logging.warning(
                    f"Field '{field}' caused a KeyError during filtering. Skipping filter."
                )
                continue

        roots = ann["root_id"].unique()

        if not len(roots):
            raise NoMatchesError("No neurons found matching the given criteria.")
        elif self.verbose:
            logging.info(
                f"Found {len(roots)} {'neurons' if len(roots) > 1 else 'neuron'} matching the given criteria."
            )

        # Remove nones and empty strings
        roots = [root for root in roots if root not in (None, "", "None", "nan", "NaN")]

        # Return as numpy array
        return np.asarray(roots)


# function to fetch annotations from Seatable
@parse_neuroncriteria()
@inject_dataset(allowed=CRANT_VALID_DATASETS)
def get_annotations(
    neurons: Union[int, str, List[Union[int, str]], "NeuronCriteria"],
    dataset: Optional[str] = None,
    clear_cache: bool = False,
    proofread_only: bool = False,
) -> pd.DataFrame:
    """Get annotations from Seatable.

    Parameters
    ----------
    neurons : int, str, list of int/str, or NeuronCriteria
        Neuron root ID(s) or a NeuronCriteria instance specifying which neurons to fetch.
        Accepts a single ID, a list/array of IDs, or a NeuronCriteria object.
    dataset : str, optional
        Dataset to fetch annotations from.
    clear_cache : bool, default False
        Whether to force reloading annotations from Seatable, bypassing the cache.
    proofread_only : bool, default False
        Whether to return only annotations marked as proofread.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the annotations for the specified neurons.

    Raises
    ------
    ValueError
        If the input type is invalid.
    NoMatchesError
        If no matching neurons are found.
    """
    # Normalize input
    root_ids = parse_root_ids(neurons)

    # Convert to string for comparison to annotations
    root_ids = [str(root) for root in root_ids]

    # Fetch annotations from Seatable
    annotations = get_all_seatable_annotations(
        proofread_only=proofread_only, clear_cache=clear_cache, dataset=dataset
    )
    # Filter annotations based on the provided root IDs
    filtered_annotations = annotations[annotations["root_id"].isin(root_ids)]
    if filtered_annotations.empty:
        raise NoMatchesError("No matching neurons found for the provided criteria.")
    return filtered_annotations


@inject_dataset(allowed=CRANT_VALID_DATASETS)
@parse_neuroncriteria()
def is_proofread(
    neurons: Union[int, str, List[Union[int, str]], "NeuronCriteria"],
    dataset: Optional[str] = None,
    clear_cache: bool = False,
) -> np.ndarray:
    """Check if the specified neurons are proofread.

    Parameters
    ----------
    neurons : Neurons = Neurons = str | int | np.int64 | navis.BaseNeuron | Iterables of previous types | navis.NeuronList | NeuronCriteria
        Neurons to check for proofread status.
    dataset : str, optional
        Dataset to fetch annotations from.
    clear_cache : bool, default False
        Whether to force updating annotations from Seatable, bypassing the cache.

    Returns
    -------
    np.ndarray
        A boolean array indicating whether each neuron is proofread.
    """
    # Normalize input
    root_ids = parse_root_ids(neurons)

    # Convert to string for comparison to annotations
    root_ids = [str(root) for root in root_ids]

    # Fetch all proofread neurons
    proofread_neurons = get_proofread_neurons(dataset=dataset, clear_cache=clear_cache)

    if len(proofread_neurons) == 0:
        logging.warning("No proofread neurons found in the dataset.")
        return np.array([], dtype=bool)

    # Check if the neurons are in the proofread list
    is_proofread = np.isin(root_ids, proofread_neurons)

    return is_proofread


# Backward compatibility: Re-export parse_neuroncriteria with deprecation warning
def _deprecated_parse_neuroncriteria(*args, **kwargs):
    """Deprecated: parse_neuroncriteria has moved to crantpy.utils.decorators.

    This function will be removed in a future version. Please update your imports:

    OLD: from crantpy.queries.neurons import parse_neuroncriteria
    NEW: from crantpy.utils.decorators import parse_neuroncriteria
    """
    warnings.warn(
        "parse_neuroncriteria has been moved from crantpy.queries.neurons to "
        "crantpy.utils.decorators. Please update your imports. This backward "
        "compatibility will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Import here to avoid circular imports
    from crantpy.utils.decorators import parse_neuroncriteria as _parse_neuroncriteria

    return _parse_neuroncriteria(*args, **kwargs)


# Re-export with the original name for backward compatibility
parse_neuroncriteria = _deprecated_parse_neuroncriteria
