# -*- coding: utf-8 -*-
"""NeuronCriteria class for filtering neurons based on Seatable annotations.
This class allows filtering neurons based on various criteria defined in the
Seatable annotations table. Filtering for multiple criteria is done using a
logical AND, i.e. only neurons that match all criteria will be selected.
Not providing any criteria will return all neurons in the annotations table.
This class is part of the CRANTB project and is designed to work with
Seatable's API to fetch and filter neuron data.
"""

import logging
import numpy as np
from crantpy.utils.seatable import (
    get_seatable_annotations,
    ALL_FIELDS,
    SEARCH_EXCLUDED_FIELDS,
)
from crantpy.utils.utils import filter_df
from crantpy.utils.exceptions import NoMatchesError

# function to fetch annotations from Seatable
def get_annotations(neurons, clear_cache=False):
    """Get annotations from Seatable.

    Parameters
    ----------
    neurons : int, str, list or NeuronCriteria
        Neurons to fetch annotations for. Can be a single root ID, a list of root IDs,
        or an instance of NeuronCriteria.
    clear_cache : bool, default False
        Whether to force reloading annotations from Seatable, bypassing the cache.
    """
    if isinstance(neurons, NeuronCriteria):
        # If neurons is a NeuronCriteria instance, fetch its roots
        neurons = neurons.get_roots()
    elif isinstance(neurons, (int, str)):
        # If integer convert to string
        if isinstance(neurons, int):
            neurons = str(neurons)
        # If a single root ID, convert to list
        neurons = [neurons]
    elif not isinstance(neurons, list):
        raise ValueError("Invalid input type. Must be int, str, or list of root IDs.")

    # Fetch annotations from Seatable
    annotations = get_seatable_annotations(clear_cache=clear_cache)
    # Filter annotations based on the provided root IDs
    filtered_annotations = annotations[annotations['root_id'].isin(neurons)]
    if filtered_annotations.empty:
        raise NoMatchesError("No matching neurons found for the provided criteria.")
    return filtered_annotations

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
    **criteria : dict
        Filtering criteria where keys are column names from `NeuronCriteria.available_fields()`
        and values are the desired values or lists of values to filter by.

    Examples
    --------
    >>> from crantbseg.crantb.annotations import NeuronCriteria as NC

    # Check available fields
    >>> NC.available_fields()
    ['root_id', 'nucleus_id', ...]

    # Get all neurons (will print a warning)
    >>> all_neurons = NC()

    # Get neurons by cell class
    >>> ol_neurons = NC(cell_class='olfactory_projection_neuron')

    # Get neurons by multiple cell types using regex
    >>> pns = NC(cell_type=['PN_.*', 'mPN_.*'], regex=True)

    # Get neurons by side and status (any status in the list)
    >>> left_proofread = NC(side='L', status=['BACKBONE_PROOFREAD', 'PRELIM_PROOFREAD'])

    # Get neurons by side and status (must have BOTH statuses)
    >>> left_damaged_and_tracing = NC(side='L', status=['DAMAGED', 'TRACING_ISSUE'], match_all=True)

    # Use in functions expecting root IDs
    # >>> neuron_data = some_function(NC(cell_class='picky_neuron'))
    """

    def __init__(self, *, regex=False, case=False, verbose=False, clear_cache=False, match_all=False, exact=True, **criteria):
        # If no criteria make sure this is intended
        if not len(criteria):
            logging.warning(
                "No criteria specified. This will query all neurons!"
            )

        # Make sure all criteria are valid fields
        valid_fields = self.available_fields()
        for field in criteria:
            if field not in valid_fields:
                # Basic suggestion without fuzzy matching for now
                suggestions = [f for f in valid_fields if field.lower() in f.lower()]
                suggestion_str = f" Did you mean one of: {suggestions}?" if suggestions else ""
                raise ValueError(
                    f'"{field}" is not a searchable field.'
                    f' Available fields are: {valid_fields}.{suggestion_str}'
                )

        self.criteria = criteria
        self.regex = regex
        self.case = case
        self.verbose = verbose
        self.clear_cache = clear_cache
        self.match_all = match_all # Store match_all
        self.exact = exact # Store exact
        self._annotations = None
        self._roots = None

    def __iter__(self):
        if self._roots is None:
            self._roots = self.get_roots()
        return iter(self._roots)

    def __len__(self):
        if self._roots is None:
            self._roots = self.get_roots()
        return len(self._roots)

    def __contains__(self, item):
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
    def available_fields(cls):
        """Return all available fields for selection."""
        # remove fields that are not searchable from the list
        searchable_fields = [f for f in ALL_FIELDS if f not in SEARCH_EXCLUDED_FIELDS]
        # Return as a list
        return searchable_fields

    @property
    def annotations(self):
        """Return annotations table (DataFrame), loading if necessary."""
        if self._annotations is None:
            # Pass clear_cache to potentially bypass cache
            self._annotations = get_seatable_annotations(clear_cache=self.clear_cache)
        return self._annotations

    @property
    def is_empty(self):
        """Returns True if no criteria are specified."""
        return len(self.criteria) == 0

    def get_roots(self):
        """Return all root IDs matching the given criteria."""
        # Use the property to get annotations (handles loading/caching)
        ann = self.annotations.copy()

        # If no criteria at all, just return all unique root IDs
        if self.is_empty:
            roots = ann['root_id'].unique()
            if self.verbose:
                logging.warning(f"Querying all {len(roots)} unique neurons based on 'root_id'!")
            # Convert to numpy array for consistency
            return np.asarray(roots)

        # Apply our filters using filter_df
        for field, value in self.criteria.items():
            if not field in ann.columns:
                logging.warning(f"Field '{field}' not found in the annotation table columns. Skipping filter.")
                continue

            # Use the imported filter_df function
            try:
                # Pass self.match_all and self.exact to filter_df
                ann = filter_df(ann, field, value, regex=self.regex, case=self.case, match_all=self.match_all, exact=self.exact)
            except ValueError as e:
                 # Catch potential errors from filter_df (e.g., dtype mismatch)
                 logging.error(f"Error filtering field '{field}' with value '{value}': {e}")
                 # Depending on desired behavior, you might want to raise the error,
                 # skip this filter, or return an empty result. Here, we'll skip.
                 continue
            except KeyError:
                 # This shouldn't happen due to the check above, but as a safeguard:
                 logging.warning(f"Field '{field}' caused a KeyError during filtering. Skipping filter.")
                 continue


        roots = ann['root_id'].unique()

        if not len(roots):
            raise NoMatchesError("No neurons found matching the given criteria.")
        elif self.verbose:
            logging.info(
                f"Found {len(roots)} {'neurons' if len(roots) > 1 else 'neuron'} matching the given criteria."
            )

        # Return as numpy array
        return np.asarray(roots)