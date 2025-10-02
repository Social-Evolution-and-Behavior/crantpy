from typing import Any, Dict, List, Optional, Set, Tuple, Union
from unittest.mock import MagicMock, patch
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from crantpy.queries.neurons import NeuronCriteria, get_annotations
from crantpy.utils.decorators import parse_neuroncriteria
from crantpy.utils.exceptions import NoMatchesError


def dummy_get_all_seatable_annotations(
    clear_cache: bool = False,
    dataset: Optional[str] = None,
    proofread_only: bool = False,
) -> pd.DataFrame:
    """Mock function for get_all_seatable_annotations.

    This is a test helper function that mocks the behavior of the real
    get_all_seatable_annotations function. It returns a DataFrame with mock
    data instead of real data from Seatable. This allows us to test the
    filtering functionality without needing to connect to the actual Seatable API.

    Parameters
    ----------
    clear_cache : bool, default False
        Dummy parameter for compatibility.
    dataset : str, optional
        Dummy parameter for compatibility.
    proofread_only : bool, default False
        Dummy parameter for compatibility.

    Returns
    -------
    pd.DataFrame
        Mock annotation data with columns: 'root_id', 'cell_class', 'status', and 'side'.
    """
    return pd.DataFrame(
        {
            "root_id": ["1", "2", "3", "4"],  # Use strings to match real behavior
            "cell_class": ["foo", "bar", "foobar", "baz"],
            "status": [["A", "B"], ["B"], ["A"], ["C"]],
            "side": ["L", "R", "L", "R"],
        }
    )


@patch(
    "crantpy.queries.neurons.get_all_seatable_annotations",
    side_effect=dummy_get_all_seatable_annotations,
)
def test_neuroncriteria_exact(mock_ann: MagicMock) -> None:
    """Test exact matching in NeuronCriteria.

    This test verifies that NeuronCriteria correctly filters neurons
    using exact string matching on the 'cell_class' column. When exact=True
    (the default), only neurons with the exact cell_class value should be returned.
    """
    nc = NeuronCriteria(cell_class="foo")
    roots = nc.get_roots()
    assert set(roots) == {"1"}  # Only neuron '1' has cell_class exactly 'foo'


@patch(
    "crantpy.queries.neurons.get_all_seatable_annotations",
    side_effect=dummy_get_all_seatable_annotations,
)
def test_neuroncriteria_substring(mock_ann: MagicMock) -> None:
    """Test substring matching in NeuronCriteria.

    This test verifies that NeuronCriteria correctly finds neurons
    using substring matching on the 'cell_class' column when exact=False.
    It should match any cell_class that contains the substring 'foo'.
    """
    nc = NeuronCriteria(cell_class="foo", exact=False)
    roots = nc.get_roots()
    assert set(roots) == {"1", "3"}  # Both 'foo' and 'foobar' contain 'foo'


@patch(
    "crantpy.queries.neurons.get_all_seatable_annotations",
    side_effect=dummy_get_all_seatable_annotations,
)
def test_neuroncriteria_list_column_any(mock_ann: MagicMock) -> None:
    """Test list column filtering with any matching in NeuronCriteria.

    This test verifies that NeuronCriteria correctly filters neurons
    based on a column that contains lists (the 'status' column). When match_all=False
    (the default), neurons should be included if their status list contains
    the specified value 'A', regardless of what other values are in the list.
    """
    nc = NeuronCriteria(status="A")
    roots = nc.get_roots()
    assert set(roots) == {"1", "3"}  # Both neurons have 'A' in their status lists


@patch(
    "crantpy.queries.neurons.get_all_seatable_annotations",
    side_effect=dummy_get_all_seatable_annotations,
)
def test_neuroncriteria_list_column_match_all(mock_ann: MagicMock) -> None:
    """Test list column filtering with all matching in NeuronCriteria.

    This test verifies that NeuronCriteria correctly filters neurons
    based on a column that contains lists using match_all=True. In this mode,
    a neuron is only included if its status list contains ALL of the specified
    values ('A' AND 'B'), not just any of them.
    """
    nc = NeuronCriteria(status=["A", "B"], match_all=True)
    roots = nc.get_roots()
    assert set(roots) == {
        "1"
    }  # Only neuron '1' has both 'A' and 'B' in its status list


@patch(
    "crantpy.queries.neurons.get_all_seatable_annotations",
    side_effect=dummy_get_all_seatable_annotations,
)
def test_neuroncriteria_side_and_status(mock_ann: MagicMock) -> None:
    """Test multiple criteria filtering in NeuronCriteria.

    This test verifies that NeuronCriteria correctly applies multiple
    filtering criteria using a logical AND. Neurons should only be included
    if they match both criteria: side='L' AND status contains 'A'.
    This is important for complex queries with multiple conditions.
    """
    nc = NeuronCriteria(side="L", status="A")
    roots = nc.get_roots()
    assert set(roots) == {
        "1",
        "3",
    }  # Both neurons are on the left side and have status 'A'


@patch(
    "crantpy.queries.neurons.get_all_seatable_annotations",
    side_effect=dummy_get_all_seatable_annotations,
)
def test_neuroncriteria_no_criteria(mock_ann: MagicMock) -> None:
    """Test NeuronCriteria with no criteria specified.

    This test verifies that NeuronCriteria returns all neurons when
    no filtering criteria are specified. While this will show a warning,
    it should still function correctly and return all unique root IDs.
    """
    nc = NeuronCriteria()
    roots = nc.get_roots()
    assert set(roots) == {"1", "2", "3", "4"}  # All neurons should be returned


@patch(
    "crantpy.queries.neurons.get_all_seatable_annotations",
    side_effect=dummy_get_all_seatable_annotations,
)
def test_neuroncriteria_invalidfield(mock_ann: MagicMock) -> None:
    """Test NeuronCriteria with invalid field.

    This test verifies that NeuronCriteria raises a ValueError with
    an informative error message when an invalid field name is specified.
    This is important for user experience, helping users identify when
    they've used a field that doesn't exist in the annotations table.
    """
    with pytest.raises(ValueError) as excinfo:
        nc = NeuronCriteria(nonexistent_field="value")
    assert "not a searchable field" in str(excinfo.value)


@patch(
    "crantpy.queries.neurons.get_all_seatable_annotations",
    side_effect=dummy_get_all_seatable_annotations,
)
def test_neuroncriteria_iter(mock_ann: MagicMock) -> None:
    """Test iteration over NeuronCriteria.

    This test verifies that the NeuronCriteria class supports iteration,
    allowing users to iterate over the root IDs directly from a NeuronCriteria
    instance. This makes the class more pythonic and easier to use in for loops.
    """
    nc = NeuronCriteria(side="L")
    roots = [root for root in nc]
    assert set(roots) == {"1", "3"}  # Can iterate over the filtered root IDs


@patch(
    "crantpy.queries.neurons.get_all_seatable_annotations",
    side_effect=dummy_get_all_seatable_annotations,
)
def test_neuroncriteria_contains(mock_ann: MagicMock) -> None:
    """Test contains operator for NeuronCriteria.

    This test verifies that the NeuronCriteria class supports the 'in' operator,
    allowing users to check if a specific root ID is included in the filtered results.
    This provides a convenient way to test membership without explicitly getting all roots.
    """
    nc = NeuronCriteria(side="L")
    assert "1" in nc  # Root ID '1' should be in the filtered results
    assert "2" not in nc  # Root ID '2' should not be in the filtered results
    assert "3" in nc  # Root ID '3' should be in the filtered results
    assert "4" not in nc  # Root ID '4' should not be in the filtered results


@patch(
    "crantpy.queries.neurons.get_all_seatable_annotations",
    side_effect=dummy_get_all_seatable_annotations,
)
def test_neuroncriteria_len(mock_ann: MagicMock) -> None:
    """Test len operator for NeuronCriteria.

    This test verifies that the NeuronCriteria class supports the len() function,
    allowing users to count the number of neurons that match the criteria without
    explicitly getting all the roots. This is useful for checking how many neurons
    match a particular set of criteria.
    """
    nc = NeuronCriteria(side="L")
    assert len(nc) == 2  # There should be 2 neurons with side='L'


@patch(
    "crantpy.queries.neurons.get_all_seatable_annotations",
    side_effect=dummy_get_all_seatable_annotations,
)
def test_neuroncriteria_is_empty(mock_ann: MagicMock) -> None:
    """Test is_empty property for NeuronCriteria.

    This test verifies that the is_empty property correctly identifies
    whether any filtering criteria are specified. This is important for
    functions that need to know if a NeuronCriteria instance will filter
    anything or just return all neurons.
    """
    nc1 = NeuronCriteria()
    assert nc1.is_empty is True  # No criteria specified

    nc2 = NeuronCriteria(side="L")
    assert nc2.is_empty is False  # Criteria specified


@patch(
    "crantpy.queries.neurons.get_all_seatable_annotations",
    side_effect=dummy_get_all_seatable_annotations,
)
def test_neuroncriteria_available_fields(mock_ann: MagicMock) -> None:
    """Test available_fields classmethod for NeuronCriteria.

    This test verifies that the available_fields() classmethod returns
    a non-empty list of valid field names that can be used for filtering.
    This is crucial for users to discover what fields they can filter by.
    """
    fields = NeuronCriteria.available_fields()
    assert isinstance(fields, list)
    assert all(isinstance(field, str) for field in fields)
    # We don't check specific field names here since they might change,
    # but we ensure the method returns a list of strings


@patch(
    "crantpy.queries.neurons.get_all_seatable_annotations",
    side_effect=dummy_get_all_seatable_annotations,
)
def test_neuroncriteria_no_matches(mock_ann: MagicMock) -> None:
    """Test NeuronCriteria with no matches.

    This test verifies that NeuronCriteria raises a NoMatchesError
    when no neurons match the specified criteria. This is important for
    proper error handling in client code that needs to know when a query
    returns no results.
    """
    with pytest.raises(NoMatchesError):
        nc = NeuronCriteria(cell_class="nonexistent")
        nc.get_roots()


@patch(
    "crantpy.queries.neurons.get_all_seatable_annotations",
    side_effect=dummy_get_all_seatable_annotations,
)
def test_parse_neuroncriteria_decorator(mock_ann: MagicMock) -> None:
    """Test parse_neuroncriteria decorator.

    This test verifies that the parse_neuroncriteria decorator correctly
    processes NeuronCriteria objects passed to a function, converting them
    to arrays of root IDs. This allows functions to transparently accept
    either NeuronCriteria objects or direct root ID lists.
    """

    @parse_neuroncriteria()
    def dummy_function(neurons: Any) -> Any:
        return neurons

    # Test with NeuronCriteria
    nc = NeuronCriteria(cell_class="foo")
    result = dummy_function(nc)
    assert set(result) == {"1"}  # Decorator should convert NeuronCriteria to root IDs

    # Test with non-NeuronCriteria (should pass through unchanged)
    result = dummy_function(["1", "2", "3"])
    assert set(result) == {"1", "2", "3"}


@patch(
    "crantpy.queries.neurons.get_all_seatable_annotations",
    side_effect=dummy_get_all_seatable_annotations,
)
def test_parse_neuroncriteria_disallow_empty(mock_ann: MagicMock) -> None:
    """Test parse_neuroncriteria decorator with allow_empty=False.

    This test verifies that the parse_neuroncriteria decorator raises
    a ValueError when allow_empty=False and an empty NeuronCriteria
    (no filtering criteria) is passed. This prevents functions from
    inadvertently querying all neurons when a specific subset is expected.
    """

    @parse_neuroncriteria(allow_empty=False)
    def dummy_function(neurons: Any) -> Any:
        return neurons

    # Test with empty NeuronCriteria - should raise ValueError
    with pytest.raises(ValueError) as excinfo:
        nc = NeuronCriteria()
        dummy_function(nc)
    assert "NeuronCriteria must contain filter conditions" in str(excinfo.value)


@patch(
    "crantpy.queries.neurons.get_all_seatable_annotations",
    side_effect=dummy_get_all_seatable_annotations,
)
def test_parse_neuroncriteria_allow_empty(mock_ann: MagicMock) -> None:
    """Test parse_neuroncriteria decorator with allow_empty=True.

    This test verifies that the parse_neuroncriteria decorator correctly
    processes empty NeuronCriteria objects (no filtering criteria) when
    allow_empty=True. This allows functions to optionally accept queries
    for all neurons when appropriate.
    """

    @parse_neuroncriteria(allow_empty=True)
    def dummy_function(neurons: Any) -> Any:
        return neurons

    # Test with empty NeuronCriteria - should return all neurons
    nc = NeuronCriteria()
    result = dummy_function(nc)
    assert set(result) == {"1", "2", "3", "4"}  # Should return all neurons


@patch(
    "crantpy.queries.neurons.get_all_seatable_annotations",
    side_effect=dummy_get_all_seatable_annotations,
)
def test_parse_neuroncriteria_with_no_matches_error(mock_ann: MagicMock) -> None:
    """Test parse_neuroncriteria decorator with NoMatchesError.

    This test verifies that the parse_neuroncriteria decorator correctly
    passes through NoMatchesError exceptions raised by the wrapped function,
    allowing proper error handling when a valid NeuronCriteria doesn't match
    any neurons. This ensures that error conditions are properly communicated
    to the calling code.
    """

    @parse_neuroncriteria(allow_empty=False)
    def dummy_function(neurons: Any) -> Any:
        if len(neurons) == 0:
            raise NoMatchesError("No matches")
        return neurons

    # Test with NeuronCriteria that won't match anything
    with pytest.raises(NoMatchesError):
        nc = NeuronCriteria(cell_class="nonexistent")
        dummy_function(nc)


@patch(
    "crantpy.queries.neurons.get_all_seatable_annotations",
    side_effect=dummy_get_all_seatable_annotations,
)
def test_get_annotations_with_single_id(mock_ann: MagicMock) -> None:
    """Test get_annotations with a single ID.

    This test verifies that the get_annotations function correctly fetches
    annotations for a single neuron ID passed as an integer. It should convert
    the integer to a string and fetch the matching neuron's annotations.
    This tests the basic usage of get_annotations with a single ID.
    """
    result = get_annotations(1)
    assert len(result) == 1
    assert result.iloc[0]["root_id"] == "1"


@patch(
    "crantpy.queries.neurons.get_all_seatable_annotations",
    side_effect=dummy_get_all_seatable_annotations,
)
def test_get_annotations_with_id_list(mock_ann: MagicMock) -> None:
    """Test get_annotations with a list of IDs.

    This test verifies that the get_annotations function correctly fetches
    annotations for multiple neuron IDs passed as a list. It should return
    a DataFrame containing annotations for all the specified neurons.
    This tests the common use case of fetching annotations for multiple neurons.
    """
    result = get_annotations([1, 2])
    assert len(result) == 2
    assert set(result["root_id"]) == {"1", "2"}


@patch(
    "crantpy.queries.neurons.get_all_seatable_annotations",
    side_effect=dummy_get_all_seatable_annotations,
)
def test_get_annotations_with_str_id(mock_ann: MagicMock) -> None:
    """Test get_annotations with a string ID.

    This test verifies that the get_annotations function correctly fetches
    annotations for a single neuron ID passed as a string. It should directly
    use the string ID to fetch the matching neuron's annotations without
    any conversion. This tests an alternative way to call get_annotations.
    """
    result = get_annotations("1")
    assert len(result) == 1
    assert result.iloc[0]["root_id"] == "1"


@patch(
    "crantpy.queries.neurons.get_all_seatable_annotations",
    side_effect=dummy_get_all_seatable_annotations,
)
def test_get_annotations_with_neuroncriteria(mock_ann: MagicMock) -> None:
    """Test get_annotations with NeuronCriteria.

    This test verifies that the get_annotations function correctly processes
    a NeuronCriteria object passed as input. The parse_neuroncriteria decorator
    should convert the NeuronCriteria to a list of root IDs, and the function
    should fetch annotations for those neurons. This tests the integration
    between NeuronCriteria and get_annotations.
    """
    nc = NeuronCriteria(cell_class="foo")
    result = get_annotations(nc)
    assert len(result) == 1
    assert result.iloc[0]["root_id"] == "1"


@patch(
    "crantpy.queries.neurons.get_all_seatable_annotations",
    side_effect=dummy_get_all_seatable_annotations,
)
def test_get_annotations_invalid_input(mock_ann: MagicMock) -> None:
    """Test get_annotations with invalid input.

    This test verifies that the get_annotations function raises a ValueError
    when given an input type that it doesn't support (like a dictionary).
    This ensures proper error handling for invalid inputs rather than
    silent failures or confusing behavior.
    """
    with pytest.raises(ValueError) as excinfo:
        result = get_annotations([{"root_id": "1"}])  # Invalid input type
    assert "Cannot convert" in str(excinfo.value)


@patch(
    "crantpy.queries.neurons.get_all_seatable_annotations",
    side_effect=dummy_get_all_seatable_annotations,
)
def test_get_annotations_no_matches(mock_ann: MagicMock) -> None:
    """Test get_annotations with no matches.

    This test verifies that the get_annotations function raises a NoMatchesError
    when given a neuron ID that doesn't exist in the annotations. This is crucial
    for proper error handling when trying to fetch annotations for non-existent
    neurons.
    """
    with pytest.raises(NoMatchesError) as excinfo:
        result = get_annotations([999])  # ID that doesn't exist
    assert "No matching neurons found" in str(excinfo.value)


def test_parse_neuroncriteria_backward_compatibility() -> None:
    """Test that parse_neuroncriteria can still be imported from the old location.

    This test verifies that the backward compatibility import works and generates
    a deprecation warning to guide users to the new import location.
    """
    # Test that we can import from the old location
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        from crantpy.queries.neurons import parse_neuroncriteria as old_import

        # Create a simple test function with the decorator
        @old_import()
        def test_function(neurons=None):
            return neurons

        # Check that a deprecation warning was issued
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "parse_neuroncriteria has been moved" in str(w[0].message)
        assert "crantpy.utils.decorators" in str(w[0].message)

        # Test that the function still works
        assert callable(test_function)
