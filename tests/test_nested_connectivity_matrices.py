# -*- coding: utf-8 -*-
"""Tests for NestedMatrix.from_synapses_by_neuropil."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

from crantpy.utils import ordering
from crantpy.queries.nested_connectivity_matrices import (
    DEFAULT_WITHIN_TYPE_ORDER,
    EB_COLUMN_ORDER,
    ColumnOrderRule,
    NeuronOrder,
    DirectedNestedMatrix,
    NestedMatrix,
    NeuropilCollection,
)


def _make_synapses(n: int = 6) -> pd.DataFrame:
    """Helper to create a small synapse DataFrame with positions."""
    return pd.DataFrame(
        {
            "pre_pt_root_id": [1, 1, 2, 2, 3, 3][:n],
            "post_pt_root_id": [4, 5, 4, 5, 4, 5][:n],
            "ctr_pt_position": [
                [100, 200, 300],
                [110, 210, 310],
                [120, 220, 320],
                [130, 230, 330],
                [140, 240, 340],
                [150, 250, 350],
            ][:n],
        }
    )


def _make_annotations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "root_id": [1, 2, 3, 4, 5],
            "cell_type": ["KC", "KC", "KC", "MB", "MB"],
        }
    )


def _make_directed_annotations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "root_id": [1, 2, 3, 4, 5],
            "cell_type": ["ER_input", "ER_input", "ER", "ER", "Other"],
        }
    )


def _mock_mesh(mask: list[bool]) -> MagicMock:
    """Create a mock mesh whose .contains() returns the given boolean array."""
    m = MagicMock()
    m.contains.return_value = np.array(mask)
    return m


def _line_collection_widths(ax: plt.Axes) -> list[float]:
    return [
        float(np.atleast_1d(collection.get_linewidths())[0])
        for collection in ax.collections
        if isinstance(collection, LineCollection)
    ]


@patch("crantpy.viz.mesh.load_neuropil_mesh")
def test_single_neuropil(mock_load: MagicMock) -> None:
    mock_load.return_value = _mock_mesh([True, True, True, False, False, False])

    result = NestedMatrix.from_synapses_by_neuropil(
        synapses_df=_make_synapses(),
        neuron_annotations=_make_annotations(),
        neuropil_names=["antennal_lobe_left"],
        coordinates="nm",
    )

    assert isinstance(result, dict)
    assert "antennal_lobe_left" in result
    assert isinstance(result["antennal_lobe_left"], NestedMatrix)
    mock_load.assert_called_once_with("antennal_lobe_left")


@patch("crantpy.viz.mesh.load_neuropil_mesh")
def test_multiple_neuropils(mock_load: MagicMock) -> None:
    masks = {
        "antennal_lobe_left": [True, True, False, False, False, False],
        "fan_shaped_body": [False, False, False, False, True, True],
    }
    mock_load.side_effect = lambda name: _mock_mesh(masks[name])

    result = NestedMatrix.from_synapses_by_neuropil(
        synapses_df=_make_synapses(),
        neuron_annotations=_make_annotations(),
        neuropil_names=["antennal_lobe_left", "fan_shaped_body"],
        coordinates="nm",
    )

    assert "antennal_lobe_left" in result
    assert "fan_shaped_body" in result
    # synapses 2-3 are unassigned -> "other"
    assert "other" in result


@patch("crantpy.viz.mesh.load_neuropil_mesh")
def test_other_category(mock_load: MagicMock) -> None:
    # Only first synapse inside the mesh
    mock_load.return_value = _mock_mesh([True, False, False, False, False, False])

    result = NestedMatrix.from_synapses_by_neuropil(
        synapses_df=_make_synapses(),
        neuron_annotations=_make_annotations(),
        neuropil_names=["antennal_lobe_left"],
        coordinates="nm",
    )

    assert "other" in result
    assert isinstance(result["other"], NestedMatrix)


@patch("crantpy.viz.mesh.load_neuropil_mesh")
def test_include_other_false(mock_load: MagicMock) -> None:
    mock_load.return_value = _mock_mesh([True, False, False, False, False, False])

    result = NestedMatrix.from_synapses_by_neuropil(
        synapses_df=_make_synapses(),
        neuron_annotations=_make_annotations(),
        neuropil_names=["antennal_lobe_left"],
        coordinates="nm",
        include_other=False,
    )

    assert "other" not in result


@patch("crantpy.viz.mesh.load_neuropil_mesh")
def test_pixel_coordinate_conversion(mock_load: MagicMock) -> None:
    mock_mesh = _mock_mesh([True, True])
    mock_load.return_value = mock_mesh

    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 2],
            "post_pt_root_id": [3, 4],
            "ctr_pt_position": [[10, 20, 5], [15, 25, 10]],
        }
    )
    annotations = pd.DataFrame(
        {"root_id": [1, 2, 3, 4], "cell_type": ["A", "A", "B", "B"]}
    )

    NestedMatrix.from_synapses_by_neuropil(
        synapses_df=synapses,
        neuron_annotations=annotations,
        neuropil_names=["antennal_lobe_left"],
        coordinates="pixels",
    )

    # Verify contains() was called with scaled coordinates (8, 8, 42)
    called_positions = mock_mesh.contains.call_args[0][0]
    expected = np.array([[10 * 8, 20 * 8, 5 * 42], [15 * 8, 25 * 8, 10 * 42]])
    np.testing.assert_array_almost_equal(called_positions, expected)


@patch("crantpy.viz.mesh.load_neuropil_mesh")
def test_nm_coordinates_unmodified(mock_load: MagicMock) -> None:
    mock_mesh = _mock_mesh([True, True])
    mock_load.return_value = mock_mesh

    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 2],
            "post_pt_root_id": [3, 4],
            "ctr_pt_position": [[100, 200, 300], [400, 500, 600]],
        }
    )
    annotations = pd.DataFrame(
        {"root_id": [1, 2, 3, 4], "cell_type": ["A", "A", "B", "B"]}
    )

    NestedMatrix.from_synapses_by_neuropil(
        synapses_df=synapses,
        neuron_annotations=annotations,
        neuropil_names=["antennal_lobe_left"],
        coordinates="nm",
    )

    called_positions = mock_mesh.contains.call_args[0][0]
    expected = np.array([[100, 200, 300], [400, 500, 600]])
    np.testing.assert_array_almost_equal(called_positions, expected)


@patch("crantpy.viz.mesh.load_neuropil_mesh")
def test_default_loads_all_neuropils(mock_load: MagicMock) -> None:
    from crantpy.utils.config import NEUROPIL_MESH_DICT

    all_names = list(NEUROPIL_MESH_DICT.values())
    mock_load.return_value = _mock_mesh([False] * 6)

    NestedMatrix.from_synapses_by_neuropil(
        synapses_df=_make_synapses(),
        neuron_annotations=_make_annotations(),
        coordinates="nm",
    )

    assert mock_load.call_count == len(all_names)
    called_names = [call.args[0] for call in mock_load.call_args_list]
    assert set(called_names) == set(all_names)


def test_invalid_neuropil_name() -> None:
    with pytest.raises(ValueError, match="Invalid neuropil name"):
        NestedMatrix.from_synapses_by_neuropil(
            synapses_df=_make_synapses(),
            neuron_annotations=_make_annotations(),
            neuropil_names=["nonexistent_neuropil"],
            coordinates="nm",
        )


@patch("crantpy.viz.mesh.load_neuropil_mesh")
def test_alias_neuropil_name(mock_load: MagicMock) -> None:
    mock_load.return_value = _mock_mesh([True, False, False, False, False, False])

    result = NestedMatrix.from_synapses_by_neuropil(
        synapses_df=_make_synapses(),
        neuron_annotations=_make_annotations(),
        neuropil_names=["nodulus_right"],
        coordinates="nm",
    )

    assert "nodulus_right" in result
    mock_load.assert_called_once_with("nodulus_right")


def test_invalid_coordinates() -> None:
    with pytest.raises(ValueError, match="coordinates must be"):
        NestedMatrix.from_synapses_by_neuropil(
            synapses_df=_make_synapses(),
            neuron_annotations=_make_annotations(),
            neuropil_names=["antennal_lobe_left"],
            coordinates="voxels",
        )


def test_empty_dataframe() -> None:
    empty_synapses = pd.DataFrame(
        columns=["pre_pt_root_id", "post_pt_root_id", "ctr_pt_position"]
    )
    result = NestedMatrix.from_synapses_by_neuropil(
        synapses_df=empty_synapses,
        neuron_annotations=_make_annotations(),
        neuropil_names=["antennal_lobe_left"],
        coordinates="nm",
    )
    assert result == {}


@patch("crantpy.viz.mesh.load_neuropil_mesh")
def test_from_synapses_by_neuropil_logs_construction(
    mock_load: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    mock_load.return_value = _mock_mesh([True, False, False, False, False, False])

    caplog.set_level("INFO", logger="crantpy.queries.nested_connectivity_matrices")
    NestedMatrix.from_synapses_by_neuropil(
        synapses_df=_make_synapses(),
        neuron_annotations=_make_annotations(),
        neuropil_names=["antennal_lobe_left"],
        coordinates="nm",
    )

    assert "Building NestedMatrix collection by neuropil" in caplog.text
    assert "Neuropil antennal_lobe_left contains 1 synapse(s)" in caplog.text
    assert (
        "Building NestedMatrix for neuropil antennal_lobe_left with 1 synapse(s)"
        in caplog.text
    )


@patch("crantpy.viz.mesh.load_neuropil_mesh")
def test_synapse_in_multiple_neuropils(mock_load: MagicMock) -> None:
    """A synapse inside two overlapping meshes appears in both."""
    masks = {
        "antennal_lobe_left": [True, True, False, False],
        "fan_shaped_body": [False, True, True, False],
    }
    mock_load.side_effect = lambda name: _mock_mesh(masks[name])

    synapses = _make_synapses(4)
    result = NestedMatrix.from_synapses_by_neuropil(
        synapses_df=synapses,
        neuron_annotations=_make_annotations(),
        neuropil_names=["antennal_lobe_left", "fan_shaped_body"],
        coordinates="nm",
    )

    assert "antennal_lobe_left" in result
    assert "fan_shaped_body" in result
    # The shared synapse (pre 1 -> post 5, index 1) must contribute an edge to
    # BOTH matrices -- assignment is not exclusive first-mesh-wins.
    assert result["antennal_lobe_left"].matrix.loc["1", "5"] > 0
    assert result["fan_shaped_body"].matrix.loc["1", "5"] > 0
    # synapse index 3 is unassigned
    assert "other" in result
    assert result["other"].matrix.loc["2", "5"] > 0


def test_collection_plot_shortcut() -> None:
    """NeuropilCollection.plot(name) delegates to the matrix's plot()."""
    mock_matrix = MagicMock(spec=NestedMatrix)
    mock_matrix.plot.return_value = ("fig", "ax")

    collection = NeuropilCollection({"ellipsoid_body": mock_matrix})
    result = collection.plot("ellipsoid_body", level="neuron")

    mock_matrix.plot.assert_called_once_with(level="neuron")
    assert result == ("fig", "ax")


def test_collection_attribute_access() -> None:
    """NeuropilCollection supports dot-access for neuropil names."""
    mock_matrix = MagicMock(spec=NestedMatrix)
    collection = NeuropilCollection({"ellipsoid_body": mock_matrix})

    assert collection.ellipsoid_body is mock_matrix


def test_collection_attribute_error() -> None:
    """Accessing a missing neuropil via attribute raises AttributeError."""
    collection = NeuropilCollection({"ellipsoid_body": MagicMock()})
    with pytest.raises(AttributeError, match="No neuropil named"):
        _ = collection.nonexistent


def test_directed_from_synapses_filters_source_and_target_types() -> None:
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 1, 2, 3],
            "post_pt_root_id": [3, 4, 3, 4],
        }
    )

    matrix = DirectedNestedMatrix.from_synapses(
        synapses,
        _make_directed_annotations(),
        source_types=["ER_input"],
        target_types=["ER"],
        weight_mode="count",
    )

    assert list(matrix.matrix.index) == ["1", "2"]
    assert list(matrix.matrix.columns) == ["3", "4"]
    assert "3" not in matrix.source_neurons
    assert matrix.matrix.loc["1", "3"] == 1.0
    assert matrix.matrix.loc["1", "4"] == 1.0
    assert matrix.matrix.loc["2", "3"] == 1.0
    assert matrix.sum_type_matrix.loc["ER_input", "ER"] == 3.0


def test_directed_selectors_accept_scalar_strings() -> None:
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 1, 2, 3],
            "post_pt_root_id": [3, 4, 3, 4],
        }
    )

    by_type = DirectedNestedMatrix.from_synapses(
        synapses,
        _make_directed_annotations(),
        source_types="ER_input",
        target_types="ER",
        weight_mode="count",
    )
    by_id = DirectedNestedMatrix.from_synapses(
        synapses,
        _make_directed_annotations(),
        source_ids="1",
        target_ids="3",
        weight_mode="count",
    )

    assert list(by_type.matrix.index) == ["1", "2"]
    assert list(by_type.matrix.columns) == ["3", "4"]
    assert list(by_id.matrix.index) == ["1"]
    assert list(by_id.matrix.columns) == ["3"]


def test_directed_from_connectivity_sums_duplicate_edges() -> None:
    connections = pd.DataFrame(
        {
            "pre": [1, 1, 2],
            "post": [3, 3, 4],
            "weight": [2, 5, 7],
        }
    )

    matrix = DirectedNestedMatrix.from_connectivity(
        connections,
        _make_directed_annotations(),
        source_types="ER_input",
        target_types="ER",
    )

    assert matrix.matrix.loc["1", "3"] == 7.0
    assert matrix.matrix.loc["2", "4"] == 7.0


def test_directed_from_connectivity_sums_string_weights_numerically() -> None:
    connections = pd.DataFrame(
        {
            "pre": [1, 1],
            "post": [3, 3],
            "weight": ["2", "5"],
        }
    )

    matrix = DirectedNestedMatrix.from_connectivity(
        connections,
        _make_directed_annotations(),
        source_types="ER_input",
        target_types="ER",
    )

    assert matrix.matrix.loc["1", "3"] == 7.0


def test_directed_from_synapses_annotation_scope_all_drops_null_ids() -> None:
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [1, np.nan, 2],
            "post_pt_root_id": [3, 4, np.nan],
        }
    )

    matrix = DirectedNestedMatrix.from_synapses(
        synapses,
        _make_directed_annotations(),
        weight_mode="count",
        annotation_scope="all",
    )

    assert list(matrix.matrix.index) == ["1"]
    assert list(matrix.matrix.columns) == ["3"]
    assert "nan" not in matrix.source_neurons
    assert "nan" not in matrix.target_neurons


def test_directed_from_synapses_matches_int_like_float_ids() -> None:
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [1.0, 1.0, 2.0],
            "post_pt_root_id": [3.0, 4.0, 3.0],
        }
    )

    matrix = DirectedNestedMatrix.from_synapses(
        synapses,
        _make_directed_annotations(),
        source_types=["ER_input"],
        target_types=["ER"],
        weight_mode="count",
    )

    assert list(matrix.matrix.index) == ["1", "2"]
    assert list(matrix.matrix.columns) == ["3", "4"]
    assert matrix.matrix.loc["1", "3"] == 1.0
    assert matrix.matrix.loc["1", "4"] == 1.0
    assert matrix.matrix.loc["2", "3"] == 1.0


def test_directed_from_connectivity_normalizes_float_axes_and_drops_null_axes() -> None:
    adjacency = pd.DataFrame(
        [[5, 0, 9], [7, 11, 13], [17, 19, 23]],
        index=[1.0, 2.0, np.nan],
        columns=[3.0, 4.0, np.nan],
    )
    annotations = pd.DataFrame(
        {
            "root_id": [1, 2, 3, 4, np.nan],
            "cell_type": ["ER_input", "ER_input", "ER", "ER", "Bogus"],
        }
    )

    matrix = DirectedNestedMatrix.from_connectivity(
        adjacency,
        annotations,
        source_types=["ER_input"],
        target_types=["ER"],
        annotation_scope="all",
    )

    assert list(matrix.matrix.index) == ["1", "2"]
    assert list(matrix.matrix.columns) == ["3", "4"]
    assert "nan" not in matrix.source_neurons
    assert "nan" not in matrix.target_neurons
    assert matrix.matrix.loc["1", "3"] == 5.0
    assert matrix.matrix.loc["2", "4"] == 11.0


def test_directed_accepts_numeric_type_labels() -> None:
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [1.0, 2.0],
            "post_pt_root_id": [3.0, 4.0],
        }
    )
    annotations = pd.DataFrame(
        {
            "root_id": [1, 2, 3, 4],
            "cell_type": [1.0, 1.0, 2.0, 2.0],
        }
    )

    matrix = DirectedNestedMatrix.from_synapses(
        synapses,
        annotations,
        source_types=1,
        target_types=2,
        weight_mode="count",
    )

    assert list(matrix.matrix.index) == ["1", "2"]
    assert list(matrix.matrix.columns) == ["3", "4"]
    assert list(matrix.source_type_boundaries) == ["1"]
    assert list(matrix.target_type_boundaries) == ["2"]


def test_directed_constructor_accepts_public_axis_metadata() -> None:
    matrix = DirectedNestedMatrix(
        matrix=pd.DataFrame([[1, 2]], index=[1.0], columns=[3.0, 4.0]),
        source_neurons=[1],
        target_neurons=[3, 4],
        source_type_boundaries={1.0: (0, 1)},
        target_type_boundaries={2.0: (0, 2)},
        source_neuron_to_type={1: 1.0},
        target_neuron_to_type={3: 2.0, 4: 2.0},
    )

    assert list(matrix.matrix.index) == ["1"]
    assert list(matrix.matrix.columns) == ["3", "4"]
    assert matrix.source_type_boundaries["1"] == (0, 1)
    assert matrix.target_type_boundaries["2"] == (0, 2)


def test_directed_accepts_independent_type_orders() -> None:
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 2],
            "post_pt_root_id": [3, 4],
        }
    )
    annotations = pd.DataFrame(
        {
            "root_id": [1, 2, 3, 4],
            "cell_type": ["A", "B", "X", "Y"],
        }
    )

    matrix = DirectedNestedMatrix.from_synapses(
        synapses,
        annotations,
        source_order=["B", "A"],
        target_order=["Y", "X"],
        weight_mode="count",
    )

    assert list(matrix.matrix.index) == ["2", "1"]
    assert list(matrix.matrix.columns) == ["4", "3"]


def test_directed_from_synapses_unions_type_and_explicit_id_selectors() -> None:
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 2, 9],
            "post_pt_root_id": [3, 4, 3],
        }
    )

    matrix = DirectedNestedMatrix.from_synapses(
        synapses,
        _make_directed_annotations(),
        source_types=["ER_input"],
        source_ids=[9],
        target_types=["ER"],
        weight_mode="count",
        annotation_scope="all",
    )

    assert list(matrix.matrix.index) == ["1", "2", "9"]
    assert list(matrix.matrix.columns) == ["3", "4"]
    assert matrix.source_type_boundaries["ER_input"] == (0, 2)
    assert "9" not in matrix.source_neuron_to_type
    assert matrix.matrix.loc["9", "3"] == 1.0


def test_directed_source_only_selector_keeps_all_target_axis_ids() -> None:
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 5],
            "post_pt_root_id": [3, 4],
        }
    )

    matrix = DirectedNestedMatrix.from_synapses(
        synapses,
        _make_directed_annotations(),
        source_types=["ER_input"],
        weight_mode="count",
    )

    assert list(matrix.matrix.index) == ["1"]
    assert list(matrix.matrix.columns) == ["3", "4"]
    assert matrix.matrix.loc["1", "3"] == 1.0
    assert matrix.matrix.loc["1", "4"] == 0.0


def test_directed_target_only_selector_keeps_all_source_axis_ids() -> None:
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 2],
            "post_pt_root_id": [3, 5],
        }
    )

    matrix = DirectedNestedMatrix.from_synapses(
        synapses,
        _make_directed_annotations(),
        target_types=["ER"],
        weight_mode="count",
    )

    assert list(matrix.matrix.index) == ["1", "2"]
    assert list(matrix.matrix.columns) == ["3"]
    assert matrix.matrix.loc["1", "3"] == 1.0
    assert matrix.matrix.loc["2", "3"] == 0.0


def test_directed_type_selectors_use_resolved_duplicate_annotations() -> None:
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 1, 2],
            "post_pt_root_id": [3, 4, 4],
        }
    )
    annotations = pd.DataFrame(
        {
            "root_id": [1, 1, 2, 3, 4, 4],
            "cell_type": ["A", "B", "B", "X", "Y", "Z"],
        }
    )

    selected_later_duplicate = DirectedNestedMatrix.from_synapses(
        synapses,
        annotations,
        source_types=["B"],
        target_types=["Z"],
        weight_mode="count",
    )
    selected_resolved_type = DirectedNestedMatrix.from_synapses(
        synapses,
        annotations,
        source_types=["A"],
        target_types=["Y"],
        weight_mode="count",
    )

    assert list(selected_later_duplicate.matrix.index) == ["2"]
    assert list(selected_later_duplicate.matrix.columns) == []
    assert "1" not in selected_later_duplicate.source_neurons
    assert "4" not in selected_later_duplicate.target_neurons

    assert list(selected_resolved_type.matrix.index) == ["1"]
    assert list(selected_resolved_type.matrix.columns) == ["4"]
    assert selected_resolved_type.source_neuron_to_type["1"] == "A"
    assert selected_resolved_type.target_neuron_to_type["4"] == "Y"


def test_directed_type_matrices_are_rectangular() -> None:
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 1, 2, 5],
            "post_pt_root_id": [3, 4, 3, 3],
        }
    )

    matrix = DirectedNestedMatrix.from_synapses(
        synapses,
        _make_directed_annotations(),
        source_types=["ER_input", "Other"],
        target_types=["ER"],
        weight_mode="count",
    )

    assert list(matrix.sum_type_matrix.index) == ["ER_input", "Other"]
    assert list(matrix.sum_type_matrix.columns) == ["ER"]
    assert matrix.sum_type_matrix.shape == (2, 1)
    assert matrix.mean_type_matrix.loc["ER_input", "ER"] == pytest.approx(0.75)
    assert matrix.mean_type_matrix.loc["Other", "ER"] == pytest.approx(0.5)


def test_directed_relative_outgoing_selected_scope_normalizes_selected_targets() -> (
    None
):
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 1, 1],
            "post_pt_root_id": [3, 4, 5],
        }
    )

    matrix = DirectedNestedMatrix.from_synapses(
        synapses,
        _make_directed_annotations(),
        source_types=["ER_input"],
        target_types=["ER"],
        weight_mode="relative_outgoing",
        normalization_scope="selected",
    )

    assert matrix.matrix.loc["1", "3"] == pytest.approx(0.5)
    assert matrix.matrix.loc["1", "4"] == pytest.approx(0.5)
    assert matrix.matrix.loc["1"].sum() == pytest.approx(1.0)


def test_directed_relative_outgoing_all_scope_uses_all_targets_denominator() -> None:
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 1, 1],
            "post_pt_root_id": [3, 4, 5],
        }
    )

    matrix = DirectedNestedMatrix.from_synapses(
        synapses,
        _make_directed_annotations(),
        source_types=["ER_input"],
        target_types=["ER"],
        weight_mode="relative_outgoing",
        normalization_scope="all",
    )

    assert matrix.matrix.loc["1", "3"] == pytest.approx(1 / 3)
    assert matrix.matrix.loc["1", "4"] == pytest.approx(1 / 3)
    assert matrix.matrix.loc["1"].sum() == pytest.approx(2 / 3)


def test_directed_relative_incoming_all_scope_uses_all_sources_denominator() -> None:
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 5],
            "post_pt_root_id": [3, 3],
        }
    )

    matrix = DirectedNestedMatrix.from_synapses(
        synapses,
        _make_directed_annotations(),
        source_types=["ER_input"],
        target_types=["ER"],
        weight_mode="relative_incoming",
        normalization_scope="all",
    )

    assert matrix.matrix.loc["1", "3"] == pytest.approx(0.5)
    assert matrix.matrix["3"].sum() == pytest.approx(0.5)


@patch("crantpy.viz.mesh.load_neuropil_mesh")
def test_directed_from_synapses_by_neuropil_returns_directed_matrices(
    mock_load: MagicMock,
) -> None:
    mock_load.return_value = _mock_mesh([True, True, False])
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 2, 5],
            "post_pt_root_id": [3, 4, 3],
            "ctr_pt_position": [[100, 200, 300], [110, 210, 310], [120, 220, 320]],
        }
    )

    result = DirectedNestedMatrix.from_synapses_by_neuropil(
        synapses,
        _make_directed_annotations(),
        neuropil_names=["antennal_lobe_left"],
        source_types=["ER_input"],
        target_types=["ER"],
        coordinates="nm",
        weight_mode="count",
    )

    assert isinstance(result["antennal_lobe_left"], DirectedNestedMatrix)
    assert list(result["antennal_lobe_left"].matrix.index) == ["1", "2"]
    assert list(result["antennal_lobe_left"].matrix.columns) == ["3", "4"]


@patch("crantpy.viz.mesh.load_neuropil_mesh")
def test_directed_from_synapses_by_neuropil_skips_selector_empty_matrices(
    mock_load: MagicMock,
) -> None:
    mock_load.return_value = _mock_mesh([True])
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [5],
            "post_pt_root_id": [3],
            "ctr_pt_position": [[100, 200, 300]],
        }
    )

    result = DirectedNestedMatrix.from_synapses_by_neuropil(
        synapses,
        _make_directed_annotations(),
        neuropil_names=["antennal_lobe_left"],
        source_types=["ER_input"],
        target_types=["ER"],
        coordinates="nm",
        weight_mode="count",
    )

    assert list(result) == []


@patch("crantpy.viz.mesh.load_neuropil_mesh")
def test_directed_from_synapses_by_neuropil_skips_null_only_matrices(
    mock_load: MagicMock,
) -> None:
    mock_load.return_value = _mock_mesh([True])
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [np.nan],
            "post_pt_root_id": [3],
            "ctr_pt_position": [[100, 200, 300]],
        }
    )

    result = DirectedNestedMatrix.from_synapses_by_neuropil(
        synapses,
        _make_directed_annotations(),
        neuropil_names=["antennal_lobe_left"],
        coordinates="nm",
        weight_mode="count",
        annotation_scope="all",
    )

    assert list(result) == []


def test_directed_plot_handles_rectangular_neuron_and_type_matrices() -> None:
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 1, 2, 5],
            "post_pt_root_id": [3, 4, 3, 3],
        }
    )
    matrix = DirectedNestedMatrix.from_synapses(
        synapses,
        _make_directed_annotations(),
        source_types=["ER_input", "Other"],
        target_types=["ER"],
        weight_mode="count",
    )

    fig_neuron, ax_neuron = matrix.plot(level="neuron")
    fig_type, ax_type = matrix.plot(level="type_sum")

    assert np.asarray(ax_neuron.images[0].get_array()).shape == (3, 2)
    assert np.asarray(ax_type.images[0].get_array()).shape == (2, 1)

    plt.close(fig_neuron)
    plt.close(fig_type)


def test_directed_plot_rejects_zero_width_matrix() -> None:
    matrix = DirectedNestedMatrix(
        matrix=pd.DataFrame(index=["1"], columns=[]),
        source_neurons=["1"],
        target_neurons=[],
        source_type_boundaries={"A": (0, 1)},
        source_neuron_to_type={"1": "A"},
    )

    with pytest.raises(ValueError, match="zero rows or columns"):
        matrix.plot()


def test_from_connectivity_accepts_pre_post_weight_edges() -> None:
    connectivity = pd.DataFrame(
        {
            "pre": [1, 2],
            "post": [3, 3],
            "weight": [5, 7],
        }
    )
    annotations = pd.DataFrame({"root_id": [1, 2, 3], "cell_type": ["ER", "ER", "PBt"]})

    matrix = NestedMatrix.from_connectivity(connectivity, annotations)

    assert list(matrix.matrix.index) == ["1", "2", "3"]
    assert list(matrix.matrix.columns) == ["1", "2", "3"]
    assert matrix.matrix.loc["1", "3"] == 5.0
    assert matrix.matrix.loc["2", "3"] == 7.0


def test_from_connectivity_filters_to_annotated_neurons(
    caplog: pytest.LogCaptureFixture,
) -> None:
    adjacency = pd.DataFrame([[0, 4], [2, 0]], index=[1, 2], columns=[1, 2])
    annotations = pd.DataFrame({"root_id": [1], "cell_type": ["ER"]})

    caplog.set_level("INFO", logger="crantpy.queries.nested_connectivity_matrices")
    matrix = NestedMatrix.from_connectivity(adjacency, annotations)

    assert list(matrix.matrix.index) == ["1"]
    assert list(matrix.matrix.columns) == ["1"]
    assert "Filtered connectivity to neurons present in annotations" in caplog.text


def test_from_connectivity_annotation_scope_all_keeps_unannotated_neurons(
    caplog: pytest.LogCaptureFixture,
) -> None:
    adjacency = pd.DataFrame([[0, 4], [2, 0]], index=[1, 2], columns=[1, 2])
    annotations = pd.DataFrame({"root_id": [1], "cell_type": ["ER"]})

    caplog.set_level("WARNING", logger="crantpy.utils.ordering")
    matrix = NestedMatrix.from_connectivity(
        adjacency,
        annotations,
        annotation_scope="all",
    )

    assert list(matrix.matrix.index) == ["1", "2"]
    assert list(matrix.matrix.columns) == ["1", "2"]
    assert "missing from annotations" in caplog.text
    assert [r.name for r in caplog.records] == ["crantpy.utils.ordering"]


def test_from_connectivity_logs_nested_ordering(
    caplog: pytest.LogCaptureFixture,
) -> None:
    connectivity = pd.DataFrame(
        {
            "pre": [1, 2],
            "post": [3, 3],
            "weight": [5, 7],
        }
    )
    annotations = pd.DataFrame({"root_id": [1, 2, 3], "cell_type": ["ER", "ER", "PBt"]})

    caplog.set_level("DEBUG", logger="crantpy.queries.nested_connectivity_matrices")
    NestedMatrix.from_connectivity(connectivity, annotations)

    assert "Building NestedMatrix from connectivity" in caplog.text
    assert "Resolved type order: ['ER', 'PBt']" in caplog.text
    assert "Type ER occupies ordered_neurons[0:2]" in caplog.text


def test_from_connectivity_respects_explicit_type_order() -> None:
    adjacency = pd.DataFrame(
        [[0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 3], [4, 0, 0, 0]],
        index=[1, 2, 3, 4],
        columns=[1, 2, 3, 4],
    )
    annotations = pd.DataFrame(
        {
            "root_id": [1, 2, 3, 4],
            "cell_type": ["EPG/PEG", "ER1", "ExR2", "delta7"],
        }
    )

    matrix = NestedMatrix.from_connectivity(
        adjacency,
        annotations,
        order=["ExR2", "EPG/PEG", "delta7", "ER1"],
    )

    assert list(matrix.type_boundaries.keys()) == [
        "ExR2",
        "EPG/PEG",
        "delta7",
        "ER1",
    ]
    assert list(matrix.matrix.index) == ["3", "1", "4", "2"]
    assert list(matrix.matrix.columns) == ["3", "1", "4", "2"]


def test_from_connectivity_preserves_annotation_order_within_type() -> None:
    adjacency = pd.DataFrame(
        [[0, 1], [2, 0]],
        index=["20", "10"],
        columns=["20", "10"],
    )
    annotations = pd.DataFrame(
        {
            "root_id": ["20", "10"],
            "cell_type": ["EPG/PEG", "EPG/PEG"],
        }
    )

    matrix = NestedMatrix.from_connectivity(adjacency, annotations)

    assert list(matrix.matrix.index) == ["20", "10"]
    assert list(matrix.matrix.columns) == ["20", "10"]


def test_from_connectivity_orders_epg_by_cell_subtype_when_available() -> None:
    adjacency = pd.DataFrame(
        np.zeros((4, 4)),
        index=["1", "2", "3", "4"],
        columns=["1", "2", "3", "4"],
    )
    annotations = pd.DataFrame(
        {
            "root_id": ["1", "2", "3", "4"],
            "cell_type": ["EPG/PEG", "EPG/PEG", "EPG/PEG", "EPG/PEG"],
            "cell_subtype": ["EPG/PEG_L5", None, "EPG/PEG_R1", "EPG/PEG_L8"],
        }
    )

    matrix = NestedMatrix.from_connectivity(adjacency, annotations)

    assert list(matrix.matrix.index) == ["3", "4", "1", "2"]
    assert list(matrix.matrix.columns) == ["3", "4", "1", "2"]


def test_from_connectivity_epg_cell_instance_precedes_cell_subtype() -> None:
    adjacency = pd.DataFrame(
        np.zeros((4, 4)),
        index=["1", "2", "3", "4"],
        columns=["1", "2", "3", "4"],
    )
    annotations = pd.DataFrame(
        {
            "root_id": ["1", "2", "3", "4"],
            "cell_type": ["EPG/PEG", "EPG/PEG", "EPG/PEG", "EPG/PEG"],
            "cell_instance": [
                "EPG/PEG_L8",
                None,
                "EPG/PEG_R1",
                "EPG/PEG_R3",
            ],
            "cell_subtype": [
                "EPG/PEG_R1",
                "EPG/PEG_L5",
                "EPG/PEG_L8",
                "EPG/PEG_L7",
            ],
        }
    )

    matrix = NestedMatrix.from_connectivity(adjacency, annotations)

    assert list(matrix.matrix.index) == ["3", "1", "4", "2"]
    assert list(matrix.matrix.columns) == ["3", "1", "4", "2"]


def test_from_connectivity_columnar_rule_can_order_non_epg_type() -> None:
    adjacency = pd.DataFrame(
        np.zeros((4, 4)),
        index=["1", "2", "3", "4"],
        columns=["1", "2", "3", "4"],
    )
    annotations = pd.DataFrame(
        {
            "root_id": ["1", "2", "3", "4"],
            "cell_type": [
                "columnar_test",
                "columnar_test",
                "columnar_test",
                "columnar_test",
            ],
            "cell_instance": ["type_L5", "type_R1", "type_L8", "type_R3"],
        }
    )

    matrix = NestedMatrix.from_connectivity(
        adjacency,
        annotations,
        order={
            "within": {
                "columnar_test": ColumnOrderRule(
                    order=EB_COLUMN_ORDER.order,
                    label_columns=("cell_instance",),
                )
            }
        },
    )

    assert list(matrix.matrix.index) == ["2", "3", "4", "1"]
    assert list(matrix.matrix.columns) == ["2", "3", "4", "1"]


def test_from_connectivity_unranked_columnar_rows_keep_annotation_order(
    caplog: pytest.LogCaptureFixture,
) -> None:
    adjacency = pd.DataFrame(
        np.zeros((4, 4)),
        index=["1", "2", "3", "4"],
        columns=["1", "2", "3", "4"],
    )
    annotations = pd.DataFrame(
        {
            "root_id": ["1", "2", "3", "4"],
            "cell_type": ["EPG/PEG", "EPG/PEG", "EPG/PEG", "EPG/PEG"],
            "cell_instance": ["no_column", "EPG/PEG_R1", None, "EPG/PEG_L8"],
            "cell_subtype": [None, None, "still_no_column", None],
        }
    )

    caplog.set_level("WARNING", logger="crantpy.utils.ordering")
    matrix = NestedMatrix.from_connectivity(adjacency, annotations)

    assert list(matrix.matrix.index) == ["2", "4", "1", "3"]
    assert list(matrix.matrix.columns) == ["2", "4", "1", "3"]
    assert "Could not resolve a ranked column label for neuron 1" in caplog.text
    assert "Could not resolve a ranked column label for neuron 3" in caplog.text
    assert {r.name for r in caplog.records} == {"crantpy.utils.ordering"}


def test_init_rejects_matrix_axis_mismatch() -> None:
    matrix = pd.DataFrame([[1, 2], [3, 4]], index=["1", "2"], columns=["1", "3"])

    with pytest.raises(ValueError, match="matrix index and columns must match exactly"):
        NestedMatrix(
            matrix=matrix,
            type_boundaries={},
            ordered_neurons=["1", "2"],
            neuron_to_type={},
        )


def test_init_rejects_duplicate_normalized_neuron_ids() -> None:
    matrix = pd.DataFrame([[0, 1], [2, 0]], index=[1, 1.0], columns=[1, 1.0])

    with pytest.raises(ValueError, match="duplicate neuron IDs"):
        NestedMatrix(
            matrix=matrix,
            type_boundaries={},
            ordered_neurons=[1, 1.0],
            neuron_to_type={},
        )


def test_directed_init_rejects_duplicate_normalized_neuron_ids() -> None:
    matrix = pd.DataFrame([[1], [2]], index=[1, 1.0], columns=[3])

    with pytest.raises(ValueError, match="duplicate neuron IDs"):
        DirectedNestedMatrix(
            matrix=matrix,
            source_neurons=[1, 1.0],
            target_neurons=[3],
        )


def test_init_rejects_noncontiguous_boundaries() -> None:
    matrix = pd.DataFrame(
        [[0, 1, 0], [0, 0, 2], [3, 0, 0]],
        index=["1", "2", "3"],
        columns=["1", "2", "3"],
    )

    with pytest.raises(ValueError, match="type_boundaries must be contiguous"):
        NestedMatrix(
            matrix=matrix,
            type_boundaries={"A": (1, 2), "B": (2, 3)},
            ordered_neurons=["1", "2", "3"],
            neuron_to_type={"2": "A", "3": "B"},
        )


def test_init_rejects_annotated_neurons_outside_boundaries() -> None:
    matrix = pd.DataFrame(
        [[0, 1, 0], [0, 0, 2], [3, 0, 0]],
        index=["1", "2", "3"],
        columns=["1", "2", "3"],
    )

    with pytest.raises(
        ValueError, match="annotated neurons must appear within type_boundaries"
    ):
        NestedMatrix(
            matrix=matrix,
            type_boundaries={"A": (0, 1)},
            ordered_neurons=["1", "2", "3"],
            neuron_to_type={"1": "A", "3": "B"},
        )


def test_from_synapses_logs_aggregation(caplog: pytest.LogCaptureFixture) -> None:
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 1, 2],
            "post_pt_root_id": [3, 4, 3],
        }
    )
    annotations = pd.DataFrame(
        {"root_id": [1, 2, 3, 4], "cell_type": ["ER", "ER", "PBt", "PBt"]}
    )

    caplog.set_level("INFO", logger="crantpy.queries.nested_connectivity_matrices")
    NestedMatrix.from_synapses(
        synapses,
        annotations,
        weight_mode="relative_outgoing",
    )

    assert "Building NestedMatrix from synapses" in caplog.text
    assert "Aggregated synapses into" in caplog.text
    assert "Constructed NestedMatrix from synapses" in caplog.text


def test_mean_type_matrix_uses_block_mean_not_sum() -> None:
    connectivity = pd.DataFrame(
        {
            "pre": [1, 2, 3],
            "post": [3, 2, 4],
            "weight": [4, 0, 2],
        }
    )
    annotations = pd.DataFrame(
        {"root_id": [1, 2, 3, 4], "cell_type": ["A", "A", "B", "B"]}
    )

    matrix = NestedMatrix.from_connectivity(connectivity, annotations)

    assert matrix.sum_type_matrix.loc["A", "B"] == 4.0
    assert matrix.mean_type_matrix.loc["A", "B"] == 1.0
    assert matrix.mean_type_matrix.loc["B", "B"] == 0.5
    assert matrix.mean_type_matrix.loc["B", "A"] == 0.0


def test_public_matrix_view_is_read_only_and_cached_types_stay_valid() -> None:
    connectivity = pd.DataFrame(
        {
            "pre": [1, 2, 3],
            "post": [3, 2, 4],
            "weight": [4, 0, 2],
        }
    )
    annotations = pd.DataFrame(
        {"root_id": [1, 2, 3, 4], "cell_type": ["A", "A", "B", "B"]}
    )

    matrix = NestedMatrix.from_connectivity(connectivity, annotations)
    original_sum = matrix.sum_type_matrix.copy()
    original_mean = matrix.mean_type_matrix.copy()

    with pytest.raises(ValueError, match="immutable"):
        matrix.matrix.loc["1", "3"] = 99

    assert matrix.matrix.loc["1", "3"] == 4.0
    pd.testing.assert_frame_equal(
        matrix.sum_type_matrix, original_sum, check_frame_type=False
    )
    pd.testing.assert_frame_equal(
        matrix.mean_type_matrix, original_mean, check_frame_type=False
    )


def test_cached_type_matrix_views_are_read_only() -> None:
    connectivity = pd.DataFrame({"pre": [1], "post": [2], "weight": [4]})
    annotations = pd.DataFrame({"root_id": [1, 2], "cell_type": ["A", "B"]})
    matrix = NestedMatrix.from_connectivity(connectivity, annotations)

    with pytest.raises(ValueError, match="immutable"):
        matrix.sum_type_matrix.loc["A", "B"] = 99

    with pytest.raises(ValueError, match="immutable"):
        matrix.mean_type_matrix.loc["A", "B"] = 99

    assert matrix.sum_type_matrix.loc["A", "B"] == 4.0
    assert matrix.mean_type_matrix.loc["A", "B"] == 4.0


def test_public_metadata_views_are_read_only() -> None:
    adjacency = pd.DataFrame([[0, 4], [2, 0]], index=[1, 2], columns=[1, 2])
    annotations = pd.DataFrame({"root_id": [1, 2], "cell_type": ["A", "B"]})
    matrix = NestedMatrix.from_connectivity(adjacency, annotations)

    with pytest.raises(TypeError):
        matrix.type_boundaries["A"] = (0, 2)  # type: ignore[index]

    with pytest.raises(TypeError):
        matrix.neuron_to_type["1"] = "B"  # type: ignore[index]

    with pytest.raises(AttributeError):
        matrix.ordered_neurons.append("3")  # type: ignore[attr-defined]


def test_constructor_inputs_are_copied_before_freezing() -> None:
    source_matrix = pd.DataFrame(
        [[0, 4], [2, 0]],
        index=["1", "2"],
        columns=["1", "2"],
    )
    type_boundaries = {"A": (0, 1), "B": (1, 2)}
    ordered_neurons = ["1", "2"]
    neuron_to_type = {"1": "A", "2": "B"}

    matrix = NestedMatrix(
        matrix=source_matrix,
        type_boundaries=type_boundaries,
        ordered_neurons=ordered_neurons,
        neuron_to_type=neuron_to_type,
    )

    source_matrix.loc["1", "2"] = 99
    type_boundaries["A"] = (0, 2)
    ordered_neurons.append("3")
    neuron_to_type["1"] = "B"

    assert matrix.matrix.loc["1", "2"] == 4.0
    assert matrix.type_boundaries["A"] == (0, 1)
    assert matrix.ordered_neurons == ("1", "2")
    assert matrix.neuron_to_type["1"] == "A"


def test_constructor_normalizes_numeric_type_metadata() -> None:
    matrix = NestedMatrix(
        matrix=pd.DataFrame([[0]], index=[1.0], columns=[1.0]),
        type_boundaries={1.0: (0, 1)},
        ordered_neurons=[1],
        neuron_to_type={1: 1.0},
    )

    assert matrix.type_boundaries["1"] == (0, 1)
    assert matrix.neuron_to_type["1"] == "1"


def test_public_matrix_copy_can_be_mutated_without_affecting_nested_matrix() -> None:
    adjacency = pd.DataFrame([[0, 4], [2, 0]], index=["1", "2"], columns=["1", "2"])
    annotations = pd.DataFrame({"root_id": [1, 2], "cell_type": ["A", "B"]})
    matrix = NestedMatrix.from_connectivity(adjacency, annotations)

    mutable_copy = matrix.matrix.copy()
    mutable_copy.loc["1", "2"] = 99

    assert mutable_copy.loc["1", "2"] == 99
    assert matrix.matrix.loc["1", "2"] == 4.0


def test_from_synapses_derives_relative_outgoing_weight_from_counts() -> None:
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "post_pt_root_id": [3, 4, 4, 4, 3, 3, 4, 4],
            "size": [100, 1, 1, 1, 1, 1, 50, 50],
        }
    )
    annotations = pd.DataFrame(
        {"root_id": [1, 2, 3, 4], "cell_type": ["ER", "ER", "PBt", "PBt"]}
    )

    matrix = NestedMatrix.from_synapses(
        synapses,
        annotations,
        weight_mode="relative_outgoing",
    )

    assert matrix.matrix.loc["1", "3"] == pytest.approx(0.25)
    assert matrix.matrix.loc["1", "4"] == pytest.approx(0.75)
    assert matrix.matrix.loc["2", "3"] == pytest.approx(0.5)
    assert matrix.matrix.loc["2", "4"] == pytest.approx(0.5)


def test_from_synapses_deduplicates_ids_before_relative_outgoing_weight_counts() -> (
    None
):
    synapses = pd.DataFrame(
        {
            "id": [10, 11, 11],
            "pre_pt_root_id": [1, 1, 1],
            "post_pt_root_id": [3, 4, 4],
        }
    )
    annotations = pd.DataFrame(
        {"root_id": [1, 3, 4], "cell_type": ["ER", "PBt", "PBt"]}
    )

    matrix = NestedMatrix.from_synapses(
        synapses,
        annotations,
        weight_mode="relative_outgoing",
    )

    assert matrix.matrix.loc["1", "3"] == pytest.approx(0.5)
    assert matrix.matrix.loc["1", "4"] == pytest.approx(0.5)


def test_from_synapses_relative_outgoing_ignores_existing_weight_columns() -> None:
    synapses = pd.DataFrame(
        {
            "id": [10, 11, 11],
            "pre_pt_root_id": [1, 1, 1],
            "post_pt_root_id": [3, 4, 4],
            "relative_weight": [0.9, 0.1, 0.1],
        }
    )
    annotations = pd.DataFrame(
        {"root_id": [1, 3, 4], "cell_type": ["ER", "PBt", "PBt"]}
    )

    matrix = NestedMatrix.from_synapses(
        synapses,
        annotations,
        weight_mode="relative_outgoing",
    )

    assert matrix.matrix.loc["1", "3"] == pytest.approx(0.5)
    assert matrix.matrix.loc["1", "4"] == pytest.approx(0.5)


def test_from_synapses_filters_to_annotated_neurons_before_relative_outgoing_counts() -> (
    None
):
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 1],
            "post_pt_root_id": [3, 99],
        }
    )
    annotations = pd.DataFrame({"root_id": [1, 3], "cell_type": ["ER", "PBt"]})

    matrix = NestedMatrix.from_synapses(
        synapses,
        annotations,
        weight_mode="relative_outgoing",
    )

    assert list(matrix.matrix.index) == ["1", "3"]
    assert matrix.matrix.loc["1", "3"] == pytest.approx(1.0)


def test_from_synapses_treats_missing_type_labels_as_unassigned() -> None:
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [1],
            "post_pt_root_id": [2],
        }
    )
    annotations = pd.DataFrame({"root_id": [1, 2], "cell_type": ["ER", np.nan]})

    matrix = NestedMatrix.from_synapses(
        synapses,
        annotations,
        weight_mode="relative_outgoing",
    )

    assert list(matrix.matrix.index) == ["1", "2"]
    assert list(matrix.type_boundaries.keys()) == ["ER"]
    assert matrix.matrix.loc["1", "2"] == pytest.approx(1.0)


def test_from_synapses_with_roi_labels_still_builds_one_matrix() -> None:
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 1],
            "post_pt_root_id": [3, 3],
            "roi": ["pb", "eb"],
            "size": [2, 5],
        }
    )
    annotations = pd.DataFrame({"root_id": [1, 3], "cell_type": ["ER", "PBt"]})

    matrix = NestedMatrix.from_synapses(
        synapses,
        annotations,
        weight_mode="column",
        weight_column="size",
    )

    assert matrix.matrix.loc["1", "3"] == 7.0


def test_from_synapses_derives_relative_incoming_weight_from_counts() -> None:
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 1, 1, 2],
            "post_pt_root_id": [3, 3, 3, 3],
        }
    )
    annotations = pd.DataFrame({"root_id": [1, 2, 3], "cell_type": ["ER", "ER", "PBt"]})

    matrix = NestedMatrix.from_synapses(
        synapses,
        annotations,
        weight_mode="relative_incoming",
    )

    assert matrix.matrix.loc["1", "3"] == pytest.approx(0.75)
    assert matrix.matrix.loc["2", "3"] == pytest.approx(0.25)


def test_from_synapses_count_mode_uses_raw_pair_counts() -> None:
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 1, 2],
            "post_pt_root_id": [3, 3, 3],
        }
    )
    annotations = pd.DataFrame({"root_id": [1, 2, 3], "cell_type": ["ER", "ER", "PBt"]})

    matrix = NestedMatrix.from_synapses(
        synapses,
        annotations,
        weight_mode="count",
    )

    assert matrix.matrix.loc["1", "3"] == 2.0
    assert matrix.matrix.loc["2", "3"] == 1.0


def test_from_synapses_annotation_scope_all_keeps_unannotated_neurons() -> None:
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 1],
            "post_pt_root_id": [3, 99],
        }
    )
    annotations = pd.DataFrame({"root_id": [1, 3], "cell_type": ["ER", "PBt"]})

    matrix = NestedMatrix.from_synapses(
        synapses,
        annotations,
        weight_mode="relative_outgoing",
        annotation_scope="all",
    )

    assert list(matrix.matrix.index) == ["1", "3", "99"]
    assert list(matrix.matrix.columns) == ["1", "3", "99"]
    assert matrix.matrix.loc["1", "3"] == pytest.approx(0.5)
    assert matrix.matrix.loc["1", "99"] == pytest.approx(0.5)


def test_from_synapses_annotation_scope_all_normalizes_before_relative_weights() -> (
    None
):
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [1, "1"],
            "post_pt_root_id": [3, 4],
        }
    )
    annotations = pd.DataFrame(
        {"root_id": [1, 3, 4], "cell_type": ["ER", "PBt", "PBt"]}
    )

    matrix = NestedMatrix.from_synapses(
        synapses,
        annotations,
        weight_mode="relative_outgoing",
        annotation_scope="all",
    )

    assert matrix.matrix.loc["1", "3"] == pytest.approx(0.5)
    assert matrix.matrix.loc["1", "4"] == pytest.approx(0.5)
    assert matrix.matrix.loc["1"].sum() == pytest.approx(1.0)


def test_from_synapses_column_mode_requires_weight_column() -> None:
    synapses = pd.DataFrame(
        {"pre_pt_root_id": [1], "post_pt_root_id": [2], "size": [3]}
    )
    annotations = pd.DataFrame({"root_id": [1, 2], "cell_type": ["ER", "PBt"]})

    with pytest.raises(ValueError, match="weight_column must be provided"):
        NestedMatrix.from_synapses(
            synapses,
            annotations,
            weight_mode="column",
        )


def test_from_synapses_rejects_weight_column_for_non_column_modes() -> None:
    synapses = pd.DataFrame(
        {"pre_pt_root_id": [1], "post_pt_root_id": [2], "size": [3]}
    )
    annotations = pd.DataFrame({"root_id": [1, 2], "cell_type": ["ER", "PBt"]})

    with pytest.raises(ValueError, match="weight_column can only be provided"):
        NestedMatrix.from_synapses(
            synapses,
            annotations,
            weight_mode="count",
            weight_column="size",
        )


def test_plot_uses_default_vivid_colormap() -> None:
    adjacency = pd.DataFrame([[0, 4], [2, 0]], index=[1, 2], columns=[1, 2])
    annotations = pd.DataFrame({"root_id": [1, 2], "cell_type": ["ER", "PBt"]})
    matrix = NestedMatrix.from_connectivity(adjacency, annotations)

    fig, ax = matrix.plot(level="type_mean")

    assert ax.images[0].cmap(1.0) == to_rgba("#2E0054")
    plt.close(fig)


def test_plot_colorbar_label_matches_level() -> None:
    adjacency = pd.DataFrame([[0, 4], [2, 0]], index=[1, 2], columns=[1, 2])
    annotations = pd.DataFrame({"root_id": [1, 2], "cell_type": ["ER", "PBt"]})
    matrix = NestedMatrix.from_connectivity(adjacency, annotations)

    fig_type_mean, ax_type_mean = matrix.plot(level="type_mean")
    fig_type_sum, ax_type_sum = matrix.plot(level="type_sum")
    fig_neuron, ax_neuron = matrix.plot(level="neuron")

    assert fig_type_mean.axes[1].get_ylabel() == "Mean Weight"
    assert fig_type_sum.axes[1].get_ylabel() == "Total Weight"
    assert fig_neuron.axes[1].get_ylabel() == "Weight"

    plt.close(fig_type_mean)
    plt.close(fig_type_sum)
    plt.close(fig_neuron)


def test_type_mean_plot_uses_mean_type_matrix() -> None:
    connectivity = pd.DataFrame(
        {
            "pre": [1, 2, 3],
            "post": [3, 2, 4],
            "weight": [4, 0, 2],
        }
    )
    annotations = pd.DataFrame(
        {"root_id": [1, 2, 3, 4], "cell_type": ["A", "A", "B", "B"]}
    )
    matrix = NestedMatrix.from_connectivity(connectivity, annotations)

    fig, ax = matrix.plot(level="type_mean")

    plotted = np.asarray(ax.images[0].get_array())
    np.testing.assert_allclose(plotted, matrix.mean_type_matrix.values)
    plt.close(fig)


def test_type_sum_plot_uses_type_matrix() -> None:
    connectivity = pd.DataFrame(
        {
            "pre": [1, 2, 3],
            "post": [3, 2, 4],
            "weight": [4, 0, 2],
        }
    )
    annotations = pd.DataFrame(
        {"root_id": [1, 2, 3, 4], "cell_type": ["A", "A", "B", "B"]}
    )
    matrix = NestedMatrix.from_connectivity(connectivity, annotations)

    fig, ax = matrix.plot(level="type_sum")

    plotted = np.asarray(ax.images[0].get_array())
    np.testing.assert_allclose(plotted, matrix.sum_type_matrix.values)
    plt.close(fig)


def test_top_level_import_star_exposes_nested_connectivity_module() -> None:
    namespace = {}

    exec("from crantpy import *", namespace)

    assert namespace["nested_connectivity_matrices"].NestedMatrix is NestedMatrix
    assert namespace["DirectedNestedMatrix"] is DirectedNestedMatrix


def test_plot_accepts_bare_output_filename(tmp_path: Path) -> None:
    adjacency = pd.DataFrame([[0, 4], [2, 0]], index=[1, 2], columns=[1, 2])
    annotations = pd.DataFrame({"root_id": [1, 2], "cell_type": ["ER", "PBt"]})
    matrix = NestedMatrix.from_connectivity(adjacency, annotations)

    cwd = os.getcwd()
    fig = None
    os.chdir(tmp_path)
    try:
        fig, _ = matrix.plot(level="type_mean", output_path="connectivity.png")
    finally:
        os.chdir(cwd)
        if fig is not None:
            plt.close(fig)

    assert (tmp_path / "connectivity.png").exists()


def test_plot_keeps_centered_type_labels_for_small_type_sets() -> None:
    adjacency = pd.DataFrame(
        [[0, 4, 1], [2, 0, 0], [0, 3, 0]],
        index=[1, 2, 3],
        columns=[1, 2, 3],
    )
    annotations = pd.DataFrame({"root_id": [1, 2, 3], "cell_type": ["ER", "ER", "PBt"]})
    matrix = NestedMatrix.from_connectivity(adjacency, annotations)

    fig, ax = matrix.plot(level="neuron", min_neurons_for_plot=1)

    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["ER", "PBt"]
    assert [tick.get_text() for tick in ax.get_yticklabels()] == ["ER", "PBt"]

    plt.close(fig)


def test_plot_linewidth_scale_preserves_and_scales_default_boundaries() -> None:
    adjacency = pd.DataFrame(
        [[0, 4, 1], [2, 0, 0], [0, 3, 0]],
        index=[1, 2, 3],
        columns=[1, 2, 3],
    )
    annotations = pd.DataFrame({"root_id": [1, 2, 3], "cell_type": ["ER", "ER", "PBt"]})
    matrix = NestedMatrix.from_connectivity(adjacency, annotations)

    fig_default, ax_default = matrix.plot(level="neuron", min_neurons_for_plot=1)
    fig_scaled, ax_scaled = matrix.plot(
        level="neuron",
        min_neurons_for_plot=1,
        linewidth_scale=0.5,
    )

    assert _line_collection_widths(ax_default) == pytest.approx(
        [3.0, 4.0, 2.0, 2.0, 2.5]
    )
    assert _line_collection_widths(ax_scaled) == pytest.approx(
        [1.5, 2.0, 1.0, 1.0, 1.25]
    )

    plt.close(fig_default)
    plt.close(fig_scaled)


def test_plot_rejects_non_positive_linewidth_scale() -> None:
    adjacency = pd.DataFrame([[0, 4], [2, 0]], index=[1, 2], columns=[1, 2])
    annotations = pd.DataFrame({"root_id": [1, 2], "cell_type": ["ER", "PBt"]})
    matrix = NestedMatrix.from_connectivity(adjacency, annotations)

    with pytest.raises(ValueError, match="linewidth_scale must be positive"):
        matrix.plot(level="type_mean", linewidth_scale=0)


def test_plot_rejects_invalid_level() -> None:
    adjacency = pd.DataFrame([[0, 4], [2, 0]], index=[1, 2], columns=[1, 2])
    annotations = pd.DataFrame({"root_id": [1, 2], "cell_type": ["ER", "PBt"]})
    matrix = NestedMatrix.from_connectivity(adjacency, annotations)

    with pytest.raises(ValueError, match="level must be one of"):
        matrix.plot(level="type_detail")  # type: ignore[arg-type]


def test_plot_suppresses_dense_centered_type_labels_and_batches_boundaries() -> None:
    n_types = 400
    neuron_ids = [str(i) for i in range(n_types)]
    adjacency = pd.DataFrame(
        np.zeros((n_types, n_types), dtype=float),
        index=neuron_ids,
        columns=neuron_ids,
    )
    annotations = pd.DataFrame(
        {
            "root_id": neuron_ids,
            "cell_type": [f"T{i}" for i in range(n_types)],
        }
    )
    matrix = NestedMatrix.from_connectivity(adjacency, annotations)

    fig, ax = matrix.plot(level="neuron", min_neurons_for_plot=1)

    assert len(ax.get_xticks()) == 0
    assert len(ax.get_yticks()) == 0
    assert len(ax.patches) == 0
    assert len(ax.lines) == 0
    assert len([c for c in ax.collections if isinstance(c, LineCollection)]) == 5

    plt.close(fig)


# ---------------------------------------------------------------------------
# order.within: the EB rule is a visible, overridable default
# ---------------------------------------------------------------------------


def _columnar_annotations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "root_id": ["40", "7", "33"],
            "cell_type": ["EPG/PEG", "EPG/PEG", "EPG/PEG"],
            "cell_subtype": ["EPG/PEG_L8", "EPG/PEG_R1", "EPG/PEG_R2"],
        }
    )


def _columnar_adjacency() -> pd.DataFrame:
    return pd.DataFrame(
        np.zeros((3, 3)),
        index=["40", "7", "33"],
        columns=["40", "7", "33"],
    )


def test_order_within_defaults_to_eb_column_rule() -> None:
    matrix = NestedMatrix.from_connectivity(
        _columnar_adjacency(), _columnar_annotations()
    )

    # R1, L8, R2 is the EB ring order, so 7 -> 40 -> 33 -- which is neither the
    # annotation row order (40, 7, 33) nor the numeric ID order (7, 33, 40).
    assert list(matrix.ordered_neurons) == ["7", "40", "33"]


def test_order_within_none_keeps_annotation_row_order() -> None:
    matrix = NestedMatrix.from_connectivity(
        _columnar_adjacency(), _columnar_annotations(), order={"within": None}
    )

    assert list(matrix.ordered_neurons) == ["40", "7", "33"]


def test_order_within_annotation_string_matches_none() -> None:
    matrix = NestedMatrix.from_connectivity(
        _columnar_adjacency(),
        _columnar_annotations(),
        order={"within": "annotation"},
    )

    assert list(matrix.ordered_neurons) == ["40", "7", "33"]


def test_order_within_id_sorts_numerically() -> None:
    matrix = NestedMatrix.from_connectivity(
        _columnar_adjacency(), _columnar_annotations(), order={"within": "id"}
    )

    assert list(matrix.ordered_neurons) == ["7", "33", "40"]


def test_order_within_accepts_label_sequence_shorthand() -> None:
    matrix = NestedMatrix.from_connectivity(
        _columnar_adjacency(),
        _columnar_annotations(),
        order={"within": {"EPG/PEG": ["R2", "L8", "R1"]}},
    )

    assert list(matrix.ordered_neurons) == ["33", "40", "7"]


def test_order_within_accepts_callable() -> None:
    def reverse_annotation_order(type_name, type_rows, neuron_id_column):
        assert type_name == "EPG/PEG"
        return list(reversed(type_rows[neuron_id_column].tolist()))

    matrix = NestedMatrix.from_connectivity(
        _columnar_adjacency(),
        _columnar_annotations(),
        order={"within": reverse_annotation_order},
    )

    assert list(matrix.ordered_neurons) == ["33", "7", "40"]


def test_order_within_mapping_only_affects_named_types() -> None:
    adjacency = pd.DataFrame(
        np.zeros((4, 4)), index=["1", "2", "3", "4"], columns=["1", "2", "3", "4"]
    )
    annotations = pd.DataFrame(
        {
            "root_id": ["3", "1", "4", "2"],
            "cell_type": ["A", "A", "B", "B"],
        }
    )

    matrix = NestedMatrix.from_connectivity(
        adjacency, annotations, order={"within": {"A": "id"}}
    )

    # "A" sorts by ID, "B" keeps annotation row order.
    assert list(matrix.ordered_neurons) == ["1", "3", "4", "2"]


def test_order_within_extends_the_documented_default() -> None:
    annotations = pd.DataFrame(
        {
            "root_id": ["10", "2", "5", "6"],
            "cell_type": ["EPG/PEG", "EPG/PEG", "PEN", "PEN"],
            "cell_subtype": ["EPG/PEG_L8", "EPG/PEG_R1", "PEN_L1", "PEN_R1"],
        }
    )
    adjacency = pd.DataFrame(
        np.zeros((4, 4)), index=["10", "2", "5", "6"], columns=["10", "2", "5", "6"]
    )

    matrix = NestedMatrix.from_connectivity(
        adjacency,
        annotations,
        order=NeuronOrder(
            types=["EPG/PEG", "PEN"],
            within={**DEFAULT_WITHIN_TYPE_ORDER, "PEN": ["R1", "L1"]},
        ),
    )

    # EPG/PEG still uses the default EB order; PEN uses the added rule.
    assert list(matrix.ordered_neurons) == ["2", "10", "6", "5"]


def test_order_within_rejects_unknown_string() -> None:
    with pytest.raises(ValueError, match="'annotation' or 'id'"):
        NestedMatrix.from_connectivity(
            _columnar_adjacency(), _columnar_annotations(), order={"within": "alpha"}
        )


def test_order_within_rejects_sorter_that_drops_neurons() -> None:
    with pytest.raises(ValueError, match="exactly once"):
        NestedMatrix.from_connectivity(
            _columnar_adjacency(),
            _columnar_annotations(),
            order={"within": lambda name, rows, id_col: rows[id_col].tolist()[:1]},
        )


def test_directed_within_type_order_can_differ_per_axis() -> None:
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": ["10", "2", "10", "2"],
            "post_pt_root_id": ["30", "40", "40", "30"],
        }
    )
    annotations = pd.DataFrame(
        {
            "root_id": ["10", "2", "40", "30"],
            "cell_type": ["A", "A", "B", "B"],
        }
    )

    matrix = DirectedNestedMatrix.from_synapses(
        synapses,
        annotations,
        weight_mode="count",
        source_order={"within": "id"},
        target_order={"within": None},
    )

    # Source sorts by ID; the target keeps annotation row order (40 before 30),
    # so neither assertion holds if the two axes share one rule.
    assert list(matrix.source_neurons) == ["2", "10"]
    assert list(matrix.target_neurons) == ["40", "30"]


def test_column_order_rule_normalizes_its_fields() -> None:
    rule = ColumnOrderRule(order=["R1", "L1", "R2"])

    assert rule.order == ("R1", "L1", "R2")
    assert rule.label_columns == ("cell_instance", "cell_subtype")


def test_eb_column_order_comes_from_config() -> None:
    from crantpy.utils.config import EB_COLUMN_LABELS, EB_COLUMNAR_CELL_TYPES

    # Spelled out literally: comparing against the constant it is built from
    # would move with any edit to it, leaving the anatomical ring unpinned.
    assert list(EB_COLUMN_ORDER.order) == [
        "R1",
        "L8",
        "R2",
        "L7",
        "R3",
        "L6",
        "R4",
        "L5",
        "R5",
        "L4",
        "R6",
        "L3",
        "R7",
        "L2",
        "R8",
        "L1",
    ]
    assert list(EB_COLUMN_LABELS) == list(EB_COLUMN_ORDER.order)
    assert tuple(EB_COLUMNAR_CELL_TYPES) == ("EPG/PEG",)
    assert list(DEFAULT_WITHIN_TYPE_ORDER) == ["EPG/PEG"]


# ---------------------------------------------------------------------------
# typed / untyped neurons are visible instead of silently appended
# ---------------------------------------------------------------------------


def test_typed_and_untyped_neurons_partition_the_axis() -> None:
    adjacency = pd.DataFrame(
        np.zeros((3, 3)), index=["1", "2", "9"], columns=["1", "2", "9"]
    )
    annotations = pd.DataFrame({"root_id": ["1", "2"], "cell_type": ["ER", "ER"]})

    matrix = NestedMatrix.from_connectivity(
        adjacency, annotations, annotation_scope="all"
    )

    assert matrix.typed_neurons == ("1", "2")
    assert matrix.untyped_neurons == ("9",)
    assert matrix.typed_neurons + matrix.untyped_neurons == matrix.ordered_neurons
    # The boundaries cover the typed neurons only.
    assert matrix.type_boundaries["ER"] == (0, 2)
    assert len(matrix.typed_neurons) == 2
    assert "9" not in matrix.neuron_to_type


def test_untyped_neurons_empty_when_everything_is_annotated() -> None:
    adjacency = pd.DataFrame(np.zeros((2, 2)), index=["1", "2"], columns=["1", "2"])
    annotations = pd.DataFrame({"root_id": ["1", "2"], "cell_type": ["ER", "ER"]})

    matrix = NestedMatrix.from_connectivity(adjacency, annotations)

    assert matrix.untyped_neurons == ()
    assert matrix.typed_neurons == matrix.ordered_neurons


def test_directed_typed_and_untyped_neurons_are_per_axis() -> None:
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": ["1", "9"],
            "post_pt_root_id": ["3", "3"],
        }
    )
    annotations = pd.DataFrame({"root_id": ["1", "3"], "cell_type": ["A", "B"]})

    matrix = DirectedNestedMatrix.from_synapses(
        synapses, annotations, weight_mode="count", annotation_scope="all"
    )

    assert matrix.source_typed_neurons == ("1",)
    assert matrix.source_untyped_neurons == ("9",)
    assert (
        matrix.source_typed_neurons + matrix.source_untyped_neurons
        == matrix.source_neurons
    )
    assert matrix.target_typed_neurons == ("3",)
    assert matrix.target_untyped_neurons == ()


# ---------------------------------------------------------------------------
# source_ids / target_ids select; source_neurons / target_neurons report
# ---------------------------------------------------------------------------


def test_directed_id_selectors_are_named_source_ids_and_target_ids() -> None:
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": ["1", "2"],
            "post_pt_root_id": ["3", "4"],
        }
    )
    annotations = pd.DataFrame(
        {"root_id": ["1", "2", "3", "4"], "cell_type": ["A", "A", "B", "B"]}
    )

    matrix = DirectedNestedMatrix.from_synapses(
        synapses, annotations, source_ids=["1"], target_ids=["3"], weight_mode="count"
    )

    assert matrix.source_neurons == ("1",)
    assert matrix.target_neurons == ("3",)


def test_directed_retired_selector_names_are_rejected() -> None:
    synapses = pd.DataFrame(
        {"pre_pt_root_id": ["1", "2"], "post_pt_root_id": ["3", "4"]}
    )
    annotations = pd.DataFrame(
        {"root_id": ["1", "2", "3", "4"], "cell_type": ["A", "A", "B", "B"]}
    )

    with pytest.raises(TypeError, match="source_neurons"):
        DirectedNestedMatrix.from_synapses(
            synapses,
            annotations,
            source_neurons=["1"],
            weight_mode="count",
        )


def test_directed_constructor_still_takes_the_resolved_order() -> None:
    # __init__ takes the resolved order, not a selector -- unchanged.
    matrix = DirectedNestedMatrix(
        matrix=pd.DataFrame([[1.0]], index=["1"], columns=["3"]),
        source_neurons=["1"],
        target_neurons=["3"],
    )

    assert matrix.source_neurons == ("1",)


# ---------------------------------------------------------------------------
# order.types: the block level has the same vocabulary as order.within
# ---------------------------------------------------------------------------


def _two_block_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    annotations = pd.DataFrame(
        {
            "root_id": ["1", "2", "3", "4"],
            "cell_type": ["A", "B", "B", "B"],
        }
    )
    adjacency = pd.DataFrame(
        np.zeros((4, 4)), index=["1", "2", "3", "4"], columns=["1", "2", "3", "4"]
    )
    return adjacency, annotations


def test_order_types_defaults_to_label_sort() -> None:
    adjacency, annotations = _two_block_inputs()

    matrix = NestedMatrix.from_connectivity(adjacency, annotations)

    # B is the larger block, so this pins "label" rather than "size".
    assert list(matrix.type_boundaries) == ["A", "B"]


def test_order_types_size_puts_largest_block_first() -> None:
    adjacency, annotations = _two_block_inputs()

    by_size = NestedMatrix.from_connectivity(
        adjacency, annotations, order={"types": "size"}
    )
    assert list(by_size.type_boundaries) == ["B", "A"]

    # Flip the sizes and the order flips with them.
    annotations = annotations.assign(cell_type=["B", "A", "A", "A"])
    by_size = NestedMatrix.from_connectivity(
        adjacency, annotations, order={"types": "size"}
    )
    assert list(by_size.type_boundaries) == ["A", "B"]


def test_order_types_accepts_callable() -> None:
    adjacency, annotations = _two_block_inputs()

    matrix = NestedMatrix.from_connectivity(
        adjacency,
        annotations,
        order={"types": lambda names, rows, col: sorted(names, reverse=True)},
    )

    assert list(matrix.type_boundaries) == ["B", "A"]


def test_order_types_rejects_unknown_string() -> None:
    adjacency, annotations = _two_block_inputs()

    with pytest.raises(ValueError, match="'label' or 'size'"):
        NestedMatrix.from_connectivity(
            adjacency, annotations, order={"types": "alphabetical"}
        )


def test_order_types_rejects_sorter_that_drops_a_block() -> None:
    adjacency, annotations = _two_block_inputs()

    with pytest.raises(ValueError, match="exactly once"):
        NestedMatrix.from_connectivity(
            adjacency, annotations, order={"types": lambda names, rows, col: names[:1]}
        )


# ---------------------------------------------------------------------------
# order= coercion
# ---------------------------------------------------------------------------


def test_order_bare_sequence_is_types_shorthand() -> None:
    adjacency, annotations = _two_block_inputs()

    shorthand = NestedMatrix.from_connectivity(adjacency, annotations, order=["B", "A"])
    explicit = NestedMatrix.from_connectivity(
        adjacency, annotations, order=NeuronOrder(types=["B", "A"])
    )

    assert list(shorthand.type_boundaries) == ["B", "A"]
    assert shorthand.ordered_neurons == explicit.ordered_neurons


def test_order_mapping_and_neuron_order_agree() -> None:
    annotations = pd.DataFrame(
        {"root_id": ["30", "4", "200", "1"], "cell_type": ["A", "A", "A", "B"]}
    )
    adjacency = pd.DataFrame(
        np.zeros((4, 4)), index=annotations.root_id, columns=annotations.root_id
    )
    spec = {"types": ["B", "A"], "within": "id"}

    as_mapping = NestedMatrix.from_connectivity(adjacency, annotations, order=spec)
    as_object = NestedMatrix.from_connectivity(
        adjacency, annotations, order=NeuronOrder(**spec)
    )
    default = NestedMatrix.from_connectivity(adjacency, annotations)

    assert as_mapping.ordered_neurons == as_object.ordered_neurons
    # ...and both differ from the default, so this cannot pass if order= is ignored.
    assert as_mapping.ordered_neurons == ("1", "4", "30", "200")
    assert default.ordered_neurons == ("30", "4", "200", "1")


def test_order_none_is_the_default_order() -> None:
    adjacency, annotations = _two_block_inputs()

    assert (
        NestedMatrix.from_connectivity(
            adjacency, annotations, order=None
        ).ordered_neurons
        == NestedMatrix.from_connectivity(adjacency, annotations).ordered_neurons
    )


def test_order_mapping_rejects_unknown_keys() -> None:
    adjacency, annotations = _two_block_inputs()

    with pytest.raises(ValueError, match="only 'types' and 'within'"):
        NestedMatrix.from_connectivity(
            adjacency, annotations, order={"EPG/PEG": ["R1", "L1"]}
        )


def test_order_bare_string_is_rejected_as_ambiguous() -> None:
    adjacency, annotations = _two_block_inputs()

    with pytest.raises(TypeError, match="ambiguous"):
        NestedMatrix.from_connectivity(adjacency, annotations, order="id")


def test_order_partial_mapping_keeps_the_other_default() -> None:
    from crantpy.utils.ordering import DEFAULT_ORDER, as_neuron_order

    assert as_neuron_order({"types": "size"}).within == DEFAULT_ORDER.within
    assert as_neuron_order({"within": None}).types == DEFAULT_ORDER.types
    assert as_neuron_order({"within": None}).within is None


def test_retired_ordering_arguments_are_rejected() -> None:
    adjacency, annotations = _two_block_inputs()
    synapses = pd.DataFrame(
        {"pre_pt_root_id": ["1", "2"], "post_pt_root_id": ["3", "4"]}
    )

    with pytest.raises(TypeError, match="type_order"):
        NestedMatrix.from_connectivity(adjacency, annotations, type_order=["B", "A"])
    with pytest.raises(TypeError, match="source_type_order"):
        DirectedNestedMatrix.from_synapses(
            synapses, annotations, weight_mode="count", source_type_order=["B", "A"]
        )
    with pytest.raises(TypeError, match="target_type_order"):
        DirectedNestedMatrix.from_synapses(
            synapses, annotations, weight_mode="count", target_type_order=["B", "A"]
        )


def test_untyped_neurons_include_null_types_under_default_scope() -> None:
    # "9" has an annotation row but no cell type, so annotated_only keeps it.
    adjacency = pd.DataFrame(
        np.zeros((3, 3)), index=["1", "2", "9"], columns=["1", "2", "9"]
    )
    annotations = pd.DataFrame(
        {"root_id": ["1", "2", "9"], "cell_type": ["ER", "ER", None]}
    )

    matrix = NestedMatrix.from_connectivity(adjacency, annotations)

    assert matrix.untyped_neurons == ("9",)
    assert matrix.typed_neurons == ("1", "2")


def test_partial_within_mapping_overlays_the_default_rules() -> None:
    annotations = pd.DataFrame(
        {
            "root_id": ["10", "2", "5", "6"],
            "cell_type": ["EPG/PEG", "EPG/PEG", "PEN", "PEN"],
            "cell_subtype": ["EPG/PEG_L8", "EPG/PEG_R1", "PEN_L1", "PEN_R1"],
        }
    )
    adjacency = pd.DataFrame(
        np.zeros((4, 4)), index=annotations.root_id, columns=annotations.root_id
    )

    matrix = NestedMatrix.from_connectivity(
        adjacency,
        annotations,
        order={"types": ["EPG/PEG", "PEN"], "within": {"PEN": ["R1", "L1"]}},
    )

    # PEN uses the added rule; EPG/PEG keeps the built-in EB order (R1 before L8).
    assert list(matrix.ordered_neurons) == ["2", "10", "6", "5"]


def test_within_mapping_can_override_a_default_rule_by_name() -> None:
    annotations = pd.DataFrame(
        {
            "root_id": ["10", "2"],
            "cell_type": ["EPG/PEG", "EPG/PEG"],
            "cell_subtype": ["EPG/PEG_L8", "EPG/PEG_R1"],
        }
    )
    adjacency = pd.DataFrame(
        np.zeros((2, 2)), index=annotations.root_id, columns=annotations.root_id
    )

    overridden = NestedMatrix.from_connectivity(
        adjacency, annotations, order={"within": {"EPG/PEG": "annotation"}}
    )
    disabled = NestedMatrix.from_connectivity(
        adjacency, annotations, order={"within": None}
    )

    assert list(overridden.ordered_neurons) == ["10", "2"]
    assert list(disabled.ordered_neurons) == ["10", "2"]


def _ab_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    annotations = pd.DataFrame(
        {"root_id": ["1", "2", "3", "4"], "cell_type": ["B", "B", "A", "A"]}
    )
    adjacency = pd.DataFrame(
        np.zeros((4, 4)), index=annotations.root_id, columns=annotations.root_id
    )
    return adjacency, annotations


def test_order_accepts_a_one_shot_iterable() -> None:
    adjacency, annotations = _ab_inputs()

    from_generator = NestedMatrix.from_connectivity(
        adjacency, annotations, order=(t for t in ["B", "A"])
    )
    from_list = NestedMatrix.from_connectivity(adjacency, annotations, order=["B", "A"])

    assert list(from_generator.type_boundaries) == ["B", "A"]
    assert from_generator.ordered_neurons == from_list.ordered_neurons


def test_neuron_order_snapshots_a_one_shot_types_rule() -> None:
    order = NeuronOrder(types=reversed(["A", "B"]))

    # The iterator is materialized at construction, not on first use.
    assert order.types == ("B", "A")

    adjacency, annotations = _ab_inputs()
    first = NestedMatrix.from_connectivity(adjacency, annotations, order=order)
    second = NestedMatrix.from_connectivity(adjacency, annotations, order=order)

    # Reusing the same NeuronOrder must give the same answer both times.
    assert list(first.type_boundaries) == ["B", "A"]
    assert list(second.type_boundaries) == ["B", "A"]


def test_neuron_order_snapshots_a_one_shot_within_rule() -> None:
    order = NeuronOrder(within={"EPG/PEG": iter(["R2", "L8", "R1"])})

    assert order.within["EPG/PEG"] == ("R2", "L8", "R1")

    first = NestedMatrix.from_connectivity(
        _columnar_adjacency(), _columnar_annotations(), order=order
    )
    second = NestedMatrix.from_connectivity(
        _columnar_adjacency(), _columnar_annotations(), order=order
    )

    assert list(first.ordered_neurons) == ["33", "40", "7"]
    assert first.ordered_neurons == second.ordered_neurons


def test_directed_one_shot_order_reaches_both_axes() -> None:
    _, annotations = _ab_inputs()
    synapses = pd.DataFrame(
        {"pre_pt_root_id": ["1", "3"], "post_pt_root_id": ["2", "4"]}
    )

    matrix = DirectedNestedMatrix.from_synapses(
        synapses, annotations, weight_mode="count", order=(t for t in ["B", "A"])
    )

    # The source axis must not drain the spec before the target axis sees it.
    assert list(matrix.source_type_boundaries) == ["B", "A"]
    assert list(matrix.target_type_boundaries) == ["B", "A"]


def test_directed_one_shot_within_mapping_reaches_both_axes() -> None:
    synapses = pd.DataFrame(
        {"pre_pt_root_id": ["1", "2"], "post_pt_root_id": ["3", "4"]}
    )
    annotations = pd.DataFrame(
        {
            "root_id": ["1", "2", "3", "4"],
            "cell_type": ["A", "A", "A", "A"],
            "cell_subtype": ["A_R1", "A_L1", "A_L1", "A_R1"],
        }
    )

    matrix = DirectedNestedMatrix.from_synapses(
        synapses,
        annotations,
        weight_mode="count",
        order={"within": {"A": (label for label in ["R1", "L1"])}},
    )

    assert matrix.source_neurons == ("1", "2")
    assert matrix.target_neurons == ("4", "3")


@pytest.mark.parametrize(
    "make_order",
    [
        lambda: {"B", "A"},
        lambda: NeuronOrder(types={"B", "A"}),
        lambda: {"within": {"A": {"R1", "L1"}}},
    ],
)
def test_order_rejects_unordered_set_rules(make_order) -> None:
    adjacency, annotations = _ab_inputs()

    with pytest.raises(TypeError, match="ordered iterable"):
        NestedMatrix.from_connectivity(adjacency, annotations, order=make_order())


def test_column_order_rule_rejects_unordered_sets() -> None:
    with pytest.raises(TypeError, match="ordered iterable"):
        ColumnOrderRule(order={"R1", "L1"})


def test_column_order_rule_rejects_bare_strings() -> None:
    # tuple("R1") == ("R", "1"): the intended label would silently never rank.
    with pytest.raises(TypeError, match="not a bare string"):
        ColumnOrderRule(order="R1")
    with pytest.raises(TypeError, match="not a bare string"):
        ColumnOrderRule(order=["R1"], label_columns="cell_subtype")


def test_within_rejects_a_mapping_nested_under_one_cell_type() -> None:
    adjacency, annotations = _ab_inputs()

    with pytest.raises(TypeError, match="belongs at the top of order.within"):
        NestedMatrix.from_connectivity(
            adjacency, annotations, order={"within": {"B": {"order": ["R1"]}}}
        )


def test_types_rejects_a_mapping() -> None:
    adjacency, annotations = _ab_inputs()

    with pytest.raises(TypeError, match="not a mapping"):
        NestedMatrix.from_connectivity(
            adjacency, annotations, order={"types": {"B": 1}}
        )


# ---------------------------------------------------------------------------
# One-shot selectors survive both axes and every ROI
# ---------------------------------------------------------------------------


def _directed_selector_inputs() -> pd.DataFrame:
    # Neuron 5 is typed "Other" and sits on BOTH axes, so a selector naming only
    # ER_input/ER (or only ids 1-4) has something to actually exclude.
    return pd.DataFrame(
        {
            "pre_pt_root_id": ["1", "2", "5", "1"],
            "post_pt_root_id": ["3", "4", "3", "5"],
        }
    )


def test_directed_one_shot_id_selector_shared_by_both_axes() -> None:
    ids = (i for i in ["1", "2", "3", "4"])

    matrix = DirectedNestedMatrix.from_synapses(
        _directed_selector_inputs(),
        _make_directed_annotations(),
        weight_mode="count",
        source_ids=ids,
        target_ids=ids,
    )

    # The source axis must not drain the selector before the target axis reads
    # it -- and the selector must actually filter: neuron 5 is on both axes and
    # is not named, so ignoring the selector would leave it in.
    assert matrix.source_neurons == ("1", "2")
    assert matrix.target_neurons == ("3", "4")
    assert matrix.matrix.shape == (2, 2)


def test_directed_one_shot_type_selector_shared_by_both_axes() -> None:
    types = (t for t in ["ER_input", "ER"])

    matrix = DirectedNestedMatrix.from_synapses(
        _directed_selector_inputs(),
        _make_directed_annotations(),
        weight_mode="count",
        source_types=types,
        target_types=types,
    )

    # "Other" neuron 5 is excluded from both axes by the shared selector.
    assert matrix.source_neurons == ("1", "2")
    assert matrix.target_neurons == ("3", "4")


def test_directed_one_shot_selector_on_a_single_axis() -> None:
    matrix = DirectedNestedMatrix.from_synapses(
        _directed_selector_inputs(),
        _make_directed_annotations(),
        weight_mode="count",
        source_ids=(i for i in ["1"]),
    )

    # Narrows the source axis to one neuron; the unselected target axis keeps
    # everything available, including the untyped-for-this-purpose neuron 5.
    assert matrix.source_neurons == ("1",)
    assert matrix.target_neurons == ("3", "4", "5")


@patch("crantpy.viz.mesh.load_neuropil_mesh")
def test_directed_one_shot_selector_survives_every_roi(mock_load: MagicMock) -> None:
    # Two ROIs, one synapse each: a drained selector would silently drop the second.
    mock_load.side_effect = [
        _mock_mesh([True, False]),
        _mock_mesh([False, True]),
    ]
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 2],
            "post_pt_root_id": [3, 4],
            "ctr_pt_position": [[100, 200, 300], [110, 210, 310]],
        }
    )

    result = DirectedNestedMatrix.from_synapses_by_neuropil(
        synapses,
        _make_directed_annotations(),
        neuropil_names=["antennal_lobe_left", "antennal_lobe_right"],
        source_types=(t for t in ["ER_input"]),
        target_types=(t for t in ["ER"]),
        coordinates="nm",
        weight_mode="count",
        include_other=False,
    )

    assert set(result) == {"antennal_lobe_left", "antennal_lobe_right"}
    assert list(result["antennal_lobe_left"].matrix.index) == ["1"]
    assert list(result["antennal_lobe_right"].matrix.index) == ["2"]


# ---------------------------------------------------------------------------
# Exported ordering helpers reject sets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda: ordering._sort_cell_types(["A", "B"], preferred={"B"}),
        lambda: ordering._order_types_by_preferred({"B", "A"}),
        lambda: ordering._resolve_type_rule({"B", "A"}),
        lambda: ordering._resolve_within_type_rule({"R1", "L1"}),
        lambda: ordering._resolve_within_type_order({"EPG/PEG": {"R1", "L1"}}),
    ],
    ids=[
        "_sort_cell_types",
        "by_preferred",
        "type_rule",
        "within_rule",
        "within_order",
    ],
)
def test_ordering_helpers_reject_sets(call) -> None:
    with pytest.raises(TypeError, match="ordered iterable|not a set"):
        call()


def test_sort_cell_types_accepts_an_ordered_iterable() -> None:
    assert ordering._sort_cell_types(["A", "B"], preferred=("B",)) == ["B", "A"]


# ---------------------------------------------------------------------------
# Order objects are reusable, copyable and picklable
# ---------------------------------------------------------------------------


def _two_type_axes() -> tuple[pd.DataFrame, pd.DataFrame]:
    annotations = pd.DataFrame(
        {"root_id": ["1", "2", "3", "4"], "cell_type": ["ER1", "PEN", "ER1", "PEN"]}
    )
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": ["1", "2", "3", "4"],
            "post_pt_root_id": ["3", "4", "1", "2"],
        }
    )
    return synapses, annotations


def test_directed_one_shot_order_shared_by_both_axis_arguments() -> None:
    synapses, annotations = _two_type_axes()
    shared = (t for t in ["PEN"])

    matrix = DirectedNestedMatrix.from_synapses(
        synapses,
        annotations,
        weight_mode="count",
        source_order=shared,
        target_order=shared,
    )

    # The source axis must not drain the spec before the target axis reads it.
    assert list(matrix.source_type_boundaries) == ["PEN", "ER1"]
    assert list(matrix.target_type_boundaries) == ["PEN", "ER1"]


def test_order_objects_survive_pickle_and_deepcopy_after_use() -> None:
    import copy
    import pickle

    from crantpy.utils.ordering import DEFAULT_ORDER, EB_COLUMN_ORDER

    # Using the module-level singletons in a build must not make them
    # unpicklable.
    NestedMatrix.from_connectivity(_columnar_adjacency(), _columnar_annotations())

    assert pickle.loads(pickle.dumps(EB_COLUMN_ORDER)) == EB_COLUMN_ORDER
    assert pickle.loads(pickle.dumps(DEFAULT_ORDER)) == DEFAULT_ORDER
    assert copy.deepcopy(DEFAULT_ORDER) == DEFAULT_ORDER

    custom = NeuronOrder(types=["B", "A"], within={"A": ["R1", "L1"]})
    assert pickle.loads(pickle.dumps(custom)) == custom


def test_key_views_are_accepted_at_every_guard() -> None:
    from crantpy.utils import ordering

    labels = {"R1": 0, "L1": 1}.keys()

    # Every guard in the module must treat a key view the same way.
    assert ordering.ColumnOrderRule(order=labels).order == ("R1", "L1")
    assert ordering.ColumnOrderRule(
        order=["R1"], label_columns=labels
    ).label_columns == ("R1", "L1")
    # Assert what they resolve TO: `is not None` would hold for any callable.
    rank_labels = {"R2": 0, "R1": 1, "L1": 2}.keys()
    within_rows = pd.DataFrame(
        {"root_id": ["2", "7", "10"], "cell_subtype": ["A_L1", "A_R1", "A_R2"]}
    )
    assert ordering._resolve_within_type_rule(rank_labels)(
        "A", within_rows, "root_id"
    ) == [
        "10",
        "7",
        "2",
    ]  # ranked R2, R1, L1 -- differs from annotation, numeric-ID and string order

    # A key view naming the cell types themselves, so it can actually reorder.
    type_labels = {"B": 0, "A": 1}.keys()
    type_rows = pd.DataFrame({"cell_type": ["A", "A", "B"]})
    assert ordering._resolve_type_rule(type_labels)(
        ["A", "B"], type_rows, "cell_type"
    ) == [
        "B",
        "A",
    ]  # preferred order from the key view, not label or size order

    # ...and a real set is still rejected at each of them.
    for call in (
        lambda: ordering.ColumnOrderRule(order={"R1", "L1"}),
        lambda: ordering.ColumnOrderRule(order=["R1"], label_columns={"a", "b"}),
        lambda: ordering._resolve_within_type_rule({"R1", "L1"}),
        lambda: ordering._resolve_type_rule({"R1", "L1"}),
    ):
        with pytest.raises(TypeError, match="ordered iterable"):
            call()


def test_non_scalar_rule_entries_are_rejected() -> None:
    # A list entry would stringify to "['x']" and silently never match.
    with pytest.raises(TypeError, match="scalar labels"):
        NeuronOrder(types=[["x"]])
    with pytest.raises(TypeError, match="scalar labels"):
        NeuronOrder(within={"A": [["x"]]})


def test_unhashable_callable_rules_are_accepted() -> None:
    from dataclasses import dataclass, field

    # eq=True without frozen sets __hash__ = None; a valid sorter regardless.
    @dataclass(eq=True)
    class MutableSorter:
        calls: list = field(default_factory=list)

        def __call__(self, type_names, typed_annotations, type_col):
            self.calls.append(list(type_names))
            return list(reversed(type_names))

    adjacency, annotations = _two_block_inputs()
    matrix = NestedMatrix.from_connectivity(
        adjacency, annotations, order=NeuronOrder(types=MutableSorter())
    )
    assert list(matrix.type_boundaries) == ["B", "A"]


def test_mutating_one_order_leaves_the_defaults_alone() -> None:
    from crantpy.utils.ordering import (
        DEFAULT_WITHIN_TYPE_ORDER,
        EB_COLUMN_ORDER,
        as_neuron_order,
    )

    # The module-level default table itself stays read-only...
    with pytest.raises(TypeError):
        DEFAULT_WITHIN_TYPE_ORDER["written"] = "through"  # type: ignore[index]

    # ...and every order snapshots its own copy of it, so editing one order
    # cannot poison the defaults a later build resolves.
    order = as_neuron_order(None)
    order.within["EPG/PEG"] = "id"  # type: ignore[index]
    assert as_neuron_order(None).within["EPG/PEG"] is EB_COLUMN_ORDER


def test_order_accepts_a_key_view_but_still_rejects_a_set() -> None:
    adjacency, annotations = _two_block_inputs()
    # Build `first` in a non-default order so the key view's insertion order is
    # observable -- a sorted() of the same keys would give ["A", "B"].
    first = NestedMatrix.from_connectivity(adjacency, annotations, order=["B", "A"])
    assert list(first.type_boundaries) == ["B", "A"]

    # Key views are Set instances but iterate in insertion order.
    relabelled = NestedMatrix.from_connectivity(
        adjacency, annotations, order=first.type_boundaries.keys()
    )
    assert list(relabelled.type_boundaries) == ["B", "A"]

    with pytest.raises(TypeError, match="ordered iterable"):
        NestedMatrix.from_connectivity(adjacency, annotations, order={"B", "A"})


def test_order_within_id_handles_non_decimal_digit_ids() -> None:
    from crantpy.utils.ordering import _order_by_id

    # "²".isdigit() is True but int("²") raises, so the numeric fast path must
    # gate on isdecimal() and fall back to a lexicographic sort here.
    assert _order_by_id("T", pd.DataFrame({"root_id": ["²", "1"]}), "root_id") == [
        "1",
        "²",
    ]
    assert _order_by_id("T", pd.DataFrame({"root_id": ["10", "9"]}), "root_id") == [
        "9",
        "10",
    ]


def test_constructor_rejects_a_null_cell_type_inside_a_block() -> None:
    # A neuron with no cell type cannot belong to a type block; accepting one
    # would break `typed_neurons + untyped_neurons == ordered_neurons`.
    with pytest.raises(ValueError, match="does not match neuron_to_type"):
        NestedMatrix(
            pd.DataFrame(np.ones((2, 2)), index=["1", "2"], columns=["1", "2"]),
            {"None": (0, 1), "B": (1, 2)},
            ["1", "2"],
            {"1": None, "2": "B"},
        )


def test_order_within_bare_sequence_applies_to_every_type() -> None:
    annotations = pd.DataFrame(
        {
            "root_id": ["40", "7", "5", "6"],
            "cell_type": ["EPG/PEG", "EPG/PEG", "PEN", "PEN"],
            "cell_subtype": ["EPG/PEG_L8", "EPG/PEG_R1", "PEN_R1", "PEN_L8"],
        }
    )
    adjacency = pd.DataFrame(
        np.zeros((4, 4)), index=annotations.root_id, columns=annotations.root_id
    )

    matrix = NestedMatrix.from_connectivity(
        adjacency,
        annotations,
        order={"types": ["EPG/PEG", "PEN"], "within": ["L8", "R1"]},
    )

    # L8 before R1 in *both* blocks: it overrides the EB default for EPG/PEG,
    # and it reverses PEN's annotation row order (which is R1 first).
    assert list(matrix.ordered_neurons) == ["40", "7", "6", "5"]


def test_annotations_with_a_duplicate_index_are_accepted() -> None:
    # set_index(drop=False) and pd.concat both yield a non-unique index; the
    # rows are positionally parallel, so the caller's index must not matter.
    annotations = pd.DataFrame(
        {"root_id": ["1", "1", "2", "2"], "cell_type": [None, "A", None, "B"]}
    )
    adjacency = pd.DataFrame(np.ones((2, 2)), index=["1", "2"], columns=["1", "2"])
    expected = NestedMatrix.from_connectivity(adjacency, annotations)

    # Interleaved, and BOTH neurons the same type -- otherwise the cell-type
    # block sort re-orders them and hides which row order was used. Neuron 1's
    # rows straddle neuron 2's and its typed row comes last, so the result is
    # decided by first appearance, not by which row carries the type.
    interleaved = pd.DataFrame(
        {"root_id": ["1", "2", "1"], "cell_type": [None, "A", "A"]}
    )
    assert NestedMatrix.from_connectivity(adjacency, interleaved).ordered_neurons == (
        "1",
        "2",
    )

    for reindexed in (
        annotations.set_index("root_id", drop=False),
        pd.concat([annotations, annotations]),
        annotations.set_axis([0, 0, 1, 1]),
    ):
        matrix = NestedMatrix.from_connectivity(adjacency, reindexed)
        assert matrix.ordered_neurons == expected.ordered_neurons
        assert (
            dict(matrix.neuron_to_type)
            == dict(expected.neuron_to_type)
            == {
                "1": "A",
                "2": "B",
            }
        )


def test_within_keys_naming_the_same_cell_type_are_rejected() -> None:
    from crantpy.utils.ordering import build_ordered_neurons

    # 1 and "1" normalize to the same cell type, so one would silently shadow
    # the other depending on insertion order.
    annotations = pd.DataFrame(
        {"root_id": ["30", "4", "200"], "cell_type": ["1", "1", "1"]}
    )
    for within in ({1: "id", "1": "annotation"}, {"1": "annotation", 1: "id"}):
        with pytest.raises(ValueError, match="naming the same cell type"):
            build_ordered_neurons(
                annotations, "cell_type", ["1"], "root_id", within=within
            )

    # A single unambiguous key is still fine.
    assert build_ordered_neurons(
        annotations, "cell_type", ["1"], "root_id", within={1: "id"}
    )[0] == ["4", "30", "200"]


def test_colliding_within_keys_survive_the_snapshot() -> None:
    # 1 and "1" are distinct dict keys; _snapshot must keep them distinct so
    # the build-time duplicate-key guard above still sees both when the
    # mapping arrives through order= rather than a direct call.
    assert NeuronOrder(within={1: "id", "1": "annotation"}) == NeuronOrder(
        within={"1": "annotation", 1: "id"}
    )

    adjacency = pd.DataFrame([[0]], index=["30"], columns=["30"])
    annotations = pd.DataFrame({"root_id": ["30"], "cell_type": ["1"]})
    with pytest.raises(ValueError, match="naming the same cell type"):
        NestedMatrix.from_connectivity(
            adjacency, annotations, order={"within": {1: "id", "1": "annotation"}}
        )


def test_order_with_a_nested_mapping_still_round_trips() -> None:
    import copy
    import pickle

    # _snapshot copies every level to plain containers, so the object pickles
    # without custom hooks.
    order = NeuronOrder(within={"A": {"B": 1}})

    restored = pickle.loads(pickle.dumps(order))
    assert restored == order
    assert copy.deepcopy(order) == order


# ---------------------------------------------------------------------------
# Behavior documented in the ordering docstrings
# ---------------------------------------------------------------------------


def test_column_rule_skips_a_matching_but_unranked_column() -> None:
    from crantpy.utils.ordering import ColumnOrderRule, _extract_ranked_label

    # A column matching the pattern but absent from `order` is skipped and
    # the next column tried.
    rule = ColumnOrderRule(order=["R1", "L1"])
    rank = {"R1": 0, "L1": 1}
    row = pd.Series({"cell_instance": "X_R9", "cell_subtype": "X_L1", "root_id": "1"})

    assert _extract_ranked_label(row, rule, "root_id", rank) == "L1"

    # ...and with no ranked label anywhere, the neuron stays unranked.
    unranked = pd.Series(
        {"cell_instance": "X_R9", "cell_subtype": "X_R8", "root_id": "1"}
    )
    assert _extract_ranked_label(unranked, rule, "root_id", rank) is None


def test_preferred_entries_match_exactly() -> None:
    from crantpy.utils.ordering import _sort_cell_types

    # A preferred entry moves only the type it names exactly...
    assert _sort_cell_types(["ER1", "ER10", "ER2", "AB3"], preferred=["ER10"]) == [
        "ER10",
        "AB3",
        "ER1",
        "ER2",
    ]

    # ...even when it is also a prefix of other types.
    assert _sort_cell_types(["ER1", "ER2", "AB3"], preferred=["ER"]) == [
        "AB3",
        "ER1",
        "ER2",
    ]


def test_sort_cell_types_puts_un_numbered_labels_last_in_their_prefix() -> None:
    from crantpy.utils.ordering import _sort_cell_types

    # The documented generic order: ER1, ER2, ER10, ER.
    assert _sort_cell_types(["ER", "ER1", "ER2", "ER10"]) == [
        "ER1",
        "ER2",
        "ER10",
        "ER",
    ]


def test_documented_ordering_pipeline_works_on_un_normalized_types() -> None:
    from crantpy.utils import ordering

    # resolve_type_order -> build_ordered_neurons is the documented pipeline and
    # both are public; step 2 normalizes type names, so step 3 must match on the
    # normalized form or every block silently comes back empty.
    annotations = pd.DataFrame({"root_id": ["1", "2"], "cell_type": [7, 7]})
    sorted_types = ordering.resolve_type_order(annotations, "cell_type")

    assert sorted_types == ["7"]
    assert ordering.build_ordered_neurons(
        annotations, "cell_type", sorted_types, "root_id"
    ) == (["1", "2"], {"7": (0, 2)})

    # Both halves must normalize identically, so chain them rather than passing
    # sorted_types by hand: str(1.0) is "1.0" but _stringify_id_value(1.0) is "1".
    # A numeric cell_type column upcasts to float as soon as it holds a NaN, so
    # this dtype is ordinary, not exotic.
    for cell_types in ([1, 2], [1.0, 2.0], ["1", "2"], [np.int64(1), np.int64(2)]):
        typed = pd.DataFrame({"root_id": ["10", "20"], "cell_type": cell_types})
        chained = ordering.build_ordered_neurons(
            typed,
            "cell_type",
            ordering.resolve_type_order(typed, "cell_type"),
            "root_id",
        )
        assert chained == (["10", "20"], {"1": (0, 1), "2": (1, 2)}), cell_types

    # Called directly with raw (un-normalized) sorted_types, as a caller
    # composing the public helpers by hand may well do.
    raw = pd.DataFrame({"root_id": ["10", "20"], "cell_type": [7.0, 7.0]})
    assert ordering.build_ordered_neurons(raw, "cell_type", [7.0], "root_id") == (
        ["10", "20"],
        {"7": (0, 2)},
    )

    # Non-integral floats keep their decimal part on both sides.
    fractional = pd.DataFrame({"root_id": ["10"], "cell_type": [1.5]})
    assert ordering.build_ordered_neurons(
        fractional,
        "cell_type",
        ordering.resolve_type_order(fractional, "cell_type"),
        "root_id",
    ) == (["10"], {"1.5": (0, 1)})


def test_order_types_size_agrees_across_cell_type_dtypes() -> None:
    from crantpy.utils import ordering

    # Type "1" has one neuron, type "2" has three, so size order is 2 then 1.
    # The count table must be keyed the same way resolve_type_order names the
    # types, or every lookup misses, all sizes tie at zero, and "size" silently
    # degrades to label order.
    for cell_types in ([1, 2, 2, 2], [1.0, 2.0, 2.0, 2.0], ["1", "2", "2", "2"]):
        raw = pd.DataFrame({"root_id": ["1", "2", "3", "4"], "cell_type": cell_types})
        assert ordering.resolve_type_order(raw, "cell_type", "size") == [
            "2",
            "1",
        ], cell_types

    # And end to end, where the column is already normalized.
    annotations = pd.DataFrame(
        {"root_id": ["1", "2", "3", "4"], "cell_type": [1.0, 2.0, 2.0, 2.0]}
    )
    adjacency = pd.DataFrame(
        np.zeros((4, 4)), index=annotations.root_id, columns=annotations.root_id
    )
    matrix = NestedMatrix.from_connectivity(
        adjacency, annotations, order={"types": "size"}
    )
    assert list(matrix.type_boundaries) == ["2", "1"]


def test_order_with_a_mapping_types_rule_still_round_trips() -> None:
    import copy
    import pickle

    # A mapping is not a legal `types` rule, but the object is constructible
    # and must still pickle and deep-copy.
    order = NeuronOrder(types={"a": 1})

    restored = pickle.loads(pickle.dumps(order))
    assert restored == order
    assert copy.deepcopy(order) == order
    assert NeuronOrder(types=["B", "A"]).types == ("B", "A")


def test_feeding_a_block_order_back_reproduces_it_exactly() -> None:
    # A bare type name alongside its numbered siblings ("ER" next to ER1/ER2)
    # matches exactly, so another matrix's block order round-trips as a
    # sequence.
    annotations = pd.DataFrame(
        {
            "root_id": ["11", "12", "13", "21", "31", "41"],
            "cell_type": ["ER", "ER", "ER", "ER1", "ER2", "AB1"],
        }
    )
    adjacency = pd.DataFrame(
        np.zeros((6, 6)), index=annotations.root_id, columns=annotations.root_id
    )
    first = NestedMatrix.from_connectivity(
        adjacency, annotations, order={"types": "size"}
    )
    assert list(first.type_boundaries) == ["ER", "AB1", "ER1", "ER2"]

    as_sequence = NestedMatrix.from_connectivity(
        adjacency, annotations, order=first.type_boundaries.keys()
    )
    assert list(as_sequence.type_boundaries) == list(first.type_boundaries)


def test_untyped_tail_order_is_pinned() -> None:
    # IDs whose string order differs from their numeric order, AND annotation
    # rows in the opposite order to the expected result -- so neither a numeric
    # sort nor plain annotation row order can produce the expectation.
    ids = ["1", "2", "30", "7", "20", "9"]
    adjacency = pd.DataFrame(np.zeros((6, 6)), index=ids, columns=ids)
    # 7 and 30 have annotation rows with no cell type; 20 and 9 have no row.
    annotations = pd.DataFrame(
        {"root_id": ["1", "2", "7", "30"], "cell_type": ["A", "A", None, None]}
    )

    matrix = NestedMatrix.from_connectivity(
        adjacency, annotations, annotation_scope="all"
    )

    # Untyped-with-a-row first, then missing-entirely; each sorted as strings.
    assert matrix.untyped_neurons == ("30", "7", "20", "9")
    assert matrix.typed_neurons == ("1", "2")
    assert matrix.ordered_neurons == ("1", "2", "30", "7", "20", "9")


def test_directed_untyped_tail_order_is_pinned_per_axis() -> None:
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": ["1", "30", "7"],
            "post_pt_root_id": ["2", "20", "9"],
        }
    )
    annotations = pd.DataFrame(
        {"root_id": ["1", "2", "7", "30"], "cell_type": ["A", "B", None, None]}
    )

    matrix = DirectedNestedMatrix.from_synapses(
        synapses, annotations, weight_mode="count", annotation_scope="all"
    )

    # Annotation order is 7 before 30; the documented string sort reverses it.
    assert matrix.source_untyped_neurons == ("30", "7")
    assert matrix.target_untyped_neurons == ("20", "9")


def test_min_neurons_for_plot_applies_only_at_neuron_level() -> None:
    # Documented as "level='neuron' only"; the type-level matrices are plotted
    # whole, so a user filtering a type-level plot gets an unfiltered figure.
    annotations = pd.DataFrame(
        {
            "root_id": ["1", "2", "3", "4", "5", "6"],
            "cell_type": ["KC", "KC", "KC", "MB", "XX", "XX"],
        }
    )
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": ["1", "2", "3", "4", "5", "6"],
            "post_pt_root_id": ["2", "3", "4", "5", "6", "1"],
        }
    )
    matrix = NestedMatrix.from_synapses(synapses, annotations, weight_mode="count")

    fig, ax = matrix.plot(level="neuron", min_neurons_for_plot=2)
    assert [t.get_text() for t in ax.get_yticklabels()] == ["KC", "XX"]
    plt.close(fig)

    for level in ("type_mean", "type_sum"):
        fig, ax = matrix.plot(level=level, min_neurons_for_plot=2)
        assert [t.get_text() for t in ax.get_yticklabels()] == ["KC", "MB", "XX"]
        plt.close(fig)


def test_column_rule_on_an_empty_block_is_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from crantpy.utils.ordering import EB_COLUMN_ORDER, _order_by_column_rule

    # pandas calls DataFrame.apply(axis=1) once on an all-NaN dummy row when the
    # frame is empty, which used to warn about a nonexistent "neuron nan".
    caplog.set_level("WARNING", logger="crantpy.utils.ordering")
    empty = pd.DataFrame({"root_id": [], "cell_subtype": []})

    assert _order_by_column_rule(EB_COLUMN_ORDER)("EPG/PEG", empty, "root_id") == []
    assert caplog.records == []


def test_find_duplicates_returns_first_repeat_order() -> None:
    from crantpy.utils.ordering import _find_duplicates

    # Documented as "first-repeat order": the value is recorded when its SECOND
    # occurrence is seen, so "b" (repeating first) precedes "a".
    assert _find_duplicates(["b", "a", "b", "a"]) == ["b", "a"]
    assert _find_duplicates(["a", "b", "b", "a"]) == ["b", "a"]
    # ...and an input whose first-repeat order is ASCENDING, so the expectation
    # cannot also be satisfied by a reverse sort.
    assert _find_duplicates(["a", "b", "a", "b"]) == ["a", "b"]
    assert _find_duplicates(["a", "b", "c"]) == []
    # Reported once however many times it repeats.
    assert _find_duplicates(["a", "a", "a"]) == ["a"]


def test_stringify_id_value_keeps_bools_distinct_from_ints() -> None:
    from crantpy.utils.ordering import _stringify_id_value

    # bool is a subclass of int, so without an explicit guard True/False would
    # normalize to "1"/"0" and collide with the integer IDs 1 and 0.
    assert _stringify_id_value(True) == "True"
    assert _stringify_id_value(False) == "False"
    assert _stringify_id_value(np.bool_(True)) == "True"
    assert _stringify_id_value(1) == "1"
    assert _stringify_id_value(0) == "0"


# ---------------------------------------------------------------------------
# Validation in the public ordering helpers
# ---------------------------------------------------------------------------


def test_sorters_returning_each_neuron_exactly_once_is_enforced_by_identity() -> None:
    from crantpy.utils.ordering import _apply_within_type_sorter

    # A same-length sorter that duplicates one neuron and drops another must
    # still fail.
    rows = pd.DataFrame({"root_id": ["1", "2", "3"]})

    with pytest.raises(ValueError, match="exactly once"):
        _apply_within_type_sorter(
            lambda name, r, col: ["1", "1", "2"], "A", rows, "root_id"
        )
    # Same length, same multiset -> accepted.
    assert _apply_within_type_sorter(
        lambda name, r, col: ["3", "1", "2"], "A", rows, "root_id"
    ) == ["3", "1", "2"]


def test_type_sorters_returning_each_block_exactly_once_is_enforced() -> None:
    from crantpy.utils.ordering import _apply_type_sorter

    rows = pd.DataFrame({"cell_type": ["A", "B"]})

    with pytest.raises(ValueError, match="exactly once"):
        _apply_type_sorter(lambda n, r, c: ["A", "A"], ["A", "B"], rows, "cell_type")
    assert _apply_type_sorter(
        lambda n, r, c: ["B", "A"], ["A", "B"], rows, "cell_type"
    ) == ["B", "A"]


def test_type_sorter_mutating_its_input_cannot_bypass_the_check() -> None:
    from crantpy.utils.ordering import _apply_type_sorter, resolve_type_order

    # A sorter editing the list it receives in place used to edit the expected
    # set too, so both sides of the comparison matched and blocks were dropped
    # silently.
    def dropping_sorter(type_names, typed_annotations, type_col):
        type_names.remove("B")
        return type_names

    rows = pd.DataFrame({"root_id": ["1", "2"], "cell_type": ["A", "B"]})
    with pytest.raises(ValueError, match="exactly once"):
        _apply_type_sorter(dropping_sorter, ["A", "B"], rows, "cell_type")
    with pytest.raises(ValueError, match="exactly once"):
        resolve_type_order(rows, "cell_type", dropping_sorter)


def test_stringify_id_axis_rejects_nulls_and_normalize_drops_them() -> None:
    from crantpy.utils.ordering import _normalize_id_values, _stringify_id_axis

    # A null slipping through would become the string "nan" and enter the axis
    # as a bogus neuron.
    with pytest.raises(ValueError, match="matrix index contains null neuron IDs"):
        _stringify_id_axis(["1", np.nan], "matrix index")
    assert list(_stringify_id_axis([1, 2.0, "3"], "matrix index")) == ["1", "2", "3"]

    # The sibling helper documents the opposite policy: drop them silently.
    assert _normalize_id_values(["1", np.nan, "2"]) == ["1", "2"]
    assert _normalize_id_values([1, None, 2.0]) == ["1", "2"]


def test_pipeline_normalizes_the_neuron_id_column_too() -> None:
    from crantpy.utils import ordering

    # A float ID column is numeric, but str(10.0) is "10.0" -- not isdecimal --
    # so without normalization "id" order fell back to a lexicographic sort AND
    # returned IDs that match nothing on a stringified matrix axis.
    for ids in ([10.0, 9.0, 100.0], [10, 9, 100], ["10", "9", "100"]):
        annotations = pd.DataFrame({"root_id": ids, "cell_type": ["A", "A", "A"]})
        assert ordering.build_ordered_neurons(
            annotations, "cell_type", ["A"], "root_id", within="id"
        ) == (["9", "10", "100"], {"A": (0, 3)}), ids

    # _order_by_id on its own honours the same normalization.
    assert ordering._order_by_id(
        "A", pd.DataFrame({"root_id": [10.0, 9.0, 100.0]}), "root_id"
    ) == ["9", "10", "100"]

    # ...and the returned IDs match a real matrix axis.
    adjacency = pd.DataFrame(
        np.zeros((3, 3)), index=["9", "10", "100"], columns=["9", "10", "100"]
    )
    float_ids = pd.DataFrame(
        {"root_id": [10.0, 9.0, 100.0], "cell_type": ["A", "A", "A"]}
    )
    assert NestedMatrix.from_connectivity(
        adjacency, float_ids, order={"within": "id"}
    ).ordered_neurons == ("9", "10", "100")


def test_build_ordered_neurons_skips_a_type_with_no_rows() -> None:
    from crantpy.utils.ordering import build_ordered_neurons

    # A caller composing the public helpers can name a type that has no rows;
    # it must be skipped, not emitted as a zero-width block.
    annotations = pd.DataFrame(
        {"root_id": ["1", "2", "3"], "cell_type": ["A", "A", "A"]}
    )

    assert build_ordered_neurons(annotations, "cell_type", ["A", "Z"], "root_id") == (
        ["1", "2", "3"],
        {"A": (0, 3)},
    )


def test_is_missing_scalar_is_false_for_non_scalars() -> None:
    from crantpy.utils.ordering import _is_missing_scalar

    # pd.isna returns an ARRAY for a list/array argument; treating that as
    # missing would raise deep inside the pipeline instead of passing it through.
    assert _is_missing_scalar(["x"]) is False
    assert _is_missing_scalar(np.array([1, 2])) is False
    assert _is_missing_scalar(np.nan) is True
    assert _is_missing_scalar(None) is True

    # Reachable end to end: a list-valued cell type is treated as a real label.
    annotations = pd.DataFrame({"root_id": ["1", "2"], "cell_type": [["x"], "B"]})
    adjacency = pd.DataFrame(np.zeros((2, 2)), index=["1", "2"], columns=["1", "2"])
    assert set(
        NestedMatrix.from_connectivity(adjacency, annotations).type_boundaries
    ) == {"B", "['x']"}


def test_unresolvable_label_warning_names_the_neuron_or_says_unknown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from crantpy.utils.ordering import ColumnOrderRule, _extract_ranked_label

    rule = ColumnOrderRule(order=["R1"])
    caplog.set_level("WARNING", logger="crantpy.utils.ordering")

    _extract_ranked_label(
        pd.Series({"root_id": "77", "cell_subtype": "x"}), rule, "root_id", {"R1": 0}
    )
    assert "for neuron 77" in caplog.text

    caplog.clear()
    # A row without the ID column at all falls back to the placeholder.
    _extract_ranked_label(pd.Series({"cell_subtype": "x"}), rule, "root_id", {"R1": 0})
    assert "for neuron <unknown>" in caplog.text


# ---------------------------------------------------------------------------
# _sort_cell_types: every component of the documented sort key
# ---------------------------------------------------------------------------


def test_sort_cell_types_breaks_ties_on_the_label_itself() -> None:
    from crantpy.utils.ordering import _sort_cell_types

    # These all parse to the same (ALPHA_PREFIX, numeric_suffix) = ("EPG", inf),
    # so without the third key component the order falls back to set iteration
    # and varies with PYTHONHASHSEED. EPG/PEG is a real CRANT cell type.
    assert _sort_cell_types(["EPG/PEG", "EPG/PEN", "EPG/PEB"]) == [
        "EPG/PEB",
        "EPG/PEG",
        "EPG/PEN",
    ]
    # Stable across repeated calls within a run.
    assert _sort_cell_types(["EPG/PEN", "EPG/PEB", "EPG/PEG"]) == [
        "EPG/PEB",
        "EPG/PEG",
        "EPG/PEN",
    ]


def test_sort_cell_types_is_case_insensitive_on_the_prefix() -> None:
    from crantpy.utils.ordering import _sort_cell_types

    # The documented key uses ALPHA_PREFIX (upper-cased), so "er1" groups with
    # "ER2" and sorts by its number, rather than all lower-case names sorting
    # after all upper-case ones.
    assert _sort_cell_types(["er1", "ER2"]) == ["er1", "ER2"]
    assert _sort_cell_types(["ER2", "er1"]) == ["er1", "ER2"]

    # The no-regex-match fallback upper-cases as well.
    assert _sort_cell_types(["_a", "_B"]) == ["_a", "_B"]


def test_sort_cell_types_drops_null_labels() -> None:
    from crantpy.utils.ordering import _sort_cell_types

    # A null would otherwise become the bogus cell types "nan"/"None".
    assert _sort_cell_types([np.nan, "A", None]) == ["A"]
    assert _sort_cell_types([np.nan, None]) == []


def test_resolve_type_order_ignores_null_types() -> None:
    from crantpy.utils.ordering import resolve_type_order

    annotations = pd.DataFrame({"root_id": ["1", "2"], "cell_type": [np.nan, "A"]})
    assert resolve_type_order(annotations, "cell_type") == ["A"]


def test_resolve_type_order_dedupes_on_normalized_type_names() -> None:
    from crantpy.utils.ordering import resolve_type_order

    # Raw 1, "1" and 1.0 are distinct to unique() but the same cell type once
    # normalized; deduping the raw values used to yield ["1", "1", ...] and a
    # spurious "exactly once" error from the sorter.
    annotations = pd.DataFrame(
        {"root_id": ["1", "2", "3", "4"], "cell_type": [1, "1", 1.0, "2"]}
    )
    assert resolve_type_order(annotations, "cell_type") == ["1", "2"]


def test_order_types_by_size_breaks_ties_by_label_order() -> None:
    from crantpy.utils.ordering import resolve_type_order

    # ER2 and ER10 both have one neuron: the tie must break on label order
    # (ER2 before ER10), not alphabetically ("ER10" < "ER2") nor on input order.
    annotations = pd.DataFrame(
        {
            "root_id": ["1", "2", "3", "4", "5"],
            "cell_type": ["ER1", "ER1", "ER1", "ER2", "ER10"],
        }
    )
    assert resolve_type_order(annotations, "cell_type", "size") == [
        "ER1",
        "ER2",
        "ER10",
    ]

    # Same-size blocks alone, to pin the label ordering of the tie-break input.
    tied = pd.DataFrame(
        {
            "root_id": ["1", "2", "3", "4"],
            "cell_type": ["ER10", "ER1", "ER2", "ER2"],
        }
    )
    assert resolve_type_order(tied, "cell_type", "size") == ["ER2", "ER1", "ER10"]


# ---------------------------------------------------------------------------
# _validate_invariants: the guards the constructor tests did not reach
# ---------------------------------------------------------------------------


def _square(ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(np.ones((len(ids), len(ids))), index=ids, columns=ids)


@pytest.mark.parametrize(
    "boundaries, ordered, neuron_to_type, message",
    [
        # zero-width block
        ({"A": (0, 0), "B": (0, 2)}, ["1", "2"], {"1": "B", "2": "B"}, "invalid slice"),
        # block running past the end of the axis
        ({"A": (0, 5)}, ["1", "2"], {"1": "A", "2": "A"}, "invalid slice"),
        # neuron_to_type naming a neuron that is not on the axis
        (
            {"A": (0, 2)},
            ["1", "2"],
            {"1": "A", "2": "A", "99": "A"},
            "not present in ordered_neurons",
        ),
        # neurons sitting in the wrong block
        (
            {"A": (0, 1), "B": (1, 2)},
            ["1", "2"],
            {"1": "B", "2": "A"},
            "does not match neuron_to_type",
        ),
        # matrix axes disagreeing with ordered_neurons
        ({"A": (0, 2)}, ["2", "1"], {"1": "A", "2": "A"}, "must match ordered_neurons"),
    ],
    ids=["zero-width", "past-end", "extra-neuron", "wrong-block", "axis-mismatch"],
)
def test_validate_invariants_rejects_inconsistent_metadata(
    boundaries, ordered, neuron_to_type, message
) -> None:
    with pytest.raises(ValueError, match=message):
        NestedMatrix(_square(["1", "2"]), boundaries, ordered, neuron_to_type)


@pytest.mark.parametrize(
    "boundaries, source, neuron_to_type, message",
    [
        ({"A": (0, 0), "B": (0, 2)}, ["1", "2"], {"1": "B", "2": "B"}, "invalid slice"),
        ({"A": (0, 5)}, ["1", "2"], {"1": "A", "2": "A"}, "invalid slice"),
        ({"A": (0, 2)}, ["1", "2"], {"1": "A", "2": "A", "99": "A"}, "not present in"),
        (
            {"A": (0, 1), "B": (1, 2)},
            ["1", "2"],
            {"1": "B", "2": "A"},
            "does not match",
        ),
    ],
    ids=["zero-width", "past-end", "extra-neuron", "wrong-block"],
)
def test_directed_validate_axis_metadata_rejects_the_same(
    boundaries, source, neuron_to_type, message
) -> None:
    with pytest.raises(ValueError, match=message):
        DirectedNestedMatrix(
            pd.DataFrame(np.ones((2, 1)), index=["1", "2"], columns=["9"]),
            source_neurons=source,
            target_neurons=["9"],
            source_type_boundaries=boundaries,
            source_neuron_to_type=neuron_to_type,
        )


def test_duplicate_adjacency_axis_labels_are_summed() -> None:
    # 1.0 and "1" both normalize to "1", so the 2x2 frame collapses to a single
    # cell holding all four weights. Without the groupby, reindexing raises.
    adjacency = pd.DataFrame([[1, 2], [3, 4]], index=[1.0, "1"], columns=[1.0, "1"])
    annotations = pd.DataFrame({"root_id": ["1"], "cell_type": ["A"]})

    matrix = NestedMatrix.from_connectivity(adjacency, annotations)

    assert list(matrix.matrix.index) == ["1"]
    assert matrix.matrix.values.tolist() == [[10.0]]


def test_relative_weights_leave_a_zero_sum_row_unchanged() -> None:
    # The exception is a zero row SUM, not the absence of output: +5 and -5
    # cancel, so the row is left raw rather than divided by zero.
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": ["1", "1"],
            "post_pt_root_id": ["2", "3"],
            "Weight": [5.0, -5.0],
        }
    )
    annotations = pd.DataFrame(
        {"root_id": ["1", "2", "3"], "cell_type": ["A", "B", "B"]}
    )
    matrix = NestedMatrix.from_synapses(
        synapses, annotations, weight_mode="column", weight_column="Weight"
    )

    relative = matrix.get_relative_weights()
    assert relative.loc["1"].tolist() == [0.0, 5.0, -5.0]
    assert relative.loc["1"].sum() == 0.0
    # A row with no output at all is likewise left at zero, not NaN.
    assert relative.loc["2"].tolist() == [0.0, 0.0, 0.0]

    # A normal row still normalizes to 1.0.
    positive = matrix.matrix.copy()
    positive.loc["1"] = [0.0, 5.0, 15.0]
    normal = NestedMatrix(
        positive,
        dict(matrix.type_boundaries),
        list(matrix.ordered_neurons),
        dict(matrix.neuron_to_type),
    )
    assert normal.get_relative_weights().loc["1"].sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Guards whose triggering condition no fixture previously constructed
# ---------------------------------------------------------------------------


def test_all_selector_accumulates_exclusions_and_sorts_its_repr() -> None:
    from crantpy.queries.nested_connectivity_matrices import All

    # minus() must accumulate, not replace.
    assert All.minus("a").minus("b").resolve(["a", "b", "c"]) == ["c"]
    assert All.minus("a", "b").resolve(["a", "b", "c"]) == ["c"]
    assert All.resolve(["a", "b"]) == ["a", "b"]
    # The original selector is not mutated by chaining.
    assert All.resolve(["a"]) == ["a"]

    # repr sorts, so it is stable across runs despite frozenset ordering.
    assert repr(All.minus("z", "a", "m")) == "All.minus('a', 'm', 'z')"
    assert repr(All) == "All"


def test_edge_list_weight_column_precedence_and_default() -> None:
    coerce = NestedMatrix._coerce_to_adjacency

    # "weight" wins over the format's alternate weight column.
    both = pd.DataFrame(
        {"source": ["1"], "target": ["2"], "weight": [5], "n_syn": [100]}
    )
    assert coerce(both).loc["1", "2"] == 5

    # n_syn is used when "weight" is absent.
    alt = pd.DataFrame({"source": ["1"], "target": ["2"], "n_syn": [100]})
    assert coerce(alt).loc["1", "2"] == 100

    # With neither, each edge counts once, and duplicates aggregate.
    neither = pd.DataFrame({"pre": ["1", "1"], "post": ["2", "2"]})
    assert coerce(neither).loc["1", "2"] == 2


def test_duplicate_axis_labels_collapse_in_first_appearance_order() -> None:
    # groupby(sort=False): the surviving label order follows first appearance,
    # not sorted order -- "2" stays before "1".
    rows = pd.DataFrame(np.ones((3, 2)), index=["2", "1", "1"], columns=["9", "8"])
    normalized = NestedMatrix._normalize_adjacency_axes(rows)
    assert list(normalized.index) == ["2", "1"]

    cols = pd.DataFrame(np.ones((2, 4)), index=["7", "6"], columns=["9", "1", "9", "1"])
    normalized = NestedMatrix._normalize_adjacency_axes(cols)
    assert list(normalized.columns) == ["9", "1"]


def test_non_numeric_weight_column_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be numeric"):
        NestedMatrix._coerce_weight_values(pd.DataFrame({"w": ["abc"]}), "w")

    # A numeric-looking string is accepted and converted.
    coerced = NestedMatrix._coerce_weight_values(pd.DataFrame({"w": ["3"]}), "w")
    assert coerced["w"].tolist() == [3]


def test_order_types_by_size_puts_an_absent_type_last() -> None:
    from crantpy.utils.ordering import _order_types_by_size

    # A name with no rows counts as zero and sorts after every present type.
    present = pd.DataFrame({"t": ["A", "A"]})
    assert _order_types_by_size(["A", "B"], present, "t") == ["A", "B"]
    assert _order_types_by_size(["B", "A"], present, "t") == ["A", "B"]


def _segment_counts(ax: plt.Axes) -> list[int]:
    return [
        len(c.get_segments()) for c in ax.collections if isinstance(c, LineCollection)
    ]


def test_boundary_grid_lines_skip_the_trailing_block() -> None:
    # The last block must not get a grid line at the matrix edge; counting
    # LineCollections alone cannot see an extra segment, so count segments.
    annotations = pd.DataFrame(
        {"root_id": ["1", "2", "3", "4"], "cell_type": ["A", "A", "B", "B"]}
    )
    adjacency = pd.DataFrame(
        np.ones((4, 4)), index=annotations.root_id, columns=annotations.root_id
    )
    matrix = NestedMatrix.from_connectivity(adjacency, annotations)

    fig, ax = matrix.plot(level="neuron")
    assert _segment_counts(ax) == [4, 8, 8, 2, 2]
    plt.close(fig)


def test_rectangular_boundary_grid_lines_skip_the_trailing_block() -> None:
    annotations = pd.DataFrame(
        {"root_id": ["1", "2", "3", "4"], "cell_type": ["A", "A", "B", "B"]}
    )
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": ["1", "2", "3", "4"],
            "post_pt_root_id": ["3", "4", "1", "2"],
        }
    )
    matrix = DirectedNestedMatrix.from_synapses(
        synapses, annotations, weight_mode="count"
    )

    fig, ax = matrix.plot(level="neuron")
    # One outer frame plus one interior line per axis -- not one per block.
    assert _segment_counts(ax) == [4, 2, 2]
    plt.close(fig)


def test_scalar_null_selector_selects_nothing() -> None:
    # A scalar null must resolve to "select nothing", not to the bogus id "nan".
    assert NestedMatrix._selector_to_str_set(np.nan) == set()
    assert NestedMatrix._selector_to_str_set(None) is None  # no filter at all
    assert NestedMatrix._selector_to_str_set(7) == {"7"}
    # Nulls inside an iterable are dropped rather than stringified.
    assert NestedMatrix._selector_to_str_set([1, np.nan, 2.0]) == {"1", "2"}


def test_resolve_relevant_annotations_drops_null_id_rows() -> None:
    from crantpy.utils.ordering import resolve_relevant_annotations

    # A null root_id must not become the neuron "nan".
    annotations = pd.DataFrame(
        {"root_id": [np.nan, "1", "2"], "cell_type": ["X", "A", "B"]}
    )
    resolved = resolve_relevant_annotations(
        {"1", "2", "nan"}, annotations, "root_id", "cell_type"
    )

    assert sorted(resolved.typed["root_id"]) == ["1", "2"]
    assert dict(resolved.id_map) == {"1": "A", "2": "B"}
    assert resolved.missing_ids == ["nan"]


def test_untyped_warning_lists_up_to_ten_neurons(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from crantpy.utils.ordering import build_axis_ordering

    # Six untyped neurons: under the documented threshold, so all are listed.
    annotations = pd.DataFrame(
        {"root_id": [str(i) for i in range(1, 7)], "cell_type": [None] * 6}
    )
    caplog.set_level("WARNING", logger="crantpy.utils.ordering")
    build_axis_ordering(
        {str(i) for i in range(1, 7)}, annotations, "root_id", "cell_type"
    )
    for neuron in ("1", "2", "3", "4", "5", "6"):
        assert f"'{neuron}'" in caplog.text

    assert len(caplog.records[-1].args[-1]) == 6

    # Twelve: over the threshold, so the payload is truncated to ten while the
    # count still reports all twelve. (IDs sort as strings, so the first ten of
    # "1".."12" are 1, 10, 11, 12, 2, 3, 4, 5, 6, 7.)
    caplog.clear()
    many = pd.DataFrame(
        {"root_id": [str(i) for i in range(1, 13)], "cell_type": [None] * 12}
    )
    build_axis_ordering({str(i) for i in range(1, 13)}, many, "root_id", "cell_type")
    logged = caplog.records[-1].args[-1]
    assert caplog.records[-1].args[0] == 12
    assert logged == ["1", "10", "11", "12", "2", "3", "4", "5", "6", "7"]


@patch("crantpy.viz.mesh.load_neuropil_mesh")
def test_neuropils_with_no_synapses_are_skipped(mock_load: MagicMock) -> None:
    # The first mesh contains both synapses, the second none: the empty ROI must
    # be dropped from the collection, not emitted as an empty matrix.
    mock_load.side_effect = [_mock_mesh([True, True]), _mock_mesh([False, False])]
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 2],
            "post_pt_root_id": [3, 4],
            "ctr_pt_position": [[1, 2, 3], [4, 5, 6]],
        }
    )

    result = NestedMatrix.from_synapses_by_neuropil(
        synapses,
        _make_directed_annotations(),
        neuropil_names=["antennal_lobe_left", "antennal_lobe_right"],
        coordinates="nm",
        weight_mode="count",
        include_other=False,
    )

    assert list(result) == ["antennal_lobe_left"]


@patch("crantpy.viz.mesh.load_neuropil_mesh")
def test_directed_include_other_collects_unassigned_synapses(
    mock_load: MagicMock,
) -> None:
    # One synapse falls outside every mesh, so it belongs to the "other" ROI.
    mock_load.return_value = _mock_mesh([True, False])
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 2],
            "post_pt_root_id": [3, 4],
            "ctr_pt_position": [[1, 2, 3], [4, 5, 6]],
        }
    )
    kwargs = dict(
        neuropil_names=["antennal_lobe_left"],
        coordinates="nm",
        weight_mode="count",
    )

    with_other = DirectedNestedMatrix.from_synapses_by_neuropil(
        synapses, _make_directed_annotations(), include_other=True, **kwargs
    )
    assert set(with_other) == {"antennal_lobe_left", "other"}
    assert list(with_other["other"].matrix.index) == ["2"]

    mock_load.return_value = _mock_mesh([True, False])
    without = DirectedNestedMatrix.from_synapses_by_neuropil(
        synapses, _make_directed_annotations(), include_other=False, **kwargs
    )
    assert set(without) == {"antennal_lobe_left"}


# ---------------------------------------------------------------------------
# Bytes rules, pattern validation, and normalized sorter inputs
# ---------------------------------------------------------------------------


def test_bytes_order_rules_are_rejected_not_silently_ignored() -> None:
    from crantpy.utils.ordering import (
        _sort_cell_types,
        as_neuron_order,
        build_ordered_neurons,
    )

    # bytes iterate as integers, so b"ER" would silently become (69, 82) and
    # the requested order would be dropped without an error.
    with pytest.raises(TypeError, match="yields integers"):
        NeuronOrder(types=b"ER")
    with pytest.raises(TypeError, match="yields integers"):
        NeuronOrder(types=bytearray(b"ER"))
    with pytest.raises(TypeError, match="yields integers"):
        NeuronOrder(within={"A": b"R1"})
    with pytest.raises(TypeError, match="yields integers"):
        as_neuron_order(b"PEN")
    with pytest.raises(TypeError, match="yields integers"):
        ColumnOrderRule(order=b"R1")
    with pytest.raises(TypeError, match="yields integers"):
        ColumnOrderRule(order=["R1"], label_columns=bytearray(b"c"))
    with pytest.raises(TypeError, match="yields integers"):
        NeuronOrder(types=memoryview(b"ER"))

    # ...and one nesting level down: entries and mapping keys, where the bytes
    # would stringify to "b'ER'" and silently never match.
    with pytest.raises(TypeError, match="never match"):
        NeuronOrder(types=[b"ER"])
    with pytest.raises(TypeError, match="never match"):
        NeuronOrder(within={b"ER1": "id"})
    with pytest.raises(TypeError, match="never match"):
        build_ordered_neurons(
            pd.DataFrame({"root_id": ["1"], "cell_type": ["A"]}),
            "cell_type",
            ["A"],
            "root_id",
            within={b"A": "id"},
        )
    with pytest.raises(TypeError, match="must be strings"):
        ColumnOrderRule(order=[b"R1"])
    with pytest.raises(TypeError, match="never match"):
        _sort_cell_types(["ER"], preferred=[b"ER"])


def test_column_rule_pattern_must_have_a_capture_group() -> None:
    import re

    with pytest.raises(ValueError, match="capture group"):
        ColumnOrderRule(order=["R1"], pattern=re.compile(r"[LR]\d+\s*$"))
    with pytest.raises(ValueError, match="capture group"):
        ColumnOrderRule(order=["R1"], pattern=r"[LR]\d+\s*$")

    # A plain string is compiled; bytes patterns and non-patterns are rejected
    # at construction instead of crashing inside DataFrame.apply.
    compiled = ColumnOrderRule(order=["R1"], pattern=r"([LR]\d+)\s*$")
    assert compiled.pattern.search("X_R1").group(1) == "R1"
    with pytest.raises(TypeError, match="str regex"):
        ColumnOrderRule(order=["R1"], pattern=re.compile(rb"([LR]\d+)$"))
    with pytest.raises(TypeError, match="str regex"):
        ColumnOrderRule(order=["R1"], pattern=None)  # type: ignore[arg-type]


def test_within_sorters_receive_a_normalized_type_column() -> None:
    from crantpy.utils.ordering import build_ordered_neurons

    # The WithinTypeSorter doc says the frame arrives with its ID and cell
    # type columns already normalized, so a sorter filtering on the type
    # column must work even when the caller's frame holds raw values.
    annotations = pd.DataFrame({"root_id": [2, 1], "cell_type": [1, 1]})

    def sorter(name: str, rows: pd.DataFrame, id_col: str) -> list[str]:
        return sorted(rows.loc[rows["cell_type"] == name, id_col])

    ordered, boundaries = build_ordered_neurons(
        annotations, "cell_type", ["1"], "root_id", within=sorter
    )
    assert ordered == ["1", "2"]
    assert boundaries == {"1": (0, 2)}


# ---------------------------------------------------------------------------
# Read-only views: extension dtypes and derived frames
# ---------------------------------------------------------------------------


def test_extension_dtype_matrices_cannot_be_mutated_through_the_view() -> None:
    frame = pd.DataFrame([[0, 4], [2, 0]], index=["1", "2"], columns=["1", "2"]).astype(
        "Int64"
    )
    matrix = NestedMatrix(frame, {"A": (0, 2)}, ["1", "2"], {"1": "A", "2": "A"})
    cached = matrix.sum_type_matrix.copy()

    view = matrix.matrix
    with pytest.raises(ValueError, match="immutable"):
        view.loc["1", "2"] = 99
    # Chained indexing has no hook to raise through; it must land on a copy,
    # never on the internal frame backing the cached aggregates.
    try:
        view["2"]["1"] = 99
    except Exception:
        pass
    assert matrix.matrix.loc["1", "2"] == 4
    pd.testing.assert_frame_equal(
        matrix.sum_type_matrix, cached, check_frame_type=False
    )


def test_derived_frames_from_the_view_are_plain_and_editable() -> None:
    connectivity = pd.DataFrame({"pre": [1], "post": [2], "weight": [4]})
    annotations = pd.DataFrame({"root_id": [1, 2], "cell_type": ["A", "B"]})
    matrix = NestedMatrix.from_connectivity(connectivity, annotations)

    derived = matrix.matrix + 1
    assert type(derived) is pd.DataFrame
    derived["new"] = 1  # fresh buffers: must not raise "immutable"
    assert list(derived["new"]) == [1, 1]

    copied = matrix.matrix.copy()
    copied.loc["1", "2"] = 99
    assert copied.loc["1", "2"] == 99
    assert matrix.matrix.loc["1", "2"] == 4.0


# ---------------------------------------------------------------------------
# Fractional voxel offsets with integer synapse positions
# ---------------------------------------------------------------------------


@patch("crantpy.viz.mesh.load_neuropil_mesh")
def test_fractional_voxel_offset_with_integer_positions(mock_load: MagicMock) -> None:
    from crantpy.utils.config import SCALE_X, SCALE_Y, SCALE_Z

    mesh = _mock_mesh([True] * 6)
    mock_load.return_value = mesh

    result = NestedMatrix.from_synapses_by_neuropil(
        synapses_df=_make_synapses(),
        neuron_annotations=_make_annotations(),
        neuropil_names=["antennal_lobe_left"],
        coordinates="pixels",
        voxel_offset=(0.5, 0.5, 0.5),
    )

    assert "antennal_lobe_left" in result
    tested = mesh.contains.call_args[0][0]
    np.testing.assert_allclose(
        tested[0],
        [100.5 * SCALE_X, 200.5 * SCALE_Y, 300.5 * SCALE_Z],
    )


@patch("crantpy.viz.mesh.load_neuropil_mesh")
def test_directed_fractional_voxel_offset_with_integer_positions(
    mock_load: MagicMock,
) -> None:
    from crantpy.utils.config import SCALE_X, SCALE_Y, SCALE_Z

    mesh = _mock_mesh([True] * 6)
    mock_load.return_value = mesh

    result = DirectedNestedMatrix.from_synapses_by_neuropil(
        synapses_df=_make_synapses(),
        neuron_annotations=_make_annotations(),
        neuropil_names=["antennal_lobe_left"],
        coordinates="pixels",
        voxel_offset=(0.5, 0.5, 0.5),
    )

    assert "antennal_lobe_left" in result
    tested = mesh.contains.call_args[0][0]
    np.testing.assert_allclose(
        tested[0],
        [100.5 * SCALE_X, 200.5 * SCALE_Y, 300.5 * SCALE_Z],
    )
