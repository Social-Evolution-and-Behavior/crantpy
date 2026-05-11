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

from crantpy.queries import nested_connectivity_matrices as ncm
from crantpy.queries.nested_connectivity_matrices import (
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
    # synapse index 3 is unassigned
    assert "other" in result


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

    caplog.set_level("WARNING", logger="crantpy.queries.nested_connectivity_matrices")
    matrix = NestedMatrix.from_connectivity(
        adjacency,
        annotations,
        annotation_scope="all",
    )

    assert list(matrix.matrix.index) == ["1", "2"]
    assert list(matrix.matrix.columns) == ["1", "2"]
    assert "missing from annotations" in caplog.text


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
        type_order=["ExR2", "EPG/PEG", "delta7", "ER1"],
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


def test_from_connectivity_columnar_rule_can_order_non_epg_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        ncm._WITHIN_TYPE_ORDER_RULES,
        "columnar_test",
        ncm._WithinTypeOrderRule(
            label_columns=("cell_instance",),
            rank=ncm._COLUMN_ORDER_RANK,
        ),
    )
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

    matrix = NestedMatrix.from_connectivity(adjacency, annotations)

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

    caplog.set_level("WARNING", logger="crantpy.queries.nested_connectivity_matrices")
    matrix = NestedMatrix.from_connectivity(adjacency, annotations)

    assert list(matrix.matrix.index) == ["2", "4", "1", "3"]
    assert list(matrix.matrix.columns) == ["2", "4", "1", "3"]
    assert "Could not resolve a ranked column label for neuron 1" in caplog.text
    assert "Could not resolve a ranked column label for neuron 3" in caplog.text


def test_init_rejects_matrix_axis_mismatch() -> None:
    matrix = pd.DataFrame([[1, 2], [3, 4]], index=["1", "2"], columns=["1", "3"])

    with pytest.raises(ValueError, match="matrix index and columns must match exactly"):
        NestedMatrix(
            matrix=matrix,
            type_boundaries={},
            ordered_neurons=["1", "2"],
            neuron_to_type={},
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
