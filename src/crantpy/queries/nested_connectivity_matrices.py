# -*- coding: utf-8 -*-
"""
Functionality for creating and visualizing nested connectivity matrices.

The NestedMatrix class enables the construction of hierarchical connectivity matrices
that organize neurons by cell type, allowing for both neuron-level and type-level
connectivity analysis. It supports visualization with customizable boundaries,
filtering, and relative weight calculations. ROI-resolved workflows are available
through ``NestedMatrix.from_synapses_by_neuropil()``, which returns one nested
matrix per neuropil ROI.

``DirectedNestedMatrix`` provides the rectangular counterpart for workflows that
need independent source and target axes, such as analyzing ``ER_input -> ER``
without also including ``ER -> ER``.


Examples
--------
>>> import pandas as pd
>>> from crantpy.queries.nested_connectivity_matrices import NestedMatrix
>>>
>>> # Create from synapse dataframe
>>> synapses_df = pd.DataFrame({
...     'pre_pt_root_id': [1, 1, 2, 2],
...     'post_pt_root_id': [3, 4, 3, 4],
...     'Weight': [10, 20, 15, 25]
... })
>>> annotations_df = pd.DataFrame({
...     'root_id': [1, 2, 3, 4],
...     'cell_type': ['KC', 'KC', 'MB', 'MB']
... })
>>> matrix = NestedMatrix.from_synapses(
...     synapses_df,
...     annotations_df,
...     weight_mode="column",
...     weight_column="Weight",
... )
>>>
>>> # Plot the matrix
>>> matrix.plot(level="neuron")
>>>
>>> relative = NestedMatrix.from_synapses(
...     synapses_df,
...     annotations_df,
...     weight_mode="relative_outgoing",
... )
>>> # Build ROI-specific matrices using neuropil meshes
>>> matrices = NestedMatrix.from_synapses_by_neuropil(
...     synapses_df,
...     annotations_df,
...     neuropil_names=["protocerebral_bridge", "ellipsoid_body"],
...     coordinates="nm",
... )
"""

from __future__ import annotations

import os
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from typing import Any, Literal, Mapping, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

__all__ = ["NestedMatrix", "DirectedNestedMatrix", "NeuropilCollection", "All"]

_ScalarSelector = str | bytes | int | float | bool | np.integer | np.floating | np.bool_
_Selector = _ScalarSelector | Iterable[Any] | None
_TypeOrder = list[Any] | tuple[Any, ...] | None

_PURPLE_CMAP = LinearSegmentedColormap.from_list(
    "purple",
    ["#FFFFFF", "#D8B4E8", "#B366D0", "#8C1BB8", "#6A0DAD", "#4B0082", "#2E0054"],
    N=100,
)
_MAX_CENTERED_TYPE_LABELS = 200
_DEFAULT_BOUNDARY_LINEWIDTHS = {
    "outer": 3.0,
    "block_light": 4.0,
    "block_dark": 2.0,
    "grid_light": 2.0,
    "grid_dark": 2.5,
}

_COLUMN_ORDER = [
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
_COLUMN_ORDER_RANK = {label: rank for rank, label in enumerate(_COLUMN_ORDER)}
_COLUMN_LABEL_PATTERN = re.compile(r"([LR]\d+)\s*$")
_READ_ONLY_MESSAGE = "NestedMatrix data is immutable; call .copy() before editing"


class _ReadOnlyIndexer:
    """Read-only wrapper for pandas indexers used by public matrix views."""

    def __init__(self, indexer: Any):
        self._indexer = indexer

    def __getitem__(self, key: Any) -> Any:
        return self._indexer[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        raise ValueError(_READ_ONLY_MESSAGE)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._indexer, name)


class _ReadOnlyDataFrame(pd.DataFrame):
    """DataFrame view that rejects common in-place mutation paths."""

    @property
    def _constructor(self):
        return _ReadOnlyDataFrame

    @property
    def loc(self) -> _ReadOnlyIndexer:
        return _ReadOnlyIndexer(super().loc)

    @property
    def iloc(self) -> _ReadOnlyIndexer:
        return _ReadOnlyIndexer(super().iloc)

    @property
    def at(self) -> _ReadOnlyIndexer:
        return _ReadOnlyIndexer(super().at)

    @property
    def iat(self) -> _ReadOnlyIndexer:
        return _ReadOnlyIndexer(super().iat)

    def __setitem__(self, key: Any, value: Any) -> None:
        raise ValueError(_READ_ONLY_MESSAGE)

    def __delitem__(self, key: Any) -> None:
        raise ValueError(_READ_ONLY_MESSAGE)

    def insert(self, *args: Any, **kwargs: Any) -> None:
        raise ValueError(_READ_ONLY_MESSAGE)

    def pop(self, item: Any) -> Any:
        raise ValueError(_READ_ONLY_MESSAGE)

    def update(self, *args: Any, **kwargs: Any) -> None:
        raise ValueError(_READ_ONLY_MESSAGE)

    def copy(self, deep: bool = True) -> pd.DataFrame:
        result = pd.DataFrame(self).copy(deep=deep)
        if deep:
            _set_dataframe_writeable(result, writeable=True)
        return result


def _set_dataframe_writeable(df: pd.DataFrame, writeable: bool) -> None:
    """Set NumPy-backed pandas blocks to writeable or read-only when possible."""

    for array in df._mgr.arrays:
        if hasattr(array, "setflags"):
            try:
                array.setflags(write=writeable)
            except ValueError:
                if not writeable:
                    raise


def _readonly_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a read-only DataFrame view over an immutable private DataFrame."""

    _set_dataframe_writeable(df, writeable=False)
    view = _ReadOnlyDataFrame(df.copy(deep=False))
    _set_dataframe_writeable(view, writeable=False)
    return view


@dataclass(frozen=True)
class _WithinTypeOrderRule:
    """Declarative ordering rule for columnar neuron types."""

    label_columns: tuple[str, ...]
    rank: Mapping[str, int]
    pattern: re.Pattern[str] = _COLUMN_LABEL_PATTERN


_WITHIN_TYPE_ORDER_RULES: dict[str, _WithinTypeOrderRule] = {
    "EPG/PEG": _WithinTypeOrderRule(
        # cell_instance is a forward-compatible placeholder if explicit
        # instance annotations are added; current annotations fall back to cell_subtype.
        label_columns=("cell_instance", "cell_subtype"),
        rank=_COLUMN_ORDER_RANK,
    )
}


class _AllSelector:
    """Selector for choosing all neuropils, optionally excluding some.

    Examples
    --------
    >>> matrices.plot(All)                          # plot every neuropil
    >>> matrices.plot(All.minus("fan_shaped_body"))  # all except one
    >>> matrices.plot(All.minus("fan_shaped_body", "protocerebral_bridge"))         # all except several
    """

    def __init__(self, exclude: frozenset[str] = frozenset()):
        self._exclude = exclude

    def minus(self, *names: str) -> "_AllSelector":
        """Return a new selector that excludes the given neuropil name(s)."""
        return _AllSelector(self._exclude | frozenset(names))

    def resolve(self, available: list[str]) -> list[str]:
        return [n for n in available if n not in self._exclude]

    def __repr__(self) -> str:
        if self._exclude:
            return f"All.minus({', '.join(repr(n) for n in sorted(self._exclude))})"
        return "All"


All = _AllSelector()


class _ResolvedAnnotations(NamedTuple):
    """Result of resolving neuron annotations against matrix IDs."""

    relevant: pd.DataFrame
    typed: pd.DataFrame
    id_map: dict[str, Any]
    untyped_ids: list[str]
    missing_ids: list[str]


class _PlotData(NamedTuple):
    """Data prepared for a single plot call."""

    matrix: pd.DataFrame
    boundaries: dict[str, tuple[int, int]]
    labels: list[str]


@dataclass(frozen=True)
class _AxisOrdering:
    """Ordered neurons and type metadata for one matrix axis."""

    ordered_neurons: tuple[str, ...]
    type_boundaries: dict[str, tuple[int, int]]
    neuron_to_type: dict[str, Any]


class _DirectedPlotData(NamedTuple):
    """Data prepared for a rectangular directed plot call."""

    matrix: pd.DataFrame
    source_boundaries: dict[str, tuple[int, int]]
    target_boundaries: dict[str, tuple[int, int]]
    source_labels: list[str]
    target_labels: list[str]


class NeuropilCollection(dict):
    """Dict subclass mapping neuropil names to nested matrix instances.

    Provides convenience methods for accessing and plotting individual
    neuropil/ROI matrices with cleaner syntax.

    Examples
    --------
    >>> matrices = NestedMatrix.from_synapses_by_neuropil(...)
    >>> matrices.plot("protocerebral_bridge", level="neuron")
    >>> matrices.plot(All)
    >>> matrices.plot(All.minus("fan_shaped_body"))
    >>> matrices.protocerebral_bridge.sum_type_matrix
    """

    def plot(
        self, name: str | _AllSelector, **kwargs
    ) -> tuple[Figure, Axes] | dict[str, tuple[Figure, Axes]]:
        """Plot connectivity matrix/matrices.

        Parameters
        ----------
        name : str or All selector
            A single neuropil name, or ``All`` / ``All.minus(...)`` to
            plot multiple neuropils at once.
        **kwargs
            Forwarded to the contained matrix object's ``plot()`` method.

        Returns
        -------
        tuple[Figure, Axes]
            When *name* is a single neuropil string.
        dict[str, tuple[Figure, Axes]]
            When *name* is an ``All`` selector.
        """
        if isinstance(name, _AllSelector):
            names = name.resolve(list(self.keys()))
            return {n: self[n].plot(**kwargs) for n in names}
        return self[name].plot(**kwargs)

    def __getattr__(self, name: str) -> "NestedMatrix | DirectedNestedMatrix":
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"No neuropil named {name!r}") from None

    def __repr__(self) -> str:
        names = list(self.keys())
        return f"NeuropilCollection({names})"


class NestedMatrix:
    """A nested connectivity matrix organized by cell type.

    This class provides functionality to create, manipulate, and visualize
    connectivity matrices where neurons are grouped by their cell type annotations.
    The matrix maintains both neuron-level and type-level connectivity information,
    allowing for hierarchical visualization of connectivity patterns.

    .. note::

       Instances are immutable after construction. Public attributes expose
       read-only views; use ``.copy()``, ``dict(...)``, or ``list(...)`` when
       mutable data is needed. Derived type matrices are cached against the
       immutable private state.

    Attributes
    ----------
    matrix : pd.DataFrame
        Full neuron-to-neuron connectivity matrix with neurons ordered by type.
    type_boundaries : Mapping[str, tuple[int, int]]
        Dictionary mapping cell type names to (start, end) index tuples
        indicating where each type appears in the matrix.
    ordered_neurons : tuple[str, ...]
        Tuple of neuron IDs in the order they appear in the matrix.
    neuron_to_type : Mapping[str, Any]
        Dictionary mapping neuron IDs to their cell type annotations.

    Examples
    --------
    >>> matrix = NestedMatrix.from_connectivity(
    ...     connections_df=adjacency_df,
    ...     neuron_annotations=annotations_df
    ... )
    >>> type_matrix = matrix.sum_type_matrix
    >>> matrix.plot(level="type_mean")
    """

    def __init__(
        self,
        matrix: pd.DataFrame,
        type_boundaries: Mapping[Any, tuple[int, int]],
        ordered_neurons: Iterable[Any],
        neuron_to_type: Mapping[Any, Any],
    ):
        matrix = matrix.copy()
        matrix.index = self._stringify_id_axis(matrix.index, "matrix index")
        matrix.columns = self._stringify_id_axis(matrix.columns, "matrix columns")
        ordered_neurons = list(
            self._stringify_id_axis(ordered_neurons, "ordered_neurons")
        )
        type_boundaries = {
            self._stringify_id_value(name): (int(start), int(end))
            for name, (start, end) in type_boundaries.items()
        }
        neuron_to_type = {
            self._stringify_id_value(neuron): (
                cell_type
                if self._is_missing_scalar(cell_type)
                else self._stringify_id_value(cell_type)
            )
            for neuron, cell_type in neuron_to_type.items()
            if not self._is_missing_scalar(neuron)
        }

        self._validate_invariants(
            matrix=matrix,
            type_boundaries=type_boundaries,
            ordered_neurons=ordered_neurons,
            neuron_to_type=neuron_to_type,
        )

        _set_dataframe_writeable(matrix, writeable=False)
        self._matrix = matrix
        self._type_boundaries = type_boundaries
        self._ordered_neurons = tuple(ordered_neurons)
        self._neuron_to_type = neuron_to_type

    @property
    def matrix(self) -> pd.DataFrame:
        """Read-only neuron-to-neuron connectivity matrix."""
        return _readonly_dataframe(self._matrix)

    @property
    def type_boundaries(self) -> Mapping[str, tuple[int, int]]:
        """Read-only mapping from cell type to its matrix slice."""
        return MappingProxyType(self._type_boundaries)

    @property
    def ordered_neurons(self) -> tuple[str, ...]:
        """Read-only neuron order used by both matrix axes."""
        return self._ordered_neurons

    @property
    def neuron_to_type(self) -> Mapping[str, Any]:
        """Read-only mapping from neuron ID to cell type."""
        return MappingProxyType(self._neuron_to_type)

    def __repr__(self) -> str:
        n_neurons = len(self._ordered_neurons)
        n_types = len(self._type_boundaries)
        types = list(self._type_boundaries.keys())
        if len(types) > 5:
            types_str = ", ".join(types[:5]) + f", ... ({n_types} total)"
        else:
            types_str = ", ".join(types)
        return f"NestedMatrix({n_neurons} neurons, types=[{types_str}])"

    @staticmethod
    def _validate_invariants(
        matrix: pd.DataFrame,
        type_boundaries: dict[str, tuple[int, int]],
        ordered_neurons: list[str],
        neuron_to_type: dict[str, Any],
    ) -> None:
        nrows, ncols = matrix.shape
        if nrows != ncols:
            raise ValueError(f"matrix must be square, got shape {matrix.shape}")

        matrix_index = list(matrix.index)
        matrix_columns = list(matrix.columns)
        duplicate_neurons = NestedMatrix._find_duplicates(ordered_neurons)
        if duplicate_neurons:
            raise ValueError(
                "ordered_neurons contains duplicate neuron IDs after normalization: "
                f"{duplicate_neurons[:10]}"
            )

        if matrix_index != matrix_columns:
            raise ValueError("matrix index and columns must match exactly")

        if matrix_index != ordered_neurons:
            raise ValueError("matrix axes must match ordered_neurons exactly")

        extra_neurons = sorted(set(neuron_to_type) - set(ordered_neurons))
        if extra_neurons:
            raise ValueError(
                "neuron_to_type contains neurons not present in ordered_neurons: "
                f"{extra_neurons[:10]}"
            )

        expected_start = 0
        for name, (start, end) in type_boundaries.items():
            if start != expected_start:
                raise ValueError(
                    "type_boundaries must be contiguous, non-overlapping, and start at 0"
                )
            if start < 0 or end > len(ordered_neurons) or end <= start:
                raise ValueError(
                    f"type boundary {name!r} has invalid slice ({start}, {end})"
                )

            boundary_neurons = ordered_neurons[start:end]
            mismatched = [
                neuron
                for neuron in boundary_neurons
                if str(neuron_to_type.get(neuron)) != name
            ]
            if mismatched:
                raise ValueError(
                    f"type boundary {name!r} does not match neuron_to_type for "
                    f"neurons {mismatched[:10]}"
                )

            expected_start = end

        annotated_after_boundaries = [
            neuron
            for neuron in ordered_neurons[expected_start:]
            if pd.notna(neuron_to_type.get(neuron))
        ]
        if annotated_after_boundaries:
            raise ValueError(
                "annotated neurons must appear within type_boundaries before any "
                "unassigned neurons"
            )

    @staticmethod
    def _log_nested_ordering(
        logger: logging.Logger,
        ordered_neurons: list[str],
        boundaries: dict[str, tuple[int, int]],
    ) -> None:
        logger.info(
            "Resolved nested ordering with %d type(s) and %d neuron(s)",
            len(boundaries),
            len(ordered_neurons),
        )
        logger.info("Resolved type order: %s", list(boundaries.keys()))
        logger.debug("Full neuron order: %s", ordered_neurons)

        for name, (start, end) in boundaries.items():
            logger.debug(
                "Type %s occupies ordered_neurons[%d:%d] with %d neuron(s): %s",
                name,
                start,
                end,
                end - start,
                ordered_neurons[start:end],
            )

    @classmethod
    def from_connectivity(
        cls,
        connections_df: pd.DataFrame,
        neuron_annotations: pd.DataFrame,
        cell_type_column: str = "cell_type",
        neuron_id_column: str = "root_id",
        type_order: _TypeOrder = None,
        annotation_scope: Literal["annotated_only", "all"] = "annotated_only",
    ) -> NestedMatrix:
        """Create a NestedMatrix from a connectivity dataframe that you can get with cp.get_connectivity().

        Constructs a nested connectivity matrix from an adjacency or edge-list
        dataframe, organizing neurons by their cell type annotations. Type
        blocks follow ``type_order`` when provided, otherwise a generic
        label-aware sort is used. Within each type block, neurons preserve the
        resolved annotation row order, except ``"EPG/PEG"`` rows which use the
        EB column order encoded in ``cell_instance`` or ``cell_subtype`` when
        available.

        Parameters
        ----------
        connections_df : pd.DataFrame
            Connectivity data in one of the following formats:
            - Adjacency matrix (index and columns are neuron IDs)
            - Edge list with columns ['type.from', 'type.to', 'weight']
            - Edge list with columns ['pre', 'post', 'weight']
            - Edge list with columns ['source', 'target', 'weight'] or ['source', 'target', 'n_syn']
              where each row is already aggregated to a unique source-target pair,
              such as the output of ``cp.get_connectivity()``
        neuron_annotations : pd.DataFrame
            DataFrame containing neuron annotations with at least the neuron ID
            and cell type columns.
        cell_type_column : str, default "cell_type"
            Name of the column in neuron_annotations containing cell type labels.
        neuron_id_column : str, default "root_id"
            Name of the column in neuron_annotations containing neuron IDs.
        type_order : list or tuple, optional
            Preferred order for cell types. Values are normalized like annotation
            labels. Types will be sorted alphabetically if not provided, with
            numeric suffixes handled intelligently.
        annotation_scope : {"annotated_only", "all"}, default "annotated_only"
            Controls which neurons are retained. ``"annotated_only"`` keeps
            only neurons present in ``neuron_annotations``. ``"all"`` keeps
            all neurons from the connectivity input, appending neurons missing
            annotations after the typed blocks.

        Returns
        -------
        NestedMatrix
            A new NestedMatrix instance with neurons organized by type.

        Examples
        --------
        >>> adjacency = pd.DataFrame({
        ...     1: [0, 10, 0], 2: [5, 0, 15], 3: [0, 20, 0]
        ... }, index=[1, 2, 3])
        >>> annotations = pd.DataFrame({
        ...     'root_id': [1, 2, 3],
        ...     'cell_type': ['ER', 'ER', 'Pbt']
        ... })
        >>> matrix = NestedMatrix.from_connectivity(adjacency, annotations)
        """
        logger = logging.getLogger(__name__)
        cls._validate_annotation_scope(annotation_scope)
        logger.info(
            "Building NestedMatrix from connectivity: input_shape=%s, "
            "annotation_rows=%d, cell_type_column=%s, neuron_id_column=%s, "
            "annotation_scope=%s",
            connections_df.shape,
            len(neuron_annotations),
            cell_type_column,
            neuron_id_column,
            annotation_scope,
        )
        if type_order:
            logger.info(
                "Requested type order (%d entries): %s",
                len(type_order),
                type_order,
            )
        else:
            logger.info("No explicit type order provided; inferring from annotations")

        adjacency = cls._coerce_to_adjacency(connections_df)
        adjacency = cls._apply_annotation_scope_to_adjacency(
            adjacency=adjacency,
            neuron_annotations=neuron_annotations,
            neuron_id_column=neuron_id_column,
            annotation_scope=annotation_scope,
        )
        logger.info(
            "Connectivity coerced to adjacency matrix with shape=%s and %d nonzero value(s)",
            adjacency.shape,
            int(np.count_nonzero(adjacency.to_numpy())),
        )

        ordered_neurons, boundaries, neuron_map = cls._align_neurons_and_boundaries(
            adjacency=adjacency,
            annotations=neuron_annotations,
            id_col=neuron_id_column,
            type_col=cell_type_column,
            type_order=type_order,
        )

        if not ordered_neurons:
            logger.info(
                "No overlapping neurons found between connectivity and annotations; "
                "returning empty NestedMatrix"
            )
            return cls(pd.DataFrame(), {}, [], neuron_map)

        matrix = adjacency.reindex(
            index=ordered_neurons, columns=ordered_neurons, fill_value=0
        ).astype(float)

        cls._log_nested_ordering(logger, ordered_neurons, boundaries)
        logger.info(
            "Constructed NestedMatrix from connectivity with matrix_shape=%s",
            matrix.shape,
        )

        return cls(
            matrix=matrix,
            type_boundaries=boundaries,
            ordered_neurons=ordered_neurons,
            neuron_to_type=neuron_map,
        )

    @classmethod
    def from_synapses(
        cls,
        synapses_df: pd.DataFrame,
        neuron_annotations: pd.DataFrame,
        pre_col: str = "pre_pt_root_id",
        post_col: str = "post_pt_root_id",
        weight_mode: Literal[
            "relative_outgoing", "relative_incoming", "count", "column"
        ] = "relative_outgoing",
        weight_column: str | None = None,
        cell_type_column: str = "cell_type",
        neuron_id_column: str = "root_id",
        type_order: _TypeOrder = None,
        annotation_scope: Literal["annotated_only", "all"] = "annotated_only",
    ) -> NestedMatrix:
        """Create a NestedMatrix from a synapse dataframe.

        Aggregates synapse-level data into a connectivity matrix and organizes
        neurons by cell type. This constructor returns a single matrix over the
        supplied synapses. For ROI-specific outputs, pre-filter ``synapses_df``
        to the ROI of interest or use ``from_synapses_by_neuropil()`` to build
        one matrix per neuropil ROI. Ordering semantics match
        ``from_connectivity()``: ``type_order`` controls type blocks, and
        neurons within each type preserve resolved annotation row order except
        for ``"EPG/PEG"`` rows, which use EB columns from ``cell_instance`` or
        ``cell_subtype`` when available.

        Parameters
        ----------
        synapses_df : pd.DataFrame
            DataFrame containing synapse data with pre- and post-synaptic
            neuron IDs. Must contain columns for presynaptic and postsynaptic IDs.
        neuron_annotations : pd.DataFrame
            DataFrame containing neuron annotations with at least the neuron ID
            and cell type columns.
        pre_col : str, default "pre_pt_root_id"
            Name of the column in synapses_df containing presynaptic neuron IDs.
        post_col : str, default "post_pt_root_id"
            Name of the column in synapses_df containing postsynaptic neuron IDs.
        weight_mode : {"relative_outgoing", "relative_incoming", "count", "column"}, default "relative_outgoing"
            How to derive edge weights from synapse rows.
            ``"relative_outgoing"`` counts synapses per pair and normalizes each
            presynaptic neuron's outgoing row to sum to 1.
            ``"relative_incoming"`` counts synapses per pair and normalizes each
            postsynaptic neuron's incoming column to sum to 1.
            ``"count"`` uses raw synapse counts per pair.
            ``"column"`` sums the values from ``weight_column`` per pair.
        weight_column : str | None, optional
            Column to sum when ``weight_mode="column"``. Must be provided for
            column-based weighting and omitted otherwise.
        cell_type_column : str, default "cell_type"
            Name of the column in neuron_annotations containing cell type labels.
        neuron_id_column : str, default "root_id"
            Name of the column in neuron_annotations containing neuron IDs.
        type_order : list or tuple, optional
            Preferred order for cell types. Values are normalized like annotation
            labels. Types will be sorted alphabetically if not provided, with
            numeric suffixes handled intelligently.
        annotation_scope : {"annotated_only", "all"}, default "annotated_only"
            Controls which synapse rows contribute to the matrix.
            ``"annotated_only"`` keeps only rows whose pre/post neurons are both
            present in ``neuron_annotations``.
            ``"all"`` keeps all rows and appends missing/untyped neurons after
            the typed blocks.

        Returns
        -------
        NestedMatrix
            A new NestedMatrix instance with neurons organized by type across
            the supplied synapses.

        Examples
        --------
        >>> synapses = pd.DataFrame({
        ...     'pre_pt_root_id': [1, 1, 2],
        ...     'post_pt_root_id': [3, 4, 3],
        ...     'Weight': [10, 20, 15]
        ... })
        >>> annotations = pd.DataFrame({
        ...     'root_id': [1, 2, 3, 4],
        ...     'cell_type': ['KC', 'KC', 'MB', 'MB']
        ... })
        >>> matrix = NestedMatrix.from_synapses(
        ...     synapses,
        ...     annotations,
        ...     weight_mode="column",
        ...     weight_column="Weight",
        ... )
        >>> relative = NestedMatrix.from_synapses(
        ...     synapses,
        ...     annotations,
        ...     weight_mode="relative_outgoing",
        ... )
        >>> # For neuropil ROI-specific matrices, use:
        >>> # NestedMatrix.from_synapses_by_neuropil(...)
        """
        logger = logging.getLogger(__name__)
        cls._validate_weighting(weight_mode, weight_column)
        cls._validate_annotation_scope(annotation_scope)
        logger.info(
            "Building NestedMatrix from synapses: synapse_rows=%d, "
            "annotation_rows=%d, pre_col=%s, post_col=%s, weight_mode=%s, "
            "weight_column=%s, annotation_scope=%s",
            len(synapses_df),
            len(neuron_annotations),
            pre_col,
            post_col,
            weight_mode,
            weight_column,
            annotation_scope,
        )
        synapses_df = cls._apply_annotation_scope_to_synapses(
            synapses_df=synapses_df,
            neuron_annotations=neuron_annotations,
            neuron_id_column=neuron_id_column,
            pre_col=pre_col,
            post_col=post_col,
            annotation_scope=annotation_scope,
        )
        synapses_df = cls._normalize_synapse_edge_ids(synapses_df, pre_col, post_col)
        df = cls._aggregate_synapse_edges(
            synapses_df=synapses_df,
            pre_col=pre_col,
            post_col=post_col,
            weight_mode=weight_mode,
            weight_column=weight_column,
        )
        logger.info(
            "Aggregated synapses into %d weighted edge(s) before matrix construction",
            len(df),
        )

        matrix = cls.from_connectivity(
            connections_df=df,
            neuron_annotations=neuron_annotations,
            cell_type_column=cell_type_column,
            neuron_id_column=neuron_id_column,
            type_order=type_order,
            annotation_scope=annotation_scope,
        )
        logger.info(
            "Constructed NestedMatrix from synapses with matrix_shape=%s",
            matrix._matrix.shape,
        )
        return matrix

    @classmethod
    def from_synapses_by_neuropil(
        cls,
        synapses_df: pd.DataFrame,
        neuron_annotations: pd.DataFrame,
        neuropil_names: list[str] | None = None,
        coordinates: str = "nm",
        position_column: str = "ctr_pt_position",
        pre_col: str = "pre_pt_root_id",
        post_col: str = "post_pt_root_id",
        weight_mode: Literal[
            "relative_outgoing", "relative_incoming", "count", "column"
        ] = "relative_outgoing",
        weight_column: str | None = None,
        cell_type_column: str = "cell_type",
        neuron_id_column: str = "root_id",
        type_order: _TypeOrder = None,
        annotation_scope: Literal["annotated_only", "all"] = "annotated_only",
        include_other: bool = True,
        voxel_offset: tuple[float, float, float] | None = None,
    ) -> "NeuropilCollection":
        """Create NestedMatrix instances per neuropil using mesh containment.

        Assigns each synapse to one or more neuropil ROIs by testing its
        spatial coordinates against neuropil meshes, then builds a separate
        NestedMatrix for each neuropil ROI that contains synapses. Weighting,
        annotation scope, and ordering semantics match ``from_synapses()``
        within each ROI-specific subset.

        Parameters
        ----------
        synapses_df : pd.DataFrame
            DataFrame containing synapse data. Must include the position column
            and pre/post neuron ID columns.
        neuron_annotations : pd.DataFrame
            DataFrame with neuron ID and cell type columns.
        neuropil_names : list[str] | None, optional
            Neuropil mesh names to test against (values from NEUROPIL_MESH_DICT).
            If None, all available neuropils are used.
        coordinates : str, default "nm"
            Coordinate system of the position column. ``"nm"`` for nanometers
            (meshes are in nm space), ``"pixels"`` to auto-convert using scale
            factors.
        position_column : str, default "ctr_pt_position"
            Column containing [x, y, z] coordinates for containment testing.
        pre_col : str, default "pre_pt_root_id"
            Column with presynaptic neuron IDs.
        post_col : str, default "post_pt_root_id"
            Column with postsynaptic neuron IDs.
        weight_mode : {"relative_outgoing", "relative_incoming", "count", "column"}, default "relative_outgoing"
            How to derive edge weights from synapse rows within each neuropil
            subset. Semantics match ``from_synapses()``.
        weight_column : str | None, optional
            Column to sum when ``weight_mode="column"``.
        cell_type_column : str, default "cell_type"
            Column in neuron_annotations with cell type labels.
        neuron_id_column : str, default "root_id"
            Column in neuron_annotations with neuron IDs.
        type_order : list or tuple, optional
            Preferred order for cell types in each matrix. Values are normalized
            like annotation labels.
        annotation_scope : {"annotated_only", "all"}, default "annotated_only"
            Controls which synapse rows contribute to the ROI-specific matrices.
            ``"annotated_only"`` keeps only rows whose pre/post neurons are both
            present in ``neuron_annotations`` before ROI assignment.
            ``"all"`` keeps all rows and appends missing/untyped neurons after
            the typed blocks within each ROI matrix.
        include_other : bool, default True
            If True, synapses not inside any neuropil mesh are collected
            under the ``"other"`` key.
        voxel_offset : tuple[float, float, float] | None, optional
            Offset to add to pixel coordinates before converting to nm,
            to align synapse positions with the neuropil mesh coordinate
            space. Only applied when ``coordinates="pixels"``.

        Returns
        -------
        NeuropilCollection
            Dict-like collection mapping neuropil ROI name (and optionally
            ``"other"``) to a NestedMatrix built from the synapses in that
            region. Supports ``collection.plot(name, **kwargs)`` shortcut
            and attribute access (e.g., ``collection.ellipsoid_body``).

        Raises
        ------
        ValueError
            If an invalid neuropil name or coordinates value is provided.

        Examples
        --------
        >>> matrices = NestedMatrix.from_synapses_by_neuropil(
        ...     synapses_df=synapses,
        ...     neuron_annotations=annotations,
        ...     neuropil_names=["antennal_lobe_left", "mushroom_body_pedunculus_and_lobes_left"],
        ...     coordinates="nm",
        ... )
        >>> for name, mat in matrices.items():
        ...     print(name, mat.sum_type_matrix.shape)
        """
        import logging

        from crantpy.utils.config import (
            NEUROPIL_MESH_DICT,
            SCALE_X,
            SCALE_Y,
            SCALE_Z,
        )
        from crantpy.viz.mesh import (
            get_supported_neuropil_mesh_labels,
            load_neuropil_mesh,
        )

        logger = logging.getLogger(__name__)
        cls._validate_weighting(weight_mode, weight_column)
        cls._validate_annotation_scope(annotation_scope)
        logger.info(
            "Building NestedMatrix collection by neuropil: synapse_rows=%d, "
            "annotation_rows=%d, coordinates=%s, include_other=%s, "
            "weight_mode=%s, annotation_scope=%s",
            len(synapses_df),
            len(neuron_annotations),
            coordinates,
            include_other,
            weight_mode,
            annotation_scope,
        )

        if neuropil_names is None:
            neuropil_names = list(NEUROPIL_MESH_DICT.values())
        else:
            supported = get_supported_neuropil_mesh_labels()
            invalid = [n for n in neuropil_names if n not in supported]
            if invalid:
                raise ValueError(
                    f"Invalid neuropil name(s): {invalid}. " f"Available: {supported}"
                )
        logger.info(
            "Using %d neuropil mask(s): %s",
            len(neuropil_names),
            neuropil_names,
        )

        if synapses_df.empty:
            logger.info(
                "Synapse dataframe is empty; returning empty NeuropilCollection"
            )
            return NeuropilCollection()

        synapses_df = cls._apply_annotation_scope_to_synapses(
            synapses_df=synapses_df,
            neuron_annotations=neuron_annotations,
            neuron_id_column=neuron_id_column,
            pre_col=pre_col,
            post_col=post_col,
            annotation_scope=annotation_scope,
        )
        if synapses_df.empty and annotation_scope == "annotated_only":
            logger.info(
                "No synapses remain after filtering to annotated neurons; "
                "returning empty NeuropilCollection"
            )
            return NeuropilCollection()

        positions = np.vstack(synapses_df[position_column].values)

        if coordinates == "pixels":
            if voxel_offset is not None:
                positions += np.array(voxel_offset)
            positions[:, 0] = positions[:, 0] * SCALE_X
            positions[:, 1] = positions[:, 1] * SCALE_Y
            positions[:, 2] = positions[:, 2] * SCALE_Z
        elif coordinates != "nm":
            raise ValueError(
                f"coordinates must be 'nm' or 'pixels', got {coordinates!r}"
            )

        neuropil_masks: dict[str, np.ndarray] = {}
        for name in neuropil_names:
            logger.info("Loading neuropil mesh: %s", name)
            mesh = load_neuropil_mesh(name)
            logger.info("Testing %d points against %s", len(positions), name)
            neuropil_masks[name] = mesh.contains(positions)
            logger.info(
                "Neuropil %s contains %d synapse(s)",
                name,
                int(neuropil_masks[name].sum()),
            )

        if include_other:
            any_assigned = np.zeros(len(positions), dtype=bool)
            for mask in neuropil_masks.values():
                any_assigned |= mask
            other_mask = ~any_assigned
            if other_mask.any():
                neuropil_masks["other"] = other_mask
                logger.info(
                    'Neuropil "other" contains %d synapse(s)',
                    int(other_mask.sum()),
                )

        result = NeuropilCollection()
        for name, mask in neuropil_masks.items():
            if not mask.any():
                logger.debug(
                    "Skipping neuropil %s because it contains no synapses", name
                )
                continue
            subset_df = synapses_df[mask]
            logger.info(
                "Building NestedMatrix for neuropil %s with %d synapse(s)",
                name,
                len(subset_df),
            )
            result[name] = cls.from_synapses(
                synapses_df=subset_df,
                neuron_annotations=neuron_annotations,
                pre_col=pre_col,
                post_col=post_col,
                weight_mode=weight_mode,
                weight_column=weight_column,
                cell_type_column=cell_type_column,
                neuron_id_column=neuron_id_column,
                type_order=type_order,
                annotation_scope=annotation_scope,
            )
            logger.info(
                "Constructed NestedMatrix for neuropil %s with matrix_shape=%s",
                name,
                result[name]._matrix.shape,
            )

        return result

    @classmethod
    def _aggregate_synapse_edges(
        cls,
        synapses_df: pd.DataFrame,
        pre_col: str,
        post_col: str,
        weight_mode: Literal[
            "relative_outgoing", "relative_incoming", "count", "column"
        ],
        weight_column: str | None,
    ) -> pd.DataFrame:
        df = synapses_df.rename(columns={pre_col: "type.from", post_col: "type.to"})

        def _relative_outgoing(df: pd.DataFrame) -> pd.DataFrame:
            edges = cls._count_synapse_pairs(df)
            source_totals = edges.groupby("type.from")["weight"].transform("sum")
            edges["weight"] = edges["weight"] / source_totals.replace(0, 1)
            return edges

        def _relative_incoming(df: pd.DataFrame) -> pd.DataFrame:
            edges = cls._count_synapse_pairs(df)
            target_totals = edges.groupby("type.to")["weight"].transform("sum")
            edges["weight"] = edges["weight"] / target_totals.replace(0, 1)
            return edges

        def _count(df: pd.DataFrame) -> pd.DataFrame:
            return cls._count_synapse_pairs(df)

        def _column(df: pd.DataFrame) -> pd.DataFrame:
            if weight_column not in df.columns:
                raise ValueError(
                    f"weight_column {weight_column!r} not found in synapses_df"
                )
            df = cls._coerce_weight_values(df, weight_column)
            return (
                df.rename(columns={weight_column: "weight"})
                .groupby(["type.from", "type.to"], as_index=False)["weight"]
                .sum()
            )

        aggregators = {
            "relative_outgoing": _relative_outgoing,
            "relative_incoming": _relative_incoming,
            "count": _count,
            "column": _column,
        }
        return aggregators[weight_mode](df)

    @staticmethod
    def _count_synapse_pairs(df: pd.DataFrame) -> pd.DataFrame:
        """Count synapses per (type.from, type.to) pair."""
        if "id" in df.columns:
            # Match get_connectivity(): the same synapse can appear twice when
            # upstream/downstream query results are combined.
            df = df.drop_duplicates("id")

        return (
            df.groupby(["type.from", "type.to"])
            .size()
            .reset_index(name="weight")  # type: ignore
        )

    @staticmethod
    def _annotation_id_set(
        neuron_annotations: pd.DataFrame, neuron_id_column: str
    ) -> set[str]:
        return set(
            neuron_annotations[neuron_id_column]
            .dropna()
            .map(NestedMatrix._stringify_id_value)
        )

    @staticmethod
    def _is_missing_scalar(value: Any) -> bool:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)):
            return bool(missing)
        return False

    @staticmethod
    def _stringify_id_value(value: Any) -> str:
        if isinstance(value, (bool, np.bool_)):
            return str(value)
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            value_float = float(value)
            if np.isfinite(value_float) and value_float.is_integer():
                return str(int(value_float))
        return str(value)

    @staticmethod
    def _stringify_id_axis(values: Iterable[Any], axis_name: str) -> pd.Index:
        ids: list[str] = []
        for value in values:
            if NestedMatrix._is_missing_scalar(value):
                raise ValueError(f"{axis_name} contains null neuron IDs")
            ids.append(NestedMatrix._stringify_id_value(value))
        return pd.Index(ids)

    @staticmethod
    def _normalize_id_values(values: Iterable[Any]) -> list[str]:
        return [
            NestedMatrix._stringify_id_value(value)
            for value in values
            if not NestedMatrix._is_missing_scalar(value)
        ]

    @staticmethod
    def _find_duplicates(values: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        duplicates: list[str] = []
        duplicate_seen: set[str] = set()
        for value in values:
            if value in seen and value not in duplicate_seen:
                duplicates.append(value)
                duplicate_seen.add(value)
            seen.add(value)
        return duplicates

    @staticmethod
    def _normalize_adjacency_axes(adjacency: pd.DataFrame) -> pd.DataFrame:
        adjacency = adjacency.copy()
        row_keep = [
            not NestedMatrix._is_missing_scalar(value) for value in adjacency.index
        ]
        col_keep = [
            not NestedMatrix._is_missing_scalar(value) for value in adjacency.columns
        ]
        if not all(row_keep) or not all(col_keep):
            adjacency = adjacency.loc[row_keep, col_keep].copy()

        adjacency.index = NestedMatrix._stringify_id_axis(
            adjacency.index, "connectivity row index"
        )
        adjacency.columns = NestedMatrix._stringify_id_axis(
            adjacency.columns, "connectivity columns"
        )

        if adjacency.index.has_duplicates:
            adjacency = adjacency.groupby(level=0, sort=False).sum()
        if adjacency.columns.has_duplicates:
            adjacency = adjacency.T.groupby(level=0, sort=False).sum().T

        return adjacency

    @staticmethod
    def _coerce_weight_values(df: pd.DataFrame, weight_column: str) -> pd.DataFrame:
        df = df.copy()
        try:
            df[weight_column] = pd.to_numeric(df[weight_column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Connectivity weight column {weight_column!r} must be numeric"
            ) from exc
        return df

    @staticmethod
    def _drop_null_edge_ids(
        df: pd.DataFrame,
        source_col: str,
        target_col: str,
    ) -> pd.DataFrame:
        keep_mask = df[source_col].notna() & df[target_col].notna()
        return df.loc[keep_mask].copy()

    @classmethod
    def _normalize_synapse_edge_ids(
        cls,
        synapses_df: pd.DataFrame,
        pre_col: str,
        post_col: str,
    ) -> pd.DataFrame:
        normalized = cls._drop_null_edge_ids(synapses_df, pre_col, post_col)
        if not normalized.empty:
            normalized[pre_col] = normalized[pre_col].map(cls._stringify_id_value)
            normalized[post_col] = normalized[post_col].map(cls._stringify_id_value)
        return normalized

    @staticmethod
    def _validate_annotation_scope(
        annotation_scope: Literal["annotated_only", "all"] | str,
    ) -> None:
        valid_scopes = {"annotated_only", "all"}
        if annotation_scope not in valid_scopes:
            raise ValueError(
                f"annotation_scope must be one of {sorted(valid_scopes)}, got {annotation_scope!r}"
            )

    @staticmethod
    def _validate_weighting(
        weight_mode: (
            Literal["relative_outgoing", "relative_incoming", "count", "column"] | str
        ),
        weight_column: str | None,
    ) -> None:
        valid_modes = {"relative_outgoing", "relative_incoming", "count", "column"}
        if weight_mode not in valid_modes:
            raise ValueError(
                f"weight_mode must be one of {sorted(valid_modes)}, got {weight_mode!r}"
            )
        if weight_mode == "column":
            if not weight_column:
                raise ValueError(
                    "weight_column must be provided when weight_mode='column'"
                )
        elif weight_column is not None:
            raise ValueError(
                "weight_column can only be provided when weight_mode='column'"
            )

    @classmethod
    def _filter_synapses_to_annotations(
        cls,
        synapses_df: pd.DataFrame,
        neuron_annotations: pd.DataFrame,
        neuron_id_column: str,
        pre_col: str,
        post_col: str,
    ) -> pd.DataFrame:
        annotation_ids = cls._annotation_id_set(neuron_annotations, neuron_id_column)
        valid_pre = synapses_df[pre_col].map(
            lambda value: not cls._is_missing_scalar(value)
        )
        valid_post = synapses_df[post_col].map(
            lambda value: not cls._is_missing_scalar(value)
        )
        pre_ids = synapses_df[pre_col].map(cls._stringify_id_value)
        post_ids = synapses_df[post_col].map(cls._stringify_id_value)
        keep_mask = (
            valid_pre
            & valid_post
            & pre_ids.isin(annotation_ids)
            & post_ids.isin(annotation_ids)
        )
        filtered = synapses_df.loc[keep_mask].copy()
        filtered[pre_col] = pre_ids[keep_mask].to_numpy()
        filtered[post_col] = post_ids[keep_mask].to_numpy()
        removed = len(synapses_df) - len(filtered)

        if removed:
            logging.getLogger(__name__).info(
                "Filtered synapses to neurons present in annotations: kept=%d, removed=%d",
                len(filtered),
                removed,
            )

        return filtered

    @classmethod
    def _filter_adjacency_to_annotations(
        cls,
        adjacency: pd.DataFrame,
        neuron_annotations: pd.DataFrame,
        neuron_id_column: str,
    ) -> pd.DataFrame:
        annotation_ids = cls._annotation_id_set(neuron_annotations, neuron_id_column)
        normalized = cls._normalize_adjacency_axes(adjacency)
        row_mask = normalized.index.isin(annotation_ids)
        col_mask = normalized.columns.isin(annotation_ids)
        filtered = normalized.loc[row_mask, col_mask].copy()

        original_ids = set(normalized.index) | set(normalized.columns)
        filtered_ids = set(filtered.index) | set(filtered.columns)
        removed = original_ids - filtered_ids
        if removed:
            logging.getLogger(__name__).info(
                "Filtered connectivity to neurons present in annotations: kept=%d, removed=%d",
                len(filtered_ids),
                len(removed),
            )

        return filtered

    @classmethod
    def _apply_annotation_scope_to_synapses(
        cls,
        synapses_df: pd.DataFrame,
        neuron_annotations: pd.DataFrame,
        neuron_id_column: str,
        pre_col: str,
        post_col: str,
        annotation_scope: Literal["annotated_only", "all"],
    ) -> pd.DataFrame:
        if annotation_scope == "all":
            return synapses_df
        return cls._filter_synapses_to_annotations(
            synapses_df=synapses_df,
            neuron_annotations=neuron_annotations,
            neuron_id_column=neuron_id_column,
            pre_col=pre_col,
            post_col=post_col,
        )

    @classmethod
    def _apply_annotation_scope_to_adjacency(
        cls,
        adjacency: pd.DataFrame,
        neuron_annotations: pd.DataFrame,
        neuron_id_column: str,
        annotation_scope: Literal["annotated_only", "all"],
    ) -> pd.DataFrame:
        if annotation_scope == "all":
            return adjacency
        return cls._filter_adjacency_to_annotations(
            adjacency=adjacency,
            neuron_annotations=neuron_annotations,
            neuron_id_column=neuron_id_column,
        )

    @staticmethod
    def _selector_to_str_set(
        values: _Selector,
    ) -> set[str] | None:
        if values is None:
            return None
        if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
            return (
                {NestedMatrix._stringify_id_value(values)}
                if not NestedMatrix._is_missing_scalar(values)
                else set()
            )
        return {
            NestedMatrix._stringify_id_value(value)
            for value in values
            if not NestedMatrix._is_missing_scalar(value)
        }

    @staticmethod
    def _select_axis_ids(
        available_ids: set[str],
        annotations: pd.DataFrame,
        id_col: str,
        type_col: str,
        selected_types: _Selector = None,
        selected_neurons: _Selector = None,
    ) -> set[str]:
        selected_type_set = NestedMatrix._selector_to_str_set(selected_types)
        selected_neuron_set = NestedMatrix._selector_to_str_set(selected_neurons)

        if selected_type_set is None and selected_neuron_set is None:
            return set(available_ids)

        axis_ids: set[str] = set()
        if selected_type_set is not None:
            resolved = NestedMatrix._resolve_relevant_annotations(
                matrix_ids=available_ids,
                annotations=annotations,
                id_col=id_col,
                type_col=type_col,
            )
            resolved_types = resolved.typed[type_col].astype(str)
            type_mask = resolved_types.isin(selected_type_set)
            axis_ids.update(
                resolved.typed.loc[type_mask, id_col].map(
                    NestedMatrix._stringify_id_value
                )
            )

        if selected_neuron_set is not None:
            axis_ids.update(selected_neuron_set & available_ids)

        return axis_ids

    @staticmethod
    def _build_axis_ordering(
        axis_ids: set[str],
        annotations: pd.DataFrame,
        id_col: str,
        type_col: str,
        type_order: _TypeOrder,
    ) -> _AxisOrdering:
        resolved = NestedMatrix._resolve_relevant_annotations(
            matrix_ids=axis_ids,
            annotations=annotations,
            id_col=id_col,
            type_col=type_col,
        )
        sorted_types = NestedMatrix._resolve_type_order(
            typed_annotations=resolved.typed,
            type_col=type_col,
            preferred_type_order=type_order,
        )
        ordered_neurons, boundaries = NestedMatrix._build_ordered_neurons(
            typed_annotations=resolved.typed,
            type_col=type_col,
            sorted_types=sorted_types,
            neuron_id_column=id_col,
        )
        if resolved.untyped_ids:
            logging.getLogger(__name__).warning(
                "%d neuron(s) present in the matrix axis have annotation rows "
                "but missing %s; they will appear in the matrix but are excluded "
                "from type-level analysis: %s",
                len(resolved.untyped_ids),
                type_col,
                (
                    resolved.untyped_ids[:10]
                    if len(resolved.untyped_ids) > 10
                    else resolved.untyped_ids
                ),
            )

        if resolved.missing_ids:
            logging.getLogger(__name__).warning(
                "%d neuron(s) present in the matrix axis but missing from "
                "annotations; they will appear in the matrix but are excluded "
                "from type-level analysis: %s",
                len(resolved.missing_ids),
                (
                    resolved.missing_ids[:10]
                    if len(resolved.missing_ids) > 10
                    else resolved.missing_ids
                ),
            )

        ordered_neurons.extend(resolved.untyped_ids)
        ordered_neurons.extend(resolved.missing_ids)
        return _AxisOrdering(
            ordered_neurons=tuple(ordered_neurons),
            type_boundaries=boundaries,
            neuron_to_type=resolved.id_map,
        )

    @property
    def sum_type_matrix(self) -> pd.DataFrame:
        """Calculate the cell type-level connectivity matrix (sum).

        Aggregates the neuron-level matrix into a type-level matrix by summing
        all connections between neurons of each type pair. The returned
        DataFrame is a read-only view; call ``.copy()`` before editing.

        Returns
        -------
        pd.DataFrame
            A square matrix with cell types as both index and columns,
            where each value represents the total connectivity weight
            between those two cell types.

        Examples
        --------
        >>> matrix = NestedMatrix.from_connectivity(...)
        >>> type_connectivity = matrix.sum_type_matrix
        >>> print(type_connectivity.loc['KC', 'MB'])  # KC -> MB connections
        """
        return _readonly_dataframe(self._sum_type_matrix)

    @cached_property
    def _sum_type_matrix(self) -> pd.DataFrame:
        return self._aggregate_type_matrix("sum")

    @property
    def mean_type_matrix(self) -> pd.DataFrame:
        """Calculate the mean cell type-level connectivity matrix.

        Aggregates the neuron-level matrix into a type-level matrix by taking
        the arithmetic mean of all neuron-to-neuron weights within each type
        pair block. This includes zero-valued entries in the block, so larger
        cell types do not automatically dominate the visualization by virtue of
        having more neurons. The returned DataFrame is a read-only view; call
        ``.copy()`` before editing.

        Returns
        -------
        pd.DataFrame
            A square matrix with cell types as both index and columns,
            where each value represents the mean connectivity weight across
            all neuron pairs in the corresponding type block.
        """
        return _readonly_dataframe(self._mean_type_matrix)

    @cached_property
    def _mean_type_matrix(self) -> pd.DataFrame:
        return self._aggregate_type_matrix("mean")

    def _aggregate_type_matrix(self, aggregate: str) -> pd.DataFrame:
        if not self._type_boundaries:
            return pd.DataFrame()

        type_names = list(self._type_boundaries.keys())
        bounds = list(self._type_boundaries.values())
        data = self._matrix.to_numpy()
        n = len(type_names)
        result = np.empty((n, n), dtype=float)

        for i, (r_start, r_end) in enumerate(bounds):
            for j, (c_start, c_end) in enumerate(bounds):
                block = data[r_start:r_end, c_start:c_end]
                if aggregate == "sum":
                    value = block.sum()
                elif aggregate == "mean":
                    value = block.mean()
                else:
                    raise ValueError(f"Unsupported type aggregation: {aggregate!r}")
                result[i, j] = float(value)

        return pd.DataFrame(result, index=type_names, columns=type_names)

    def get_relative_weights(self, by_type: bool = False) -> pd.DataFrame:
        """Calculate relative connection weights normalized by row sums.

        Computes the proportion of each neuron's (or type's) total output
        that goes to each target. Values range from 0 to 1, where 1 indicates
        all output goes to that target.

        Parameters
        ----------
        by_type : bool, default False
            If True, calculate relative weights at the type level.
            If False, calculate at the neuron level.

        Returns
        -------
        pd.DataFrame
            Matrix of relative weights where each row sums to 1.0
            (except for rows with zero total output).

        Examples
        --------
        >>> matrix = NestedMatrix.from_connectivity(...)
        >>> relative = matrix.get_relative_weights(by_type=True)
        >>> # Shows what proportion of each type's output goes to each target type
        """
        df = self._sum_type_matrix if by_type else self._matrix
        row_sums = df.sum(axis=1).replace(0, 1)
        return df.div(row_sums, axis=0)

    def plot(
        self,
        output_path: str | None = None,
        level: Literal["neuron", "type_mean", "type_sum"] = "neuron",
        figsize: tuple[int, int] = (16, 14),
        show_neuron_labels: bool = False,
        vmin_percentile: float = 0.0,
        vmax_percentile: float = 100.0,
        min_neurons_for_plot: int = 1,
        linewidth_scale: float = 1.0,
    ) -> tuple[Figure, Axes]:
        """Plot the connectivity matrix as a heatmap.

        Creates a visualization of the connectivity matrix with optional
        type boundaries, customizable color scaling, and filtering options.

        Parameters
        ----------
        output_path : str | None, optional
            If provided, save the plot to this file path. Directory will be
            created if it doesn't exist.
        level : {"neuron", "type_mean", "type_sum"}, default "neuron"
            Which matrix to visualize. ``"neuron"`` plots the neuron-level
            matrix with nested type boundaries. ``"type_mean"`` plots
            ``mean_type_matrix``. ``"type_sum"`` plots ``sum_type_matrix``.
        figsize : tuple[int, int], default (16, 14)
            Figure size in inches (width, height).
        show_neuron_labels: bool, default False
            If True, show individual neuron IDs on axes. Only applies when
            ``level="neuron"``. If False, shows type labels at boundaries.
        vmin_percentile : float, default 0.0
            Percentile for minimum color scale value (0-100). Computed
            over nonzero values only, so zero-weight entries always map
            to the bottom of the colormap.
        vmax_percentile : float, default 100.0
            Percentile for maximum color scale value (0-100). Computed
            over nonzero values only. Values below 100 clip the strongest
            connections to the max color, making mid-range weights more
            visible.
        min_neurons_for_plot : int, default 1
            Minimum number of neurons required per type to include in plot.
            Types with fewer neurons will be filtered out.
        linewidth_scale : float, default 1.0
            Multiplier applied to the default boundary line widths. Use values
            above 1.0 for thicker boundaries and below 1.0 for thinner ones.

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            Matplotlib figure and axes objects for further customization.

        Examples
        --------
        >>> matrix = NestedMatrix.from_connectivity(...)
        >>> fig, ax = matrix.plot(level="type_mean", output_path='connectivity.png')
        >>> plt.show()
        """
        valid_levels = {"neuron", "type_mean", "type_sum"}
        if level not in valid_levels:
            raise ValueError(
                f"level must be one of {sorted(valid_levels)}, got {level!r}"
            )
        if linewidth_scale <= 0:
            raise ValueError(
                f"linewidth_scale must be positive, got {linewidth_scale!r}"
            )

        if level == "type_mean":
            data = self._mean_type_matrix
            boundaries = {name: (i, i + 1) for i, name in enumerate(data.index)}
            labels = list(data.index)
            label_boundaries = None
            colorbar_label = "Mean Weight"
        elif level == "type_sum":
            data = self._sum_type_matrix
            boundaries = {name: (i, i + 1) for i, name in enumerate(data.index)}
            labels = list(data.index)
            label_boundaries = None
            colorbar_label = "Total Weight"
        else:
            plot_data = self._filter_for_plot(min_neurons_for_plot)
            data = plot_data.matrix
            boundaries = plot_data.boundaries
            labels = plot_data.labels
            label_boundaries = (
                boundaries
                if not show_neuron_labels
                and len(boundaries) <= _MAX_CENTERED_TYPE_LABELS
                else None
            )
            colorbar_label = "Weight"

        fig, ax = plt.subplots(figsize=figsize)

        self._draw_heatmap(
            ax, data, vmin_percentile, vmax_percentile, colorbar_label=colorbar_label
        )

        self._draw_boundaries(
            ax,
            boundaries,
            shape=data.shape,
            linewidth_scale=linewidth_scale,
        )

        self._configure_axes(
            ax,
            labels=labels if show_neuron_labels or level != "neuron" else None,
            boundaries=label_boundaries,
        )

        plt.tight_layout()
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")

        return fig, ax

    def _draw_heatmap(
        self,
        ax: Axes,
        data: pd.DataFrame,
        vmin_p: float,
        vmax_p: float,
        colorbar_label: str = "Weight",
    ) -> None:
        vals = data.values.flatten()
        vals = vals[vals > 0]

        if len(vals) > 0:
            vmin = float(np.percentile(vals, vmin_p))
            vmax = float(np.percentile(vals, vmax_p))
            if vmax == vmin:
                vmax += 1.0
        else:
            vmin, vmax = 0.0, 1.0

        im = ax.imshow(
            data.values,
            cmap=_PURPLE_CMAP,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            interpolation="nearest",
            aspect="auto",
        )

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(colorbar_label, fontsize=16, fontweight="bold")
        cbar.ax.tick_params(labelsize=14)

    def _draw_boundaries(
        self,
        ax: Axes,
        boundaries: dict[str, tuple[int, int]],
        shape: tuple[int, int],
        linewidth_scale: float = 1.0,
    ) -> None:
        nrows, ncols = shape
        if nrows == 0 or ncols == 0:
            return

        linewidths = {
            name: width * linewidth_scale
            for name, width in _DEFAULT_BOUNDARY_LINEWIDTHS.items()
        }

        outer_segments = [
            [(-0.5, -0.5), (ncols - 0.5, -0.5)],
            [(ncols - 0.5, -0.5), (ncols - 0.5, nrows - 0.5)],
            [(ncols - 0.5, nrows - 0.5), (-0.5, nrows - 0.5)],
            [(-0.5, nrows - 0.5), (-0.5, -0.5)],
        ]
        ax.add_collection(
            LineCollection(
                outer_segments,
                colors="black",
                linewidths=linewidths["outer"],
            )
        )

        grid_positions = []
        block_segments = []

        for _, (start, end) in boundaries.items():
            lower = start - 0.5
            upper = end - 0.5
            block_segments.extend(
                [
                    [(lower, lower), (upper, lower)],
                    [(upper, lower), (upper, upper)],
                    [(upper, upper), (lower, upper)],
                    [(lower, upper), (lower, lower)],
                ]
            )

            if end < max(nrows, ncols):
                grid_positions.append(end - 0.5)

        if block_segments:
            ax.add_collection(
                LineCollection(
                    block_segments,
                    colors="white",
                    linewidths=linewidths["block_light"],
                )
            )
            ax.add_collection(
                LineCollection(
                    block_segments,
                    colors="black",
                    linewidths=linewidths["block_dark"],
                )
            )

        if grid_positions:
            grid_segments = []
            for pos in grid_positions:
                grid_segments.extend(
                    [
                        [(-0.5, pos), (ncols - 0.5, pos)],
                        [(pos, -0.5), (pos, nrows - 0.5)],
                    ]
                )

            ax.add_collection(
                LineCollection(
                    grid_segments,
                    colors="white",
                    linewidths=linewidths["grid_light"],
                    alpha=1.0,
                )
            )
            ax.add_collection(
                LineCollection(
                    grid_segments,
                    colors="black",
                    linewidths=linewidths["grid_dark"],
                    alpha=0.9,
                )
            )

    def _configure_axes(
        self,
        ax: Axes,
        labels: list[str] | None,
        boundaries: dict[str, tuple[int, int]] | None,
    ) -> None:
        if boundaries:
            ticks = []
            tick_labels = []
            for name, (start, end) in boundaries.items():
                center = (start + end - 1) / 2
                ticks.append(center)
                tick_labels.append(name)

            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=14)
            ax.set_yticklabels(tick_labels, fontsize=14)

        elif labels:
            indices = np.arange(len(labels))
            ax.set_xticks(indices)
            ax.set_yticks(indices)
            ax.set_xticklabels(labels, rotation=90, ha="right", fontsize=8)
            ax.set_yticklabels(labels, fontsize=8)

        else:
            ax.set_xticks([])
            ax.set_yticks([])

        ax.set_xlabel("Postsynaptic", fontsize=18, fontweight="bold")
        ax.set_ylabel("Presynaptic", fontsize=18, fontweight="bold")

    def _filter_for_plot(self, min_neurons: int) -> _PlotData:
        if min_neurons <= 1:
            return _PlotData(
                self._matrix, self._type_boundaries, list(self._ordered_neurons)
            )

        kept_indices = []
        new_boundaries = {}
        current_idx = 0

        for name, (start, end) in self._type_boundaries.items():
            count = end - start
            if count >= min_neurons:
                indices = list(range(start, end))
                kept_indices.extend(indices)
                new_boundaries[name] = (current_idx, current_idx + count)
                current_idx += count

        filtered_matrix = self._matrix.iloc[kept_indices, kept_indices]
        filtered_labels = [self._ordered_neurons[i] for i in kept_indices]

        return _PlotData(filtered_matrix, new_boundaries, filtered_labels)

    @staticmethod
    def _coerce_to_adjacency(connectivity: pd.DataFrame) -> pd.DataFrame:
        cols = set(connectivity.columns)

        edge_list_formats = [
            ({"type.from", "type.to"}, "type.from", "type.to", "weight"),
            ({"pre", "post"}, "pre", "post", "weight"),
            ({"source", "target"}, "source", "target", "n_syn"),
        ]

        matches = [fmt for fmt in edge_list_formats if fmt[0].issubset(cols)]
        if len(matches) > 1:
            matched_names = [sorted(fmt[0]) for fmt in matches]
            raise ValueError(
                f"Ambiguous edge-list format: columns match multiple formats "
                f"{matched_names}. Rename or drop columns so only one format matches."
            )

        if matches:
            _, idx_col, col_col, alt_weight = matches[0]
            df = NestedMatrix._drop_null_edge_ids(connectivity, idx_col, col_col)
            weight = "weight" if "weight" in cols else alt_weight
            if weight not in cols:
                df[weight] = 1
            df = NestedMatrix._coerce_weight_values(df, weight)
            df[idx_col] = df[idx_col].map(NestedMatrix._stringify_id_value)
            df[col_col] = df[col_col].map(NestedMatrix._stringify_id_value)
            adjacency = df.pivot_table(
                index=idx_col,
                columns=col_col,
                values=weight,
                aggfunc="sum",
                fill_value=0,
            )
            return NestedMatrix._normalize_adjacency_axes(adjacency)

        return NestedMatrix._normalize_adjacency_axes(
            connectivity.fillna(0).astype(float)
        )

    @staticmethod
    def _sort_cell_types(
        types: list[Any], preferred: _TypeOrder = None
    ) -> list[str]:
        unique = {
            NestedMatrix._stringify_id_value(t)
            for t in types
            if not NestedMatrix._is_missing_scalar(t)
        }

        def parse_label(label):
            m = re.match(r"^([A-Za-z]+)(\d*)(.*)$", label.strip())
            if not m:
                return (label.upper(), float("inf"), "")
            pfx, num, sfx = m.groups()
            return (pfx.upper(), int(num) if num else float("inf"), sfx.upper())

        parsed = {label: parse_label(label) for label in unique}
        generic_order = sorted(unique, key=lambda x: (*parsed[x][:2], x))

        if not preferred:
            return generic_order

        result = []
        remaining = set(generic_order)

        for preferred_value in preferred:
            if NestedMatrix._is_missing_scalar(preferred_value):
                continue
            p = NestedMatrix._stringify_id_value(preferred_value)
            matches = [t for t in generic_order if t == p or parsed[t][0] == p]
            for m in matches:
                if m in remaining:
                    result.append(m)
                    remaining.remove(m)

        result.extend([t for t in generic_order if t in remaining])
        return result

    @staticmethod
    def _resolve_relevant_annotations(
        matrix_ids: set[str],
        annotations: pd.DataFrame,
        id_col: str,
        type_col: str,
    ) -> _ResolvedAnnotations:
        valid_ids = annotations[id_col].map(
            lambda value: not NestedMatrix._is_missing_scalar(value)
        )
        normalized = annotations.loc[valid_ids].copy()
        ann_ids = normalized[id_col].map(NestedMatrix._stringify_id_value)
        relevant = normalized[ann_ids.isin(matrix_ids)].copy()
        relevant = relevant.assign(
            **{
                id_col: ann_ids.loc[relevant.index],
                "__has_type__": relevant[type_col].notna(),
                "__row_order__": np.arange(len(relevant)),
            }
        )
        relevant["__first_seen__"] = relevant.groupby(id_col)[
            "__row_order__"
        ].transform("min")
        relevant = relevant.sort_values(
            by=["__first_seen__", "__has_type__", "__row_order__"],
            ascending=[True, False, True],
        ).drop_duplicates(subset=[id_col], keep="first")

        relevant[type_col] = relevant[type_col].map(
            lambda value: NestedMatrix._stringify_id_value(value)
            if not NestedMatrix._is_missing_scalar(value)
            else value
        )

        typed = relevant[relevant[type_col].notna()].copy()
        id_map = dict(zip(typed[id_col], typed[type_col]))
        typed_ids = set(typed[id_col])
        untyped_ids = sorted(set(relevant[id_col]) - typed_ids)
        missing_ids = sorted(matrix_ids - set(relevant[id_col]))

        return _ResolvedAnnotations(relevant, typed, id_map, untyped_ids, missing_ids)

    @staticmethod
    def _resolve_type_order(
        typed_annotations: pd.DataFrame,
        type_col: str,
        preferred_type_order: _TypeOrder,
    ) -> list[str]:
        present_types = [
            cell_type
            for cell_type in typed_annotations[type_col].unique()
            if pd.notna(cell_type)
        ]
        return NestedMatrix._sort_cell_types(
            present_types, preferred=preferred_type_order
        )

    @staticmethod
    def _default_within_type_order(
        type_rows: pd.DataFrame,
        neuron_id_column: str,
    ) -> list[str]:
        return type_rows[neuron_id_column].tolist()

    @staticmethod
    def _is_missing_label(value: Any) -> bool:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)):
            return bool(missing)
        return False

    @staticmethod
    def _extract_ranked_label(
        row: pd.Series,
        rule: _WithinTypeOrderRule,
        neuron_id_column: str,
    ) -> str | None:
        for column in rule.label_columns:
            if column not in row.index:
                continue

            value = row[column]
            if NestedMatrix._is_missing_label(value):
                continue

            match = rule.pattern.search(str(value).strip())
            if not match:
                continue

            label = match.group(1)
            if label in rule.rank:
                return label

        logging.getLogger(__name__).warning(
            "Could not resolve a ranked column label for neuron %s from columns %s; "
            "it will be ordered after ranked neurons",
            row.get(neuron_id_column, "<unknown>"),
            rule.label_columns,
        )
        return None

    @staticmethod
    def _order_neurons_by_rule(
        type_rows: pd.DataFrame,
        neuron_id_column: str,
        rule: _WithinTypeOrderRule,
    ) -> list[str]:
        sorted_rows = type_rows.copy()
        sorted_rows["__group_order__"] = np.arange(len(sorted_rows))
        sorted_rows["__column_label__"] = sorted_rows.apply(
            NestedMatrix._extract_ranked_label,
            axis=1,
            rule=rule,
            neuron_id_column=neuron_id_column,
        )
        sorted_rows["__column_rank__"] = sorted_rows["__column_label__"].map(
            rule.rank
        )
        sorted_rows["__has_column_rank__"] = sorted_rows["__column_rank__"].notna()
        sorted_rows["__column_rank__"] = sorted_rows["__column_rank__"].fillna(
            np.inf
        )
        sorted_rows = sorted_rows.sort_values(
            by=["__has_column_rank__", "__column_rank__", "__group_order__"],
            ascending=[False, True, True],
        )

        return sorted_rows[neuron_id_column].tolist()

    @staticmethod
    def _order_neurons_within_type(
        type_name: str,
        type_rows: pd.DataFrame,
        neuron_id_column: str,
    ) -> list[str]:
        rule = _WITHIN_TYPE_ORDER_RULES.get(type_name)
        if rule is None:
            return NestedMatrix._default_within_type_order(type_rows, neuron_id_column)

        return NestedMatrix._order_neurons_by_rule(type_rows, neuron_id_column, rule)

    @staticmethod
    def _build_ordered_neurons(
        typed_annotations: pd.DataFrame,
        type_col: str,
        sorted_types: list[str],
        neuron_id_column: str,
    ) -> tuple[list[str], dict[str, tuple[int, int]]]:
        ordered_neurons = []
        boundaries = {}
        current_pos = 0

        for c_type in sorted_types:
            type_rows = typed_annotations[typed_annotations[type_col] == c_type]
            final_group = NestedMatrix._order_neurons_within_type(
                type_name=str(c_type),
                type_rows=type_rows,
                neuron_id_column=neuron_id_column,
            )
            if not final_group:
                continue

            ordered_neurons.extend(final_group)
            boundaries[str(c_type)] = (current_pos, current_pos + len(final_group))
            current_pos += len(final_group)

        return ordered_neurons, boundaries

    @staticmethod
    def _align_neurons_and_boundaries(
        adjacency: pd.DataFrame,
        annotations: pd.DataFrame,
        id_col: str,
        type_col: str,
        type_order: _TypeOrder,
    ) -> tuple[list[str], dict[str, tuple[int, int]], dict[str, Any]]:
        matrix_ids = set(NestedMatrix._normalize_id_values(adjacency.index)) | set(
            NestedMatrix._normalize_id_values(adjacency.columns)
        )
        axis = NestedMatrix._build_axis_ordering(
            axis_ids=matrix_ids,
            annotations=annotations,
            id_col=id_col,
            type_col=type_col,
            type_order=type_order,
        )
        return (
            list(axis.ordered_neurons),
            dict(axis.type_boundaries),
            dict(axis.neuron_to_type),
        )


class DirectedNestedMatrix:
    """A rectangular nested connectivity matrix with independent axes.

    Rows are presynaptic/source neurons and columns are postsynaptic/target
    neurons. Unlike ``NestedMatrix``, the matrix may be rectangular and the
    source and target cell type sets may differ. Type and explicit neuron
    selectors on the same axis are unioned. If an axis selector is ``None``,
    that axis includes all available neurons for that axis independently of the
    other axis selector.
    """

    def __init__(
        self,
        matrix: pd.DataFrame,
        source_neurons: Iterable[Any],
        target_neurons: Iterable[Any],
        source_type_boundaries: Mapping[Any, tuple[int, int]] | None = None,
        target_type_boundaries: Mapping[Any, tuple[int, int]] | None = None,
        source_neuron_to_type: Mapping[Any, Any] | None = None,
        target_neuron_to_type: Mapping[Any, Any] | None = None,
    ):
        matrix = matrix.copy()
        matrix.index = NestedMatrix._stringify_id_axis(matrix.index, "matrix index")
        matrix.columns = NestedMatrix._stringify_id_axis(
            matrix.columns, "matrix columns"
        )
        source_axis = self._normalize_axis(
            _AxisOrdering(
                ordered_neurons=tuple(source_neurons),
                type_boundaries=dict(source_type_boundaries or {}),
                neuron_to_type=dict(source_neuron_to_type or {}),
            )
        )
        target_axis = self._normalize_axis(
            _AxisOrdering(
                ordered_neurons=tuple(target_neurons),
                type_boundaries=dict(target_type_boundaries or {}),
                neuron_to_type=dict(target_neuron_to_type or {}),
            )
        )

        self._validate_invariants(
            matrix=matrix,
            source_axis=source_axis,
            target_axis=target_axis,
        )

        _set_dataframe_writeable(matrix, writeable=False)
        self._matrix = matrix
        self._source_neurons = source_axis.ordered_neurons
        self._target_neurons = target_axis.ordered_neurons
        self._source_type_boundaries = source_axis.type_boundaries
        self._target_type_boundaries = target_axis.type_boundaries
        self._source_neuron_to_type = source_axis.neuron_to_type
        self._target_neuron_to_type = target_axis.neuron_to_type

    @classmethod
    def _from_axes(
        cls,
        matrix: pd.DataFrame,
        source_axis: _AxisOrdering,
        target_axis: _AxisOrdering,
    ) -> "DirectedNestedMatrix":
        return cls(
            matrix=matrix,
            source_neurons=source_axis.ordered_neurons,
            target_neurons=target_axis.ordered_neurons,
            source_type_boundaries=source_axis.type_boundaries,
            target_type_boundaries=target_axis.type_boundaries,
            source_neuron_to_type=source_axis.neuron_to_type,
            target_neuron_to_type=target_axis.neuron_to_type,
        )

    @staticmethod
    def _normalize_axis(axis: _AxisOrdering) -> _AxisOrdering:
        return _AxisOrdering(
            ordered_neurons=tuple(
                NestedMatrix._stringify_id_axis(
                    axis.ordered_neurons, "axis ordered_neurons"
                )
            ),
            type_boundaries={
                NestedMatrix._stringify_id_value(name): (int(start), int(end))
                for name, (start, end) in axis.type_boundaries.items()
            },
            neuron_to_type={
                NestedMatrix._stringify_id_value(neuron): (
                    cell_type
                    if NestedMatrix._is_missing_scalar(cell_type)
                    else NestedMatrix._stringify_id_value(cell_type)
                )
                for neuron, cell_type in axis.neuron_to_type.items()
                if not NestedMatrix._is_missing_scalar(neuron)
            },
        )

    @staticmethod
    def _validate_axis_metadata(
        axis_name: str,
        ordered_neurons: tuple[str, ...],
        type_boundaries: dict[str, tuple[int, int]],
        neuron_to_type: dict[str, Any],
    ) -> None:
        duplicate_neurons = NestedMatrix._find_duplicates(ordered_neurons)
        if duplicate_neurons:
            raise ValueError(
                f"{axis_name}_neurons contains duplicate neuron IDs after "
                f"normalization: {duplicate_neurons[:10]}"
            )

        extra_neurons = sorted(set(neuron_to_type) - set(ordered_neurons))
        if extra_neurons:
            raise ValueError(
                f"{axis_name}_neuron_to_type contains neurons not present in "
                f"{axis_name}_neurons: {extra_neurons[:10]}"
            )

        expected_start = 0
        for name, (start, end) in type_boundaries.items():
            if start != expected_start:
                raise ValueError(
                    f"{axis_name}_type_boundaries must be contiguous, "
                    "non-overlapping, and start at 0"
                )
            if start < 0 or end > len(ordered_neurons) or end <= start:
                raise ValueError(
                    f"{axis_name} type boundary {name!r} has invalid slice "
                    f"({start}, {end})"
                )

            boundary_neurons = ordered_neurons[start:end]
            mismatched = [
                neuron
                for neuron in boundary_neurons
                if str(neuron_to_type.get(neuron)) != name
            ]
            if mismatched:
                raise ValueError(
                    f"{axis_name} type boundary {name!r} does not match "
                    f"neuron_to_type for neurons {mismatched[:10]}"
                )

            expected_start = end

        annotated_after_boundaries = [
            neuron
            for neuron in ordered_neurons[expected_start:]
            if pd.notna(neuron_to_type.get(neuron))
        ]
        if annotated_after_boundaries:
            raise ValueError(
                f"annotated {axis_name} neurons must appear within "
                "type_boundaries before any unassigned neurons"
            )

    @classmethod
    def _validate_invariants(
        cls,
        matrix: pd.DataFrame,
        source_axis: _AxisOrdering,
        target_axis: _AxisOrdering,
    ) -> None:
        if list(matrix.index) != list(source_axis.ordered_neurons):
            raise ValueError("matrix index must match source_neurons exactly")
        if list(matrix.columns) != list(target_axis.ordered_neurons):
            raise ValueError("matrix columns must match target_neurons exactly")

        cls._validate_axis_metadata(
            axis_name="source",
            ordered_neurons=source_axis.ordered_neurons,
            type_boundaries=source_axis.type_boundaries,
            neuron_to_type=source_axis.neuron_to_type,
        )
        cls._validate_axis_metadata(
            axis_name="target",
            ordered_neurons=target_axis.ordered_neurons,
            type_boundaries=target_axis.type_boundaries,
            neuron_to_type=target_axis.neuron_to_type,
        )

    def __repr__(self) -> str:
        return (
            "DirectedNestedMatrix("
            f"{len(self._source_neurons)} source neurons, "
            f"{len(self._target_neurons)} target neurons, "
            f"source_types={list(self._source_type_boundaries.keys())}, "
            f"target_types={list(self._target_type_boundaries.keys())})"
        )

    @property
    def matrix(self) -> pd.DataFrame:
        """Read-only source-to-target neuron connectivity matrix."""
        return _readonly_dataframe(self._matrix)

    @property
    def source_neurons(self) -> tuple[str, ...]:
        """Read-only ordered source neuron IDs."""
        return self._source_neurons

    @property
    def target_neurons(self) -> tuple[str, ...]:
        """Read-only ordered target neuron IDs."""
        return self._target_neurons

    @property
    def source_type_boundaries(self) -> Mapping[str, tuple[int, int]]:
        """Read-only mapping from source cell type to its row slice."""
        return MappingProxyType(self._source_type_boundaries)

    @property
    def target_type_boundaries(self) -> Mapping[str, tuple[int, int]]:
        """Read-only mapping from target cell type to its column slice."""
        return MappingProxyType(self._target_type_boundaries)

    @property
    def source_neuron_to_type(self) -> Mapping[str, Any]:
        """Read-only mapping from source neuron ID to cell type."""
        return MappingProxyType(self._source_neuron_to_type)

    @property
    def target_neuron_to_type(self) -> Mapping[str, Any]:
        """Read-only mapping from target neuron ID to cell type."""
        return MappingProxyType(self._target_neuron_to_type)

    @staticmethod
    def _validate_normalization_scope(normalization_scope: str) -> None:
        valid_scopes = {"selected", "all"}
        if normalization_scope not in valid_scopes:
            raise ValueError(
                "normalization_scope must be one of "
                f"{sorted(valid_scopes)}, got {normalization_scope!r}"
            )

    @staticmethod
    def _stringify_adjacency_axes(adjacency: pd.DataFrame) -> pd.DataFrame:
        return NestedMatrix._normalize_adjacency_axes(adjacency)

    @staticmethod
    def _build_axis_from_available(
        available_ids: set[str],
        neuron_annotations: pd.DataFrame,
        neuron_id_column: str,
        cell_type_column: str,
        type_order: _TypeOrder,
        selected_types: _Selector,
        selected_neurons: _Selector,
    ) -> _AxisOrdering:
        axis_ids = NestedMatrix._select_axis_ids(
            available_ids=available_ids,
            annotations=neuron_annotations,
            id_col=neuron_id_column,
            type_col=cell_type_column,
            selected_types=selected_types,
            selected_neurons=selected_neurons,
        )
        return NestedMatrix._build_axis_ordering(
            axis_ids=axis_ids,
            annotations=neuron_annotations,
            id_col=neuron_id_column,
            type_col=cell_type_column,
            type_order=type_order,
        )

    @classmethod
    def from_connectivity(
        cls,
        connections_df: pd.DataFrame,
        neuron_annotations: pd.DataFrame,
        source_types: _Selector = None,
        target_types: _Selector = None,
        source_neurons: _Selector = None,
        target_neurons: _Selector = None,
        cell_type_column: str = "cell_type",
        neuron_id_column: str = "root_id",
        type_order: _TypeOrder = None,
        source_type_order: _TypeOrder = None,
        target_type_order: _TypeOrder = None,
        annotation_scope: Literal["annotated_only", "all"] = "annotated_only",
    ) -> "DirectedNestedMatrix":
        """Create a rectangular directed nested matrix from connectivity data.

        ``source_*`` selectors apply to rows and ``target_*`` selectors apply
        to columns. Type and explicit neuron selectors on the same axis are
        unioned. ``None`` selects all available neurons on that axis.
        """
        NestedMatrix._validate_annotation_scope(annotation_scope)
        adjacency = NestedMatrix._coerce_to_adjacency(connections_df)
        adjacency = NestedMatrix._apply_annotation_scope_to_adjacency(
            adjacency=adjacency,
            neuron_annotations=neuron_annotations,
            neuron_id_column=neuron_id_column,
            annotation_scope=annotation_scope,
        )
        adjacency = cls._stringify_adjacency_axes(adjacency)

        source_axis = cls._build_axis_from_available(
            available_ids=set(adjacency.index),
            neuron_annotations=neuron_annotations,
            neuron_id_column=neuron_id_column,
            cell_type_column=cell_type_column,
            type_order=source_type_order if source_type_order is not None else type_order,
            selected_types=source_types,
            selected_neurons=source_neurons,
        )
        target_axis = cls._build_axis_from_available(
            available_ids=set(adjacency.columns),
            neuron_annotations=neuron_annotations,
            neuron_id_column=neuron_id_column,
            cell_type_column=cell_type_column,
            type_order=target_type_order if target_type_order is not None else type_order,
            selected_types=target_types,
            selected_neurons=target_neurons,
        )

        matrix = adjacency.reindex(
            index=source_axis.ordered_neurons,
            columns=target_axis.ordered_neurons,
            fill_value=0,
        ).astype(float)
        return cls._from_axes(matrix=matrix, source_axis=source_axis, target_axis=target_axis)

    @classmethod
    def from_synapses(
        cls,
        synapses_df: pd.DataFrame,
        neuron_annotations: pd.DataFrame,
        source_types: _Selector = None,
        target_types: _Selector = None,
        source_neurons: _Selector = None,
        target_neurons: _Selector = None,
        pre_col: str = "pre_pt_root_id",
        post_col: str = "post_pt_root_id",
        weight_mode: Literal[
            "relative_outgoing", "relative_incoming", "count", "column"
        ] = "relative_outgoing",
        weight_column: str | None = None,
        normalization_scope: Literal["selected", "all"] = "selected",
        cell_type_column: str = "cell_type",
        neuron_id_column: str = "root_id",
        type_order: _TypeOrder = None,
        source_type_order: _TypeOrder = None,
        target_type_order: _TypeOrder = None,
        annotation_scope: Literal["annotated_only", "all"] = "annotated_only",
    ) -> "DirectedNestedMatrix":
        """Create a rectangular directed nested matrix from synapse rows.

        The default ``normalization_scope="selected"`` normalizes relative
        weights after source/target filtering. Use ``"all"`` to normalize by
        all partners for the selected source or target neurons before the
        matrix is restricted to the selected axes.
        """
        NestedMatrix._validate_weighting(weight_mode, weight_column)
        NestedMatrix._validate_annotation_scope(annotation_scope)
        cls._validate_normalization_scope(normalization_scope)

        scoped_synapses = NestedMatrix._apply_annotation_scope_to_synapses(
            synapses_df=synapses_df,
            neuron_annotations=neuron_annotations,
            neuron_id_column=neuron_id_column,
            pre_col=pre_col,
            post_col=post_col,
            annotation_scope=annotation_scope,
        ).copy()
        scoped_synapses = NestedMatrix._normalize_synapse_edge_ids(
            scoped_synapses, pre_col, post_col
        )

        source_axis = cls._build_axis_from_available(
            available_ids=set(
                NestedMatrix._normalize_id_values(scoped_synapses[pre_col])
            ),
            neuron_annotations=neuron_annotations,
            neuron_id_column=neuron_id_column,
            cell_type_column=cell_type_column,
            type_order=source_type_order if source_type_order is not None else type_order,
            selected_types=source_types,
            selected_neurons=source_neurons,
        )
        target_axis = cls._build_axis_from_available(
            available_ids=set(
                NestedMatrix._normalize_id_values(scoped_synapses[post_col])
            ),
            neuron_annotations=neuron_annotations,
            neuron_id_column=neuron_id_column,
            cell_type_column=cell_type_column,
            type_order=target_type_order if target_type_order is not None else type_order,
            selected_types=target_types,
            selected_neurons=target_neurons,
        )

        edges = cls._aggregate_directed_synapse_edges(
            synapses_df=scoped_synapses,
            pre_col=pre_col,
            post_col=post_col,
            source_ids=set(source_axis.ordered_neurons),
            target_ids=set(target_axis.ordered_neurons),
            weight_mode=weight_mode,
            weight_column=weight_column,
            normalization_scope=normalization_scope,
        )
        adjacency = NestedMatrix._coerce_to_adjacency(edges)
        adjacency = cls._stringify_adjacency_axes(adjacency)
        matrix = adjacency.reindex(
            index=source_axis.ordered_neurons,
            columns=target_axis.ordered_neurons,
            fill_value=0,
        ).astype(float)

        return cls._from_axes(matrix=matrix, source_axis=source_axis, target_axis=target_axis)

    @classmethod
    def _aggregate_directed_synapse_edges(
        cls,
        synapses_df: pd.DataFrame,
        pre_col: str,
        post_col: str,
        source_ids: set[str],
        target_ids: set[str],
        weight_mode: Literal[
            "relative_outgoing", "relative_incoming", "count", "column"
        ],
        weight_column: str | None,
        normalization_scope: Literal["selected", "all"],
    ) -> pd.DataFrame:
        valid_pre = synapses_df[pre_col].map(
            lambda value: not NestedMatrix._is_missing_scalar(value)
        )
        valid_post = synapses_df[post_col].map(
            lambda value: not NestedMatrix._is_missing_scalar(value)
        )
        pre_ids = synapses_df[pre_col].map(NestedMatrix._stringify_id_value)
        post_ids = synapses_df[post_col].map(NestedMatrix._stringify_id_value)

        if normalization_scope == "all" and weight_mode == "relative_outgoing":
            keep_mask = valid_pre & valid_post & pre_ids.isin(source_ids)
        elif normalization_scope == "all" and weight_mode == "relative_incoming":
            keep_mask = valid_pre & valid_post & post_ids.isin(target_ids)
        else:
            keep_mask = (
                valid_pre
                & valid_post
                & pre_ids.isin(source_ids)
                & post_ids.isin(target_ids)
            )

        filtered = synapses_df.loc[keep_mask].copy()
        filtered[pre_col] = pre_ids[keep_mask].to_numpy()
        filtered[post_col] = post_ids[keep_mask].to_numpy()

        return NestedMatrix._aggregate_synapse_edges(
            synapses_df=filtered,
            pre_col=pre_col,
            post_col=post_col,
            weight_mode=weight_mode,
            weight_column=weight_column,
        )

    @classmethod
    def from_synapses_by_neuropil(
        cls,
        synapses_df: pd.DataFrame,
        neuron_annotations: pd.DataFrame,
        neuropil_names: list[str] | None = None,
        coordinates: str = "nm",
        position_column: str = "ctr_pt_position",
        source_types: _Selector = None,
        target_types: _Selector = None,
        source_neurons: _Selector = None,
        target_neurons: _Selector = None,
        pre_col: str = "pre_pt_root_id",
        post_col: str = "post_pt_root_id",
        weight_mode: Literal[
            "relative_outgoing", "relative_incoming", "count", "column"
        ] = "relative_outgoing",
        weight_column: str | None = None,
        normalization_scope: Literal["selected", "all"] = "selected",
        cell_type_column: str = "cell_type",
        neuron_id_column: str = "root_id",
        type_order: _TypeOrder = None,
        source_type_order: _TypeOrder = None,
        target_type_order: _TypeOrder = None,
        annotation_scope: Literal["annotated_only", "all"] = "annotated_only",
        include_other: bool = True,
        voxel_offset: tuple[float, float, float] | None = None,
    ) -> "NeuropilCollection":
        """Create directed nested matrices per neuropil using mesh containment."""
        from crantpy.utils.config import (
            NEUROPIL_MESH_DICT,
            SCALE_X,
            SCALE_Y,
            SCALE_Z,
        )
        from crantpy.viz.mesh import (
            get_supported_neuropil_mesh_labels,
            load_neuropil_mesh,
        )

        NestedMatrix._validate_weighting(weight_mode, weight_column)
        NestedMatrix._validate_annotation_scope(annotation_scope)
        cls._validate_normalization_scope(normalization_scope)

        if neuropil_names is None:
            neuropil_names = list(NEUROPIL_MESH_DICT.values())
        else:
            supported = get_supported_neuropil_mesh_labels()
            invalid = [n for n in neuropil_names if n not in supported]
            if invalid:
                raise ValueError(
                    f"Invalid neuropil name(s): {invalid}. " f"Available: {supported}"
                )

        if synapses_df.empty:
            return NeuropilCollection()

        scoped_synapses = NestedMatrix._apply_annotation_scope_to_synapses(
            synapses_df=synapses_df,
            neuron_annotations=neuron_annotations,
            neuron_id_column=neuron_id_column,
            pre_col=pre_col,
            post_col=post_col,
            annotation_scope=annotation_scope,
        )
        if scoped_synapses.empty and annotation_scope == "annotated_only":
            return NeuropilCollection()

        positions = np.vstack(scoped_synapses[position_column].values)
        if coordinates == "pixels":
            if voxel_offset is not None:
                positions += np.array(voxel_offset)
            positions[:, 0] = positions[:, 0] * SCALE_X
            positions[:, 1] = positions[:, 1] * SCALE_Y
            positions[:, 2] = positions[:, 2] * SCALE_Z
        elif coordinates != "nm":
            raise ValueError(
                f"coordinates must be 'nm' or 'pixels', got {coordinates!r}"
            )

        neuropil_masks: dict[str, np.ndarray] = {}
        for name in neuropil_names:
            neuropil_masks[name] = load_neuropil_mesh(name).contains(positions)

        if include_other:
            any_assigned = np.zeros(len(positions), dtype=bool)
            for mask in neuropil_masks.values():
                any_assigned |= mask
            other_mask = ~any_assigned
            if other_mask.any():
                neuropil_masks["other"] = other_mask

        result = NeuropilCollection()
        for name, mask in neuropil_masks.items():
            if not mask.any():
                continue
            matrix = cls.from_synapses(
                synapses_df=scoped_synapses[mask],
                neuron_annotations=neuron_annotations,
                source_types=source_types,
                target_types=target_types,
                source_neurons=source_neurons,
                target_neurons=target_neurons,
                pre_col=pre_col,
                post_col=post_col,
                weight_mode=weight_mode,
                weight_column=weight_column,
                normalization_scope=normalization_scope,
                cell_type_column=cell_type_column,
                neuron_id_column=neuron_id_column,
                type_order=type_order,
                source_type_order=source_type_order,
                target_type_order=target_type_order,
                annotation_scope=annotation_scope,
            )
            if matrix.matrix.shape[0] == 0 or matrix.matrix.shape[1] == 0:
                continue
            result[name] = matrix

        return result

    @property
    def sum_type_matrix(self) -> pd.DataFrame:
        """Sum source-to-target connectivity by source and target cell type."""
        return _readonly_dataframe(self._sum_type_matrix)

    @cached_property
    def _sum_type_matrix(self) -> pd.DataFrame:
        return self._aggregate_type_matrix("sum")

    @property
    def mean_type_matrix(self) -> pd.DataFrame:
        """Mean source-to-target connectivity by source and target cell type."""
        return _readonly_dataframe(self._mean_type_matrix)

    @cached_property
    def _mean_type_matrix(self) -> pd.DataFrame:
        return self._aggregate_type_matrix("mean")

    def _aggregate_type_matrix(self, aggregate: str) -> pd.DataFrame:
        source_names = list(self._source_type_boundaries.keys())
        target_names = list(self._target_type_boundaries.keys())
        source_bounds = list(self._source_type_boundaries.values())
        target_bounds = list(self._target_type_boundaries.values())
        data = self._matrix.to_numpy()
        result = np.empty((len(source_names), len(target_names)), dtype=float)

        for i, (r_start, r_end) in enumerate(source_bounds):
            for j, (c_start, c_end) in enumerate(target_bounds):
                block = data[r_start:r_end, c_start:c_end]
                if aggregate == "sum":
                    value = block.sum()
                elif aggregate == "mean":
                    value = block.mean()
                else:
                    raise ValueError(f"Unsupported type aggregation: {aggregate!r}")
                result[i, j] = float(value)

        return pd.DataFrame(result, index=source_names, columns=target_names)

    def get_relative_weights(self, by_type: bool = False) -> pd.DataFrame:
        """Calculate row-normalized source-to-target weights."""
        df = self._sum_type_matrix if by_type else self._matrix
        row_sums = df.sum(axis=1).replace(0, 1)
        return df.div(row_sums, axis=0)

    def plot(
        self,
        output_path: str | None = None,
        level: Literal["neuron", "type_mean", "type_sum"] = "neuron",
        figsize: tuple[int, int] = (16, 14),
        show_neuron_labels: bool = False,
        vmin_percentile: float = 0.0,
        vmax_percentile: float = 100.0,
        min_neurons_for_plot: int = 1,
        linewidth_scale: float = 1.0,
    ) -> tuple[Figure, Axes]:
        """Plot the rectangular directed connectivity matrix as a heatmap."""
        valid_levels = {"neuron", "type_mean", "type_sum"}
        if level not in valid_levels:
            raise ValueError(
                f"level must be one of {sorted(valid_levels)}, got {level!r}"
            )
        if linewidth_scale <= 0:
            raise ValueError(
                f"linewidth_scale must be positive, got {linewidth_scale!r}"
            )

        if level == "type_mean":
            data = self._mean_type_matrix
            source_boundaries = {
                name: (i, i + 1) for i, name in enumerate(data.index)
            }
            target_boundaries = {
                name: (i, i + 1) for i, name in enumerate(data.columns)
            }
            source_labels = list(data.index)
            target_labels = list(data.columns)
            colorbar_label = "Mean Weight"
        elif level == "type_sum":
            data = self._sum_type_matrix
            source_boundaries = {
                name: (i, i + 1) for i, name in enumerate(data.index)
            }
            target_boundaries = {
                name: (i, i + 1) for i, name in enumerate(data.columns)
            }
            source_labels = list(data.index)
            target_labels = list(data.columns)
            colorbar_label = "Total Weight"
        else:
            plot_data = self._filter_for_plot(min_neurons_for_plot)
            data = plot_data.matrix
            source_boundaries = plot_data.source_boundaries
            target_boundaries = plot_data.target_boundaries
            source_labels = plot_data.source_labels
            target_labels = plot_data.target_labels
            colorbar_label = "Weight"

        if data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError(
                "Cannot plot a directed matrix with zero rows or columns; "
                f"got shape {data.shape}"
            )

        fig, ax = plt.subplots(figsize=figsize)
        NestedMatrix._draw_heatmap(
            self,
            ax,
            data,
            vmin_percentile,
            vmax_percentile,
            colorbar_label=colorbar_label,
        )

        self._draw_rectangular_boundaries(
            ax,
            source_boundaries=source_boundaries,
            target_boundaries=target_boundaries,
            shape=data.shape,
            linewidth_scale=linewidth_scale,
        )

        show_type_labels = (
            not show_neuron_labels
            and level == "neuron"
            and len(source_boundaries) <= _MAX_CENTERED_TYPE_LABELS
            and len(target_boundaries) <= _MAX_CENTERED_TYPE_LABELS
        )
        self._configure_rectangular_axes(
            ax,
            source_labels=source_labels if show_neuron_labels else None,
            target_labels=target_labels if show_neuron_labels else None,
            source_boundaries=source_boundaries
            if show_type_labels or level != "neuron"
            else None,
            target_boundaries=target_boundaries
            if show_type_labels or level != "neuron"
            else None,
        )

        plt.tight_layout()
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")

        return fig, ax

    @staticmethod
    def _filtered_axis_indices(
        boundaries: dict[str, tuple[int, int]],
        min_neurons: int,
    ) -> tuple[list[int], dict[str, tuple[int, int]]]:
        kept_indices = []
        new_boundaries = {}
        current_idx = 0
        for name, (start, end) in boundaries.items():
            count = end - start
            if count >= min_neurons:
                indices = list(range(start, end))
                kept_indices.extend(indices)
                new_boundaries[name] = (current_idx, current_idx + count)
                current_idx += count
        return kept_indices, new_boundaries

    def _filter_for_plot(self, min_neurons: int) -> _DirectedPlotData:
        if min_neurons <= 1:
            return _DirectedPlotData(
                self._matrix,
                self._source_type_boundaries,
                self._target_type_boundaries,
                list(self._source_neurons),
                list(self._target_neurons),
            )

        source_indices, source_boundaries = self._filtered_axis_indices(
            self._source_type_boundaries,
            min_neurons,
        )
        target_indices, target_boundaries = self._filtered_axis_indices(
            self._target_type_boundaries,
            min_neurons,
        )
        filtered_matrix = self._matrix.iloc[source_indices, target_indices]
        source_labels = [self._source_neurons[i] for i in source_indices]
        target_labels = [self._target_neurons[i] for i in target_indices]

        return _DirectedPlotData(
            filtered_matrix,
            source_boundaries,
            target_boundaries,
            source_labels,
            target_labels,
        )

    def _draw_rectangular_boundaries(
        self,
        ax: Axes,
        source_boundaries: dict[str, tuple[int, int]],
        target_boundaries: dict[str, tuple[int, int]],
        shape: tuple[int, int],
        linewidth_scale: float = 1.0,
    ) -> None:
        nrows, ncols = shape
        if nrows == 0 or ncols == 0:
            return

        linewidths = {
            name: width * linewidth_scale
            for name, width in _DEFAULT_BOUNDARY_LINEWIDTHS.items()
        }
        outer_segments = [
            [(-0.5, -0.5), (ncols - 0.5, -0.5)],
            [(ncols - 0.5, -0.5), (ncols - 0.5, nrows - 0.5)],
            [(ncols - 0.5, nrows - 0.5), (-0.5, nrows - 0.5)],
            [(-0.5, nrows - 0.5), (-0.5, -0.5)],
        ]
        ax.add_collection(
            LineCollection(
                outer_segments,
                colors="black",
                linewidths=linewidths["outer"],
            )
        )

        grid_segments = []
        for _, (_, end) in source_boundaries.items():
            if end < nrows:
                pos = end - 0.5
                grid_segments.append([(-0.5, pos), (ncols - 0.5, pos)])
        for _, (_, end) in target_boundaries.items():
            if end < ncols:
                pos = end - 0.5
                grid_segments.append([(pos, -0.5), (pos, nrows - 0.5)])

        if grid_segments:
            ax.add_collection(
                LineCollection(
                    grid_segments,
                    colors="white",
                    linewidths=linewidths["grid_light"],
                    alpha=1.0,
                )
            )
            ax.add_collection(
                LineCollection(
                    grid_segments,
                    colors="black",
                    linewidths=linewidths["grid_dark"],
                    alpha=0.9,
                )
            )

    @staticmethod
    def _axis_ticks(
        boundaries: dict[str, tuple[int, int]] | None,
        labels: list[str] | None,
    ) -> tuple[list[float] | np.ndarray, list[str]]:
        if boundaries:
            ticks = []
            tick_labels = []
            for name, (start, end) in boundaries.items():
                ticks.append((start + end - 1) / 2)
                tick_labels.append(name)
            return ticks, tick_labels

        if labels:
            return np.arange(len(labels)), labels

        return [], []

    def _configure_rectangular_axes(
        self,
        ax: Axes,
        source_labels: list[str] | None,
        target_labels: list[str] | None,
        source_boundaries: dict[str, tuple[int, int]] | None,
        target_boundaries: dict[str, tuple[int, int]] | None,
    ) -> None:
        xticks, xtick_labels = self._axis_ticks(target_boundaries, target_labels)
        yticks, ytick_labels = self._axis_ticks(source_boundaries, source_labels)

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=14)
        ax.set_yticklabels(ytick_labels, fontsize=14)
        ax.set_xlabel("Postsynaptic", fontsize=18, fontweight="bold")
        ax.set_ylabel("Presynaptic", fontsize=18, fontweight="bold")
