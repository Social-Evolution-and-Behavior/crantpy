# -*- coding: utf-8 -*-
"""
Functionality for creating and visualizing nested connectivity matrices.

The NestedMatrix class enables the construction of hierarchical connectivity matrices
that organize neurons by cell type, allowing for both neuron-level and type-level
connectivity analysis. It supports visualization with customizable boundaries,
filtering, and relative weight calculations. ROI-resolved workflows are available
through ``NestedMatrix.from_synapses_by_neuropil()``, which returns one nested
matrix per neuropil ROI.


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
from functools import cached_property
from typing import Any, Literal, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

__all__ = ["NestedMatrix", "NeuropilCollection", "All"]

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


class NeuropilCollection(dict):
    """Dict subclass mapping neuropil names to NestedMatrix instances.

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
            Forwarded to ``NestedMatrix.plot()``.

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

    def __getattr__(self, name: str) -> "NestedMatrix":
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

       Instances are treated as immutable after construction. Do not mutate
       ``matrix``, ``type_boundaries``, ``ordered_neurons``, or
       ``neuron_to_type`` — derived properties like ``sum_type_matrix`` and
       ``mean_type_matrix`` are cached on first access and will not reflect
       later changes.

    Attributes
    ----------
    matrix : pd.DataFrame
        Full neuron-to-neuron connectivity matrix with neurons ordered by type.
    type_boundaries : dict[str, tuple[int, int]]
        Dictionary mapping cell type names to (start, end) index tuples
        indicating where each type appears in the matrix.
    ordered_neurons : list[str]
        List of neuron IDs in the order they appear in the matrix.
    neuron_to_type : dict[str, Any]
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
        type_boundaries: dict[str, tuple[int, int]],
        ordered_neurons: list[str],
        neuron_to_type: dict[str, Any],
    ):
        matrix = matrix.copy()
        matrix.index = matrix.index.astype(str)
        matrix.columns = matrix.columns.astype(str)
        ordered_neurons = [str(neuron) for neuron in ordered_neurons]
        type_boundaries = {
            str(name): (int(start), int(end))
            for name, (start, end) in type_boundaries.items()
        }
        neuron_to_type = {
            str(neuron): cell_type for neuron, cell_type in neuron_to_type.items()
        }

        self._validate_invariants(
            matrix=matrix,
            type_boundaries=type_boundaries,
            ordered_neurons=ordered_neurons,
            neuron_to_type=neuron_to_type,
        )

        self.matrix = matrix
        self.type_boundaries = type_boundaries
        self.ordered_neurons = ordered_neurons
        self.neuron_to_type = neuron_to_type

    def __repr__(self) -> str:
        n_neurons = len(self.ordered_neurons)
        n_types = len(self.type_boundaries)
        types = list(self.type_boundaries.keys())
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
        type_order: list[str] | None = None,
        annotation_scope: Literal["annotated_only", "all"] = "annotated_only",
    ) -> NestedMatrix:
        """Create a NestedMatrix from a connectivity dataframe that you can get with cp.get_connectivity().

        Constructs a nested connectivity matrix from an adjacency or edge-list
        dataframe, organizing neurons by their cell type annotations. Type
        blocks follow ``type_order`` when provided, otherwise a generic
        label-aware sort is used. Within each type block, neurons preserve the
        resolved annotation row order, except ``"EPG/PEG"`` rows which use the
        EB column order encoded in ``cell_subtype`` when available.

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
        type_order : list[str] | None, optional
            Preferred order for cell types. Types will be sorted alphabetically
            if not provided, with numeric suffixes handled intelligently.
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
        type_order: list[str] | None = None,
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
        for ``"EPG/PEG"`` rows, which use ``cell_subtype`` EB columns when
        available.

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
        type_order : list[str] | None, optional
            Preferred order for cell types. Types will be sorted alphabetically
            if not provided, with numeric suffixes handled intelligently.
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
            matrix.matrix.shape,
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
        type_order: list[str] | None = None,
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
        type_order : list[str] | None, optional
            Preferred order for cell types in each matrix.
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
                result[name].matrix.shape,
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
        return set(neuron_annotations[neuron_id_column].dropna().astype(str))

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
        keep_mask = synapses_df[pre_col].astype(str).isin(annotation_ids) & synapses_df[
            post_col
        ].astype(str).isin(annotation_ids)
        filtered = synapses_df.loc[keep_mask].copy()
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
        row_mask = adjacency.index.astype(str).isin(annotation_ids)
        col_mask = adjacency.columns.astype(str).isin(annotation_ids)
        filtered = adjacency.loc[row_mask, col_mask].copy()

        original_ids = set(adjacency.index.astype(str)) | set(
            adjacency.columns.astype(str)
        )
        filtered_ids = set(filtered.index.astype(str)) | set(
            filtered.columns.astype(str)
        )
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

    @cached_property
    def sum_type_matrix(self) -> pd.DataFrame:
        """Calculate the cell type-level connectivity matrix (sum).

        Aggregates the neuron-level matrix into a type-level matrix by summing
        all connections between neurons of each type pair.

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
        return self._aggregate_type_matrix("sum")

    @cached_property
    def mean_type_matrix(self) -> pd.DataFrame:
        """Calculate the mean cell type-level connectivity matrix.

        Aggregates the neuron-level matrix into a type-level matrix by taking
        the arithmetic mean of all neuron-to-neuron weights within each type
        pair block. This includes zero-valued entries in the block, so larger
        cell types do not automatically dominate the visualization by virtue of
        having more neurons.

        Returns
        -------
        pd.DataFrame
            A square matrix with cell types as both index and columns,
            where each value represents the mean connectivity weight across
            all neuron pairs in the corresponding type block.
        """
        return self._aggregate_type_matrix("mean")

    def _aggregate_type_matrix(self, aggregate: str) -> pd.DataFrame:
        if not self.type_boundaries:
            return pd.DataFrame()

        type_names = list(self.type_boundaries.keys())
        bounds = list(self.type_boundaries.values())
        data = self.matrix.to_numpy()
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
        df = self.sum_type_matrix if by_type else self.matrix
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
            data = self.mean_type_matrix
            boundaries = {name: (i, i + 1) for i, name in enumerate(data.index)}
            labels = list(data.index)
            label_boundaries = None
            colorbar_label = "Mean Weight"
        elif level == "type_sum":
            data = self.sum_type_matrix
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
            return _PlotData(self.matrix, self.type_boundaries, self.ordered_neurons)

        kept_indices = []
        new_boundaries = {}
        current_idx = 0

        for name, (start, end) in self.type_boundaries.items():
            count = end - start
            if count >= min_neurons:
                indices = list(range(start, end))
                kept_indices.extend(indices)
                new_boundaries[name] = (current_idx, current_idx + count)
                current_idx += count

        filtered_matrix = self.matrix.iloc[kept_indices, kept_indices]
        filtered_labels = [self.ordered_neurons[i] for i in kept_indices]

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
            df = connectivity.copy()
            weight = "weight" if "weight" in cols else alt_weight
            if weight not in cols:
                df[weight] = 1
            df[idx_col] = df[idx_col].astype(str)
            df[col_col] = df[col_col].astype(str)
            return df.pivot(index=idx_col, columns=col_col, values=weight).fillna(0)

        return connectivity.fillna(0).astype(float)

    @staticmethod
    def _sort_cell_types(
        types: list[str], preferred: list[str] | None = None
    ) -> list[str]:
        unique = {str(t) for t in types if pd.notna(t)}

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

        for p in preferred:
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
        ann_ids = annotations[id_col].astype(str)
        relevant = annotations[ann_ids.isin(matrix_ids)].copy()
        relevant = relevant.assign(
            **{
                id_col: relevant[id_col].astype(str),
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
        preferred_type_order: list[str] | None,
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
    def _extract_eb_column_label(value: Any) -> str | None:
        if pd.isna(value):
            return None

        match = re.search(r"([LR]\d+)\s*$", str(value).strip())
        if not match:
            return None

        return match.group(1)

    @staticmethod
    def _default_within_type_order(
        type_rows: pd.DataFrame,
        neuron_id_column: str,
    ) -> list[str]:
        return type_rows[neuron_id_column].tolist()

    @staticmethod
    def _order_epg_neurons(
        type_rows: pd.DataFrame,
        neuron_id_column: str,
    ) -> list[str]:
        # TODO: Extend explicit column-based ordering to other columnar types
        # when reliable subtype metadata is available.
        sorted_rows = type_rows.copy()
        sorted_rows["__group_order__"] = np.arange(len(sorted_rows))
        sorted_rows["__eb_column__"] = sorted_rows["cell_subtype"].map(
            NestedMatrix._extract_eb_column_label
        )
        sorted_rows["__eb_rank__"] = sorted_rows["__eb_column__"].map(
            _COLUMN_ORDER_RANK
        )
        sorted_rows["__has_eb_rank__"] = sorted_rows["__eb_rank__"].notna()
        sorted_rows["__eb_rank__"] = sorted_rows["__eb_rank__"].fillna(np.inf)
        sorted_rows = sorted_rows.sort_values(
            by=["__has_eb_rank__", "__eb_rank__", "__group_order__"],
            ascending=[False, True, True],
        )

        return sorted_rows[neuron_id_column].tolist()

    @staticmethod
    def _order_neurons_within_type(
        type_name: str,
        type_rows: pd.DataFrame,
        neuron_id_column: str,
    ) -> list[str]:
        row_orderers = {"EPG/PEG": NestedMatrix._order_epg_neurons}
        orderer = row_orderers.get(type_name, NestedMatrix._default_within_type_order)

        if (
            orderer is NestedMatrix._order_epg_neurons
            and "cell_subtype" not in type_rows.columns
        ):
            orderer = NestedMatrix._default_within_type_order

        return orderer(type_rows, neuron_id_column)

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
        type_order: list[str] | None,
    ) -> tuple[list[str], dict[str, tuple[int, int]], dict[str, Any]]:
        matrix_ids = set(adjacency.index.astype(str)) | set(
            adjacency.columns.astype(str)
        )
        resolved = NestedMatrix._resolve_relevant_annotations(
            matrix_ids=matrix_ids,
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
            logger = logging.getLogger(__name__)
            logger.warning(
                "%d neuron(s) present in the adjacency matrix have annotation rows "
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
            logger = logging.getLogger(__name__)
            logger.warning(
                "%d neuron(s) present in the adjacency matrix but missing from "
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

        return ordered_neurons, boundaries, resolved.id_map
