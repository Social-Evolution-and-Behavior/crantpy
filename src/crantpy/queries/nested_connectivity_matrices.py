"""Helpers for building and plotting nested connectivity matrices."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "create_nested_connectivity_matrix",
    "plot_nested_connectivity_matrix",
    "calculate_relative_weights",
    "convert_synapses_to_filter_format",
    "NestedMatrix",
]

#macros for custom plots
BOUNDARY_LINE_WIDTH = 2.5
BOUNDARY_LINE_ALPHA_INNER = 0.9
PLOT_DPI = 300
NEURON_LABEL_STEP_DIVISOR = 50
NEURON_LABEL_FONTSIZE = 10
TICK_LABEL_FONTSIZE = 16
AXIS_LABEL_FONTSIZE = 18
COLORBAR_LABEL_FONTSIZE = 18
COLORBAR_TICK_FONTSIZE = 16

#macros for the custom cmap
SORT_PRIORITY_ER_PREFIX = 0
SORT_PRIORITY_OTHER_PREFIX = 1
SORT_PRIORITY_NO_MATCH = 2
SORT_FALLBACK_NUMBER = 999999

RELATIVE_WEIGHT_TOLERANCE = 1e-3

COLORBAR_FRACTION = 0.046
COLORBAR_PAD = 0.04
#macros for the nests/plots 
OUTER_BOUNDARY_WIDTH = 3
OUTER_BOUNDARY_ALPHA = 1.0
PLOT_PIXEL_OFFSET = 0.5


def _sorted_cell_types(types: List[str], preferred: List[str] | None = None) -> List[str]:
    """Return cell types ordered numerically when possible (ER1, ER2, …)."""

    unique: List[str] = []
    seen: set[str] = set()
    for label in types:
        if label is None:
            continue
        if label not in seen:
            seen.add(label)
            unique.append(label)

    def sort_key(label: str) -> Tuple[int, int, str]:
        if not isinstance(label, str):
            return (SORT_PRIORITY_NO_MATCH, SORT_FALLBACK_NUMBER, str(label))

        prefix_match = re.match(r"([A-Za-z]+)(\d+)(.*)", label)
        if prefix_match:
            prefix, number, suffix = prefix_match.groups()
            priority = SORT_PRIORITY_ER_PREFIX if prefix.upper() == "ER" else SORT_PRIORITY_OTHER_PREFIX
            return (priority, int(number), suffix.upper())

        return (SORT_PRIORITY_NO_MATCH, SORT_FALLBACK_NUMBER, label.upper())

    ordered = sorted(unique, key=sort_key)

    if preferred:
        preferred_order = [label for label in preferred if label in seen]
        preferred_order.extend([label for label in ordered if label not in preferred_order])
        return preferred_order

    return ordered


def _build_type_matrix(
    matrix: pd.DataFrame,
    type_boundaries: Dict[str, Tuple[int, int]],
    ordered_neurons: List[str],
) -> pd.DataFrame:
    """Aggregate neuron-level weights into type-to-type blocks."""

    if not type_boundaries:
        return pd.DataFrame()

    type_names = list(type_boundaries.keys())
    type_matrix = pd.DataFrame(0.0, index=type_names, columns=type_names)

    for from_type, (row_start, row_end) in type_boundaries.items():
        from_neurons = ordered_neurons[row_start:row_end]
        if not from_neurons:
            continue
        row_block = matrix.loc[from_neurons]

        for to_type, (col_start, col_end) in type_boundaries.items():
            to_neurons = ordered_neurons[col_start:col_end]
            if not to_neurons:
                continue
            type_matrix.loc[from_type, to_type] = (
                row_block.loc[:, to_neurons].sum().sum()
            )

    return type_matrix


@dataclass
class NestedMatrix:
    """Container holding neuron- and type-level connectivity matrices."""

    matrix: pd.DataFrame
    type_boundaries: Dict[str, Tuple[int, int]]
    ordered_neurons: List[str]
    neuron_to_type: Dict[str, Any]
    type_matrix: pd.DataFrame

    def to_type_matrix(self, recompute: bool = False) -> pd.DataFrame:
        """Return the type-level matrix, recomputing if requested."""

        if recompute or self.type_matrix.empty:
            self.type_matrix = _build_type_matrix(
                self.matrix, self.type_boundaries, self.ordered_neurons
            )
        return self.type_matrix.copy()


def _coerce_to_adjacency_matrix(connectivity: pd.DataFrame) -> pd.DataFrame:
    """Return a neuron-by-neuron adjacency matrix from various inputs."""

    if not isinstance(connectivity, pd.DataFrame):
        raise TypeError("connectivity input must be a pandas DataFrame")

    df = connectivity.copy()

    def _pivot(frame: pd.DataFrame, source_col: str, target_col: str, weight_col: str) -> pd.DataFrame:
        subset = frame[[source_col, target_col, weight_col]].copy()
        subset[source_col] = subset[source_col].astype(str)
        subset[target_col] = subset[target_col].astype(str)
        return (
            subset.pivot(index=source_col, columns=target_col, values=weight_col)
            .fillna(0)
            .astype(float)
        )

    columns = set(df.columns)

    if {"type.from", "type.to"}.issubset(columns):
        weight_col = None
        for candidate in ("weight", "weightRelative", "weight_relative"):
            if candidate in columns:
                weight_col = candidate
                break
        if weight_col is None:
            raise ValueError(
                "Expected a weight column alongside 'type.from'/'type.to' in connectivity DataFrame"
            )
        adjacency = _pivot(df, "type.from", "type.to", weight_col)
    elif {"pre", "post"}.issubset(columns):
        frame = df.rename(columns={"pre": "source", "post": "target"})
        if "weight" not in frame.columns:
            frame["weight"] = 1
        adjacency = _pivot(frame, "source", "target", "weight")
    elif {"source", "target"}.issubset(columns):
        frame = df.copy()
        if "weight" not in frame.columns and "n_syn" in frame.columns:
            frame = frame.rename(columns={"n_syn": "weight"})
        if "weight" not in frame.columns:
            frame["weight"] = 1
        adjacency = _pivot(frame, "source", "target", "weight")
    elif df.index.size and df.columns.size:
        adjacency = df.astype(float)
    else:
        adjacency = df.astype(float)

    return adjacency.fillna(0)


def _build_ordered_neurons(
    relevant_annotations: pd.DataFrame,
    neuron_ids: set[str],
    neuron_id_column: str,
    cell_type_column: str = "cell_type",
    type_order: List[str] | None = None,
) -> tuple[list[str], dict[str, tuple[int, int]]]:
    """Return (ordered_neurons, type_boundaries) using annotation order.

    Preserves first-occurrence order of types in annotations and places
    annotated neurons first, then remaining neurons sorted alphabetically.
    """
    ann = relevant_annotations.copy()
    if neuron_id_column in ann.columns:
        ann = ann.assign(**{neuron_id_column: ann[neuron_id_column].astype(str)})
        ann = ann.drop_duplicates(subset=[neuron_id_column], keep="first")
    else:
        ann = ann.copy()

    seen = set()
    present_types: list[str] = []
    for c in ann[cell_type_column]:
        if pd.notna(c) and c not in seen:
            seen.add(c)
            present_types.append(str(c))

    ordered_types = _sorted_cell_types(present_types, preferred=type_order)

    id_to_type_map = {
        str(k): v
        for k, v in zip(
            ann[neuron_id_column].astype(str),
            ann[cell_type_column],
        )
    }

    neuron_ids_str = set(map(str, neuron_ids))

    ordered_neurons: list[str] = []
    type_boundaries: dict[str, tuple[int, int]] = {}
    current_pos = 0

    for cell_type in ordered_types:
        all_type_neurons = [nid for nid in neuron_ids_str if id_to_type_map.get(nid) == cell_type]

        annotated_neurons = list(
            ann[ann[cell_type_column] == cell_type][neuron_id_column].astype(str)
        )
        annotated_unique = []
        seen_local = set()
        for nid in annotated_neurons:
            if nid in all_type_neurons and nid not in seen_local:
                seen_local.add(nid)
                annotated_unique.append(nid)

        remaining = [nid for nid in sorted(all_type_neurons) if nid not in set(annotated_unique)]

        type_neurons = annotated_unique + remaining

        if type_neurons:
            type_boundaries[cell_type] = (current_pos, current_pos + len(type_neurons))
            ordered_neurons.extend(type_neurons)
            current_pos += len(type_neurons)

    unassigned = [nid for nid in sorted(neuron_ids_str) if nid not in set(ordered_neurons)]
    if unassigned:
        ordered_neurons.extend(unassigned)

    return ordered_neurons, type_boundaries



def create_nested_connectivity_matrix(
    connections_df: pd.DataFrame,
    neuron_annotations: pd.DataFrame,
    cell_type_column: str = "cell_type",
    neuron_id_column: str = "root_id",
    type_order: List[str] | None = None,
) -> NestedMatrix:
    """Return a neuron-level nested connectivity matrix grouped by cell type."""

    adjacency_matrix = _coerce_to_adjacency_matrix(connections_df)
    adjacency_matrix.index = adjacency_matrix.index.astype(str)
    adjacency_matrix.columns = adjacency_matrix.columns.astype(str)

    neuron_id_order = list(
        dict.fromkeys(
            adjacency_matrix.index.tolist() + adjacency_matrix.columns.tolist()
        )
    )
    neuron_ids = set(neuron_id_order)

    relevant_annotations = neuron_annotations[
        neuron_annotations[neuron_id_column].astype(str).isin(neuron_ids)
    ].copy()

    if cell_type_column not in relevant_annotations.columns:
        raise KeyError(
            f"Column '{cell_type_column}' not found in annotations dataframe"
        )

    def _prepare_type(value: object) -> str | None:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        value_str = str(value).strip()
        return value_str if value_str else None

    relevant_annotations["cell_type"] = relevant_annotations[
        cell_type_column
    ].apply(_prepare_type)

    neuron_to_type = {
        str(neuron_id): cell_type
        for neuron_id, cell_type in zip(
            relevant_annotations[neuron_id_column].astype(str),
            relevant_annotations["cell_type"],
        )
    }

    ordered_neurons, type_boundaries = _build_ordered_neurons(
        relevant_annotations,
        neuron_ids,
        neuron_id_column,
        cell_type_column,
        type_order,
    )

    if not ordered_neurons:
        empty = pd.DataFrame()
        return NestedMatrix(
            matrix=empty,
            type_boundaries={},
            ordered_neurons=[],
            neuron_to_type=neuron_to_type,
            type_matrix=empty,
        )

    matrix = adjacency_matrix.reindex(
        index=ordered_neurons, columns=ordered_neurons, fill_value=0
    ).astype(float)

    type_matrix = _build_type_matrix(matrix, type_boundaries, ordered_neurons)

    return NestedMatrix(
        matrix=matrix,
        type_boundaries=type_boundaries,
        ordered_neurons=ordered_neurons,
        neuron_to_type=neuron_to_type,
        type_matrix=type_matrix,
    )






def _filter_neuron_level_matrix(
    matrix: pd.DataFrame,
    boundaries: dict[str, tuple[int, int]],
    ordered_neurons: list[str],
    min_neurons_for_plot: int,
) -> tuple[pd.DataFrame, dict[str, tuple[int, int]], list[str], list[str]]:
    """Remove small neuron clusters from the plot matrix."""

    if min_neurons_for_plot <= 1:
        return matrix, boundaries, ordered_neurons, []

    kept_neurons: list[str] = []
    new_boundaries: dict[str, tuple[int, int]] = {}
    removed_types: list[str] = []
    current_index = 0

    for cell_type, (start, end) in boundaries.items():
        neurons = ordered_neurons[start:end]
        size = len(neurons)
        if size >= min_neurons_for_plot:
            new_boundaries[cell_type] = (current_index, current_index + size)
            kept_neurons.extend(neurons)
            current_index += size
        else:
            removed_types.append(cell_type)

    if not kept_neurons:
        return matrix, boundaries, ordered_neurons, removed_types

    
    kept_indices = [i for i, nid in enumerate(ordered_neurons) if nid in kept_neurons]
    
    filtered_matrix = matrix.iloc[kept_indices, kept_indices].copy()
    
    filtered_matrix.index = pd.Index(kept_neurons)
    filtered_matrix.columns = pd.Index(kept_neurons)

    return filtered_matrix, new_boundaries, kept_neurons, removed_types






def plot_nested_connectivity_matrix(
    nested: NestedMatrix,
    *,
    figsize: tuple[int, int] = (16, 14),
    show_neuron_labels: bool = False,
    output_path: str | None = None,
    min_neurons_for_plot: int = 4,
    column_boundaries: dict[str, tuple[int, int]] | None = None,
    column_order: list[str] | None = None,
    vmin_percentile: float = 0.0,
    vmax_percentile: float = 100.0,
):
    """Plot a nested neuron-by-neuron connectivity matrix with type boundaries."""

    relative_matrix = nested.matrix
    type_boundaries = nested.type_boundaries
    ordered_neurons = nested.ordered_neurons

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle


    colors = [
        "#FFFFFF",
        "#E8D5E8",
        "#D1AAD1",
        "#BA7FBA",
        "#A355A3",
        "#8B2B8B",
        "#740074",
    ]
    purple_cmap = LinearSegmentedColormap.from_list("purple", colors, N=100)

    row_order = ordered_neurons if ordered_neurons else list(relative_matrix.index)
    if column_order is None:
        if relative_matrix.shape[1] == len(row_order):
            resolved_column_order = row_order
        else:
            resolved_column_order = list(relative_matrix.columns)
    else:
        resolved_column_order = column_order

    rectangular_layout = (
        column_boundaries is not None
        or column_order is not None
        or relative_matrix.shape[0] != relative_matrix.shape[1]
        or resolved_column_order != row_order
    )

    total_weight = float(np.nansum(relative_matrix.values)) if relative_matrix.size else 0.0
    cbar_label = "Relative Weight" if np.isclose(total_weight, 1.0, rtol=RELATIVE_WEIGHT_TOLERANCE) else "Weight"

    if rectangular_layout:
        plot_matrix = relative_matrix.loc[row_order, resolved_column_order]
        fig, ax = plt.subplots(figsize=figsize)

        plot_vals = plot_matrix.values.astype(float)
        plot_vals = np.where(plot_vals == 0, np.nan, plot_vals)
        nrows, ncols = plot_vals.shape
        extent = (-PLOT_PIXEL_OFFSET, ncols - PLOT_PIXEL_OFFSET, -PLOT_PIXEL_OFFSET, nrows - PLOT_PIXEL_OFFSET)
        im = ax.imshow(plot_vals, cmap=purple_cmap, origin='lower', extent=extent, interpolation='nearest')

        cbar = plt.colorbar(im, ax=ax, fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD)
        cbar.set_label(
            cbar_label.lower(), fontsize=COLORBAR_LABEL_FONTSIZE, fontweight="bold"
        )

        ax.set_xlabel('postsynaptic neuron', fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
        ax.set_ylabel('presynaptic neuron', fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        plt.tight_layout()
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
        return fig, ax

    relative_matrix = pd.DataFrame(relative_matrix).reindex(
        index=row_order, columns=row_order
    )

    is_type_level = all((end - start) <= 1 for start, end in type_boundaries.values())

    plot_matrix = relative_matrix
    plot_boundaries = type_boundaries
    filtered_neurons = row_order

    if not is_type_level:
        plot_matrix, plot_boundaries, filtered_neurons, removed_types = (
            _filter_neuron_level_matrix(
                relative_matrix, type_boundaries, row_order, min_neurons_for_plot
            )
        )

        if removed_types:
            print(
                f"Filtered out small clusters (< {min_neurons_for_plot} neurons): {removed_types}"
            )

        if filtered_neurons != row_order:
            print(
                f"Filtered matrix: kept {len(filtered_neurons)}/{len(row_order)} neurons "
                f"({len(plot_boundaries)} types)"
            )

        

    else:
        filtered_neurons = list(plot_matrix.index)

    values = plot_matrix.values.flatten()
    values = values[values > 0]
    if len(values) > 0:
        color_min = float(np.percentile(values, vmin_percentile))
        color_max = float(np.percentile(values, vmax_percentile))
        if color_max == color_min:
            color_max = color_min + 1.0
    else:
        color_min = 0.0
        color_max = 1.0

    fig, ax = plt.subplots(figsize=figsize)
    plot_vals = plot_matrix.values.astype(float)
    plot_vals = np.where(plot_vals == 0, np.nan, plot_vals)
    nrows, ncols = plot_vals.shape
    extent = (-PLOT_PIXEL_OFFSET, ncols - PLOT_PIXEL_OFFSET, -PLOT_PIXEL_OFFSET, nrows - PLOT_PIXEL_OFFSET)
    im = ax.imshow(plot_vals, cmap=purple_cmap, vmin=color_min, vmax=color_max, origin='lower', extent=extent, interpolation='nearest')

    for cell_type, (start_pos, end_pos) in plot_boundaries.items():
        width = end_pos - start_pos
        height = end_pos - start_pos
        rect = Rectangle((start_pos - PLOT_PIXEL_OFFSET, start_pos - PLOT_PIXEL_OFFSET), width, height, linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

    ax.set_xlabel('Postsynaptic Neuron', fontsize=AXIS_LABEL_FONTSIZE, labelpad=40)
    ax.set_ylabel('Presynaptic Neuron', fontsize=AXIS_LABEL_FONTSIZE, labelpad=40)
    ax.set_xlim(-PLOT_PIXEL_OFFSET, ncols - PLOT_PIXEL_OFFSET)
    ax.set_ylim(-PLOT_PIXEL_OFFSET, nrows - PLOT_PIXEL_OFFSET)

    plt.tight_layout()

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')

    cbar = plt.colorbar(im, ax=ax, fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD)
    cbar.set_label(
        cbar_label.lower(), fontsize=COLORBAR_LABEL_FONTSIZE, fontweight='bold'
    )
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

    type_names = list(plot_boundaries.keys())
    type_boundary_positions = []

    for cell_type in type_names:
        start_pos, end_pos = plot_boundaries[cell_type]
        if start_pos > 0:
            boundary_position = start_pos - PLOT_PIXEL_OFFSET
            type_boundary_positions.append(boundary_position)
            ax.axhline(boundary_position, color="white", linewidth=2, alpha=OUTER_BOUNDARY_ALPHA)
            ax.axvline(boundary_position, color="white", linewidth=2, alpha=OUTER_BOUNDARY_ALPHA)

    matrix_size = len(plot_matrix)
    ax.axhline(-PLOT_PIXEL_OFFSET, color="black", linewidth=OUTER_BOUNDARY_WIDTH, alpha=OUTER_BOUNDARY_ALPHA)
    ax.axhline(matrix_size - PLOT_PIXEL_OFFSET, color="black", linewidth=OUTER_BOUNDARY_WIDTH, alpha=OUTER_BOUNDARY_ALPHA)
    ax.axvline(-PLOT_PIXEL_OFFSET, color="black", linewidth=OUTER_BOUNDARY_WIDTH, alpha=OUTER_BOUNDARY_ALPHA)
    ax.axvline(matrix_size - PLOT_PIXEL_OFFSET, color="black", linewidth=OUTER_BOUNDARY_WIDTH, alpha=OUTER_BOUNDARY_ALPHA)

    for boundary_pos in type_boundary_positions:
        ax.axhline(
            boundary_pos,
            color="black",
            linewidth=BOUNDARY_LINE_WIDTH,
            alpha=BOUNDARY_LINE_ALPHA_INNER,
        )
        ax.axvline(
            boundary_pos,
            color="black",
            linewidth=BOUNDARY_LINE_WIDTH,
            alpha=BOUNDARY_LINE_ALPHA_INNER,
        )

    if show_neuron_labels and filtered_neurons:
        n_neurons = len(filtered_neurons)
        step = max(1, n_neurons // NEURON_LABEL_STEP_DIVISOR)
        tick_positions = list(range(0, n_neurons, step))
        tick_labels = [filtered_neurons[i] for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(
            tick_labels, rotation=90, ha="right", fontsize=NEURON_LABEL_FONTSIZE
        )
        ax.set_yticklabels(tick_labels, fontsize=NEURON_LABEL_FONTSIZE)
    else:
        type_tick_positions = []
        type_labels = []
        for cell_type in type_names:
            if cell_type in plot_boundaries:
                start_pos, end_pos = plot_boundaries[cell_type]
                center_pos = (start_pos + end_pos - 1) / 2
                type_tick_positions.append(center_pos)
                type_labels.append(cell_type)

        ax.set_xticks(type_tick_positions)
        ax.set_yticks(type_tick_positions)
        ax.set_xticklabels(
            type_labels, rotation=45, ha="right", fontsize=TICK_LABEL_FONTSIZE
        )
        ax.set_yticklabels(type_labels, fontsize=TICK_LABEL_FONTSIZE)

    ax.set_xlabel(
        "postsynaptic neuron", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold"
    )
    ax.set_ylabel("presynaptic neuron", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")

    mappable = ax.collections[0] if ax.collections else None
    if mappable is not None:
        cbar = plt.colorbar(mappable, ax=ax, fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD)
        cbar.set_label(
            cbar_label.lower(), fontsize=COLORBAR_LABEL_FONTSIZE, fontweight="bold"
        )
        cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

    

    plt.tight_layout()

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")

    return fig, ax


def calculate_relative_weights(
    df: pd.DataFrame,
    roi_column: str = "roi",
    type_to_column: str = "type.to",
    weight_column: str = "weight",
) -> pd.DataFrame:
    """Compute per-row relative weight within each ROI/target group.

    The helper computes a new column ``weightRelative`` which is the fraction
    of the row's weight relative to the total weight for the same
    (roi, type.to) grouping. This is useful when converting synapse counts
    into fractions that sum to 1 over the postsynaptic target group within an
    ROI.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing at least the columns specified by
        ``roi_column``, ``type_to_column`` and ``weight_column``.
    roi_column : str
        Column name representing the region-of-interest grouping. Defaults to
        ``'roi'``; if not present callers often set it to ``'all'`` before
        calling this helper.
    type_to_column : str
        Column name identifying the postsynaptic neuron (or target type)
        against which weights are normalised.
    weight_column : str
        Column that contains the numeric weight to normalise.

    Returns
    -------
    pd.DataFrame
        A copy of the input dataframe with an added ``weightRelative`` column.
    """

    result_df = df.copy()

    total_weights = result_df.groupby([roi_column, type_to_column])[weight_column].sum()

    result_df["weightRelative"] = result_df.apply(
        lambda row: row[weight_column]
        / total_weights.get((row[roi_column], row[type_to_column]), 1),
        axis=1,
    )

    return result_df


def convert_synapses_to_filter_format(
    synapses_df: pd.DataFrame,
    roi_column: str | None = None,
    type_from_column: str = "pre_pt_root_id",
    type_to_column: str = "post_pt_root_id",
    weight_column: str = "Weight",
    which_weights: str = "relative",
) -> pd.DataFrame:
    """Convert a synapse-level table to the standard filter/connectivity format.

    This helper turns a DataFrame of synapse objects (or simple synapse rows)
    into a compact connection table summarised per (roi, pre, post). The
    produced table contains the following columns (at minimum):
    - ``roi``: region-of-interest (defaults to ``'all'`` when not provided)
    - ``type.from``: presynaptic neuron id
    - ``type.to``: postsynaptic neuron id
    - ``weight``: number of synapses between the presynaptic and postsynaptic
      pair (or the value of ``weight_column`` if provided)
    - ``weightRelative``: fraction of the weight relative to all weights for
      the same (roi, type.to) grouping (computed via
      :func:`calculate_relative_weights`).

    The function also adds convenience columns ``from``, ``to``, ``name.from``
    and ``name.to`` which mirror ``type.from``/``type.to`` for compatibility
    with other crantpy helpers and plotting functions.

    Parameters
    ----------
    synapses_df : pd.DataFrame
        Input synapse-level table. May contain explicit pre/post id columns
        and an optional weight column.
    roi_column : str or None
        Column name to use for ROI. If ``None`` or if the supplied name is not
        in the dataframe, a default column ``'roi'`` with value ``'all'`` is
        added.
    type_from_column, type_to_column : str
        Column names in ``synapses_df`` that contain presynaptic and
        postsynaptic neuron ids respectively.
    weight_column : str
        Column name containing synapse weights. If absent the function will
        count synapse rows per (roi, pre, post) and use that as the weight.

    Returns
    -------
    pd.DataFrame
        Connection table with relative weights and convenience ID columns.
    """

    result_df = synapses_df.copy()

    if roi_column is None:
        result_df["roi"] = "all"
    elif roi_column not in result_df.columns:
        result_df["roi"] = "all"
    else:
        result_df = result_df.rename(columns={roi_column: "roi"})

    column_mapping = {type_from_column: "type.from", type_to_column: "type.to"}
    result_df = result_df.rename(columns=column_mapping)

    if weight_column not in result_df.columns:
        synapse_counts = (
            result_df.groupby(["roi", "type.from", "type.to"]).size().reset_index(name="weight")
        )
        result_df = synapse_counts
    else:
        result_df = result_df.rename(columns={weight_column: "weight"})
    if which_weights == "relative":
        result_df = calculate_relative_weights(result_df)
    else:
        result_df["weightRelative"] = result_df["weight"]

    
    result_df["from"] = result_df["type.from"]
    result_df["to"] = result_df["type.to"]
    result_df["name.from"] = result_df["type.from"]
    result_df["name.to"] = result_df["type.to"]

    return result_df.reset_index(drop=True)
