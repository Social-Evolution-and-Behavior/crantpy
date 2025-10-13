"""Helpers for building and plotting nested connectivity matrices."""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

__all__ = [
    "create_nested_connectivity_matrix",
    "plot_nested_connectivity_matrix",
    "calculate_relative_weights",
    "convert_synapses_to_filter_format",
]



BOUNDARY_LINE_WIDTH = 2.5
BOUNDARY_LINE_ALPHA_INNER = 0.9
PLOT_DPI = 300
NEURON_LABEL_STEP_DIVISOR = 50
NEURON_LABEL_FONTSIZE = 10
TICK_LABEL_FONTSIZE = 16
AXIS_LABEL_FONTSIZE = 18
COLORBAR_LABEL_FONTSIZE = 18
COLORBAR_TICK_FONTSIZE = 16




def create_nested_connectivity_matrix(
    connections_df: pd.DataFrame,
    neuron_annotations: pd.DataFrame,
    cell_type_column: str = "cell_type",
    neuron_id_column: str = "root_id",
    detail_level: str = "type",
) -> tuple[pd.DataFrame, dict[str, tuple[int, int]], list[str], dict[Any, Any]]:
    """Create a nested connectivity matrix with optional neuron-level detail.

    This helper builds a neuron-by-neuron connectivity matrix (rows = presynaptic,
    columns = postsynaptic) from a connectivity table and an annotation table
    mapping neuron identifiers to cell types. The output can be returned either
    at the aggregated type-level (type-to-type weight matrix) or at the full
    neuron-level with type block boundaries and ordering.

    Parameters
    ----------
    connections_df : pd.DataFrame
        Table of connections. Expected columns (common names used by crantpy):
        - ``type.from``: presynaptic neuron id
        - ``type.to``: postsynaptic neuron id
        - ``weight``: numeric weight or synapse counts
        Optionally can contain ``weightRelative`` and ``roi`` columns.
    neuron_annotations : pd.DataFrame
        Annotation table keyed by neuron id (column set by ``neuron_id_column``)
        containing a cell type column (default ``cell_type``). The function
        uses these labels to group neurons into type blocks.
    cell_type_column : str, default "cell_type"
        Column name in ``neuron_annotations`` to use as the type label.
    neuron_id_column : str, default "root_id"
        Column name in ``neuron_annotations`` that identifies neurons and
        matches the ids used in ``connections_df``.
    detail_level : {"type", "neuron"}, default "type"
        If ``"neuron"``, return the full neuron-by-neuron matrix (and
        boundaries mapping types to index ranges). If ``"type"``, aggregate
        weights into a type-by-type matrix.

    Returns
    -------
    tuple
        When ``detail_level == 'neuron'``:
            (relative_matrix, type_boundaries, ordered_neurons, id_to_type)
            - ``relative_matrix``: DataFrame (n_neurons x n_neurons) of relative
              weights aligned to ``ordered_neurons``.
            - ``type_boundaries``: dict mapping type -> (start_index, end_index)
              giving the slices of ``ordered_neurons`` belonging to each type.
            - ``ordered_neurons``: list of neuron ids in the order used for rows
              and columns.
            - ``id_to_type``: dict mapping neuron id -> type label (or None).

        When ``detail_level == 'type'``:
            (type_matrix, type_level_boundaries, present_types, id_to_type)
            - ``type_matrix``: DataFrame aggregated by type (type x type).
            - ``type_level_boundaries``: boundaries for type-level matrix (each
              type maps to a 1-row/1-col block).
            - ``present_types``: list of types present in the returned matrix.
            - ``id_to_type``: mapping as above.

    Notes
    -----
    - If ``weightRelative`` is absent from ``connections_df`` the function
      computes it by calling :func:`calculate_relative_weights`. A missing
      ``roi`` column will be filled with the value ``'all'`` before computing
      relative weights.
    - The function preserves the order of types found in ``neuron_annotations``
      (first occurrence order) and falls back to sorting neuron ids alphabetically
      within each type.

    Example
    -------
    >>> type_matrix, boundaries, types, id_map = create_nested_connectivity_matrix(
    ...     connections_df, neuron_annotations, detail_level='type')
    """

    if detail_level not in {"type", "neuron"}:
        raise ValueError("detail_level must be either 'type' or 'neuron'")

    
    data = connections_df.copy()
    if "weightRelative" not in data.columns:
        
        if "roi" not in data.columns:
            data["roi"] = "all"
        data = calculate_relative_weights(data)

    neuron_ids = set(map(str, data["type.from"])) | set(map(str, data["type.to"]))

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

    type_series = relevant_annotations[cell_type_column].apply(_prepare_type)
    relevant_annotations["cell_type"] = type_series

    id_to_type = {
        str(k): (str(v) if pd.notna(v) else None)
        for k, v in zip(
            relevant_annotations[neuron_id_column].astype(str),
            relevant_annotations["cell_type"],
        )
    }

    
    seen = set()
    present_types = []
    for c in relevant_annotations["cell_type"]:
        if pd.notna(c) and c not in seen:
            seen.add(c)
            present_types.append(c)

    resolved_order = present_types

    ordered_neurons: list[str] = []
    type_boundaries: dict[str, tuple[int, int]] = {}
    current_pos = 0

    for cell_type in resolved_order:
        type_neurons = [nid for nid in neuron_ids if id_to_type.get(nid) == cell_type]
        if type_neurons:
            type_boundaries[cell_type] = (current_pos, current_pos + len(type_neurons))
            ordered_neurons.extend(sorted(type_neurons))
            current_pos += len(type_neurons)

    n_neurons = len(ordered_neurons)
    connectivity_matrix = pd.DataFrame(
        np.zeros((n_neurons, n_neurons)),
        index=pd.Index(ordered_neurons),
        columns=pd.Index(ordered_neurons),
    )

    
    for _, row in data.iterrows():
        from_id = str(row["type.from"])
        to_id = str(row["type.to"])
        weight = row.get("weightRelative", row.get("weight", 0))
        try:
            weight_val = float(weight)
        except Exception:
            weight_val = 0.0
        if (
            from_id in connectivity_matrix.index
            and to_id in connectivity_matrix.columns
        ):
            current_cell = connectivity_matrix.loc[from_id, to_id]
            # coerce the existing cell value to numeric, default to 0.0 on failure
            current_val = pd.to_numeric(current_cell, errors="coerce")
            if pd.isna(current_val):
                current_val = 0.0
            else:
                current_val = float(current_val)
            connectivity_matrix.loc[from_id, to_id] = current_val + weight_val

    
    relative_matrix = connectivity_matrix.copy()

    if detail_level == "neuron":
        return relative_matrix, type_boundaries, ordered_neurons, id_to_type

    present_types = [ctype for ctype in resolved_order if ctype in type_boundaries]
    if not present_types:
        return pd.DataFrame(), {}, [], id_to_type

    type_to_neurons = {}
    for cell_type in present_types:
        type_to_neurons[cell_type] = [
            nid for nid in ordered_neurons if id_to_type.get(nid) == cell_type
        ]

    type_matrix = pd.DataFrame(0.0, index=present_types, columns=present_types)

    for from_type in present_types:
        from_neurons = type_to_neurons[from_type]
        if not from_neurons:
            continue

        from_block = relative_matrix.loc[from_neurons]
        for to_type in present_types:
            to_neurons = type_to_neurons[to_type]
            if not to_neurons:
                continue
            type_matrix.loc[from_type, to_type] = (
                from_block.loc[:, to_neurons].sum().sum()
            )

    type_level_boundaries: dict[str, tuple[int, int]] = {}
    current_pos = 0
    for ctype in present_types:
        type_level_boundaries[ctype] = (current_pos, current_pos + 1)
        current_pos += 1

    return type_matrix, type_level_boundaries, present_types, id_to_type






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

    filtered_matrix = pd.DataFrame(matrix.loc[kept_neurons, kept_neurons]).copy()
    filtered_matrix.index = pd.Index(kept_neurons)
    filtered_matrix.columns = pd.Index(kept_neurons)

    return filtered_matrix, new_boundaries, kept_neurons, removed_types






def plot_nested_connectivity_matrix(
    relative_matrix: pd.DataFrame,
    type_boundaries: dict[str, tuple[int, int]],
    ordered_neurons: list[str],
    neuron_to_type: dict[str, str],
    figsize: tuple[int, int] = (16, 14),
    show_neuron_labels: bool = False,
    output_path: str | None = None,
    normalization: str = "global",
    scale_mode: str = "auto",
    min_neurons_for_plot: int = 4,
    column_boundaries: dict[str, tuple[int, int]] | None = None,
    column_order: list[str] | None = None,
    vmin_percentile: float = 0.0,
    vmax_percentile: float = 100.0,
):
    """Plot a nested connectivity matrix with type boundaries and labelling.

    The function accepts either a square neuron-by-neuron matrix aligned to
    ``ordered_neurons`` or a rectangular matrix (if columns are a different
    ordering). It draws a heatmap of relative weights with optional
    seaborn-based rendering, type-level separators and optional neuron labels.

    Parameters
    ----------
    relative_matrix : pd.DataFrame
        Square (or rectangular) matrix of weights. Rows are presynaptic,
        columns are postsynaptic. Indices/columns must be aligned with
        ``ordered_neurons`` when plotting neuron-level matrices.
    type_boundaries : dict
        Mapping type -> (start_index, end_index) defining blocks in the
        matrix corresponding to cell types. For type-level matrices each
        block typically has size 1.
    ordered_neurons : list[str]
        Neuron id ordering used for rows/columns of a neuron-level matrix.
    neuron_to_type : dict
        Mapping neuron id -> type label. Used to compute labels and to filter
        by cluster size when requested.
    figsize : tuple, optional
        Matplotlib figure size in inches.
    show_neuron_labels : bool, default False
        Show individual neuron ids on the x/y axes. For large matrices labels
        are downsampled (see ``NEURON_LABEL_STEP_DIVISOR``).
    output_path : str or None
        If provided, the plot will be saved to this path as a PNG.
    (ploting is seaborn-only; the `use_seaborn` flag was removed)
    normalization : {"global", "none"}, default "global"
        Only accepted to preserve compatibility; current implementation uses
        the matrix values directly (set to "none" to avoid internal scaling).
    min_neurons_for_plot : int, default 4
        When plotting neuron-level matrices, clusters (types) smaller than
        this threshold are filtered out to avoid visual clutter.
    column_boundaries, column_order : optional
        Advanced options for plotting rectangular matrices with different
        column groupings or custom column orders.
    vmin_percentile, vmax_percentile : float
        Percentiles used to set the colormap minimum and maximum (0-100).

    Returns
    -------
    (fig, ax)
        Matplotlib figure and axes objects for the drawn heatmap.

    Notes
    -----
    - The function will draw thick black separators around the whole matrix
      and thinner separators between type blocks.
    - When ``use_seaborn=True`` zero values are masked for improved contrast.
    """

    normalization = normalization.lower()
    if normalization not in {"global", "none"}:
        raise ValueError("normalization must be either 'global' or 'none'")

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

    cbar_label = "Relative Weight"

    if rectangular_layout:
        
        plot_matrix = relative_matrix.loc[row_order, resolved_column_order]
        fig, ax = plt.subplots(figsize=figsize)
        mask = plot_matrix.values == 0
        sns.heatmap(
            plot_matrix.values,
            cmap=purple_cmap,
            vmin=0,
            vmax=float(np.nanmax(plot_matrix.values)) if plot_matrix.size else 1.0,
            square=False,
            cbar=False,
            xticklabels=False,
            yticklabels=False,
            ax=ax,
            linewidths=0,
            rasterized=True,
            mask=mask,
        )
        # create matplotlib colorbar from seaborn mappable
        mappable = ax.collections[0] if ax.collections else None
        if mappable is not None:
            cbar = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(cbar_label, fontsize=COLORBAR_LABEL_FONTSIZE, fontweight="bold")

        ax.set_xlabel(
            "postsynaptic neuron", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold"
        )
        ax.set_ylabel(
            "presynaptic neuron", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold"
        )
        plt.tight_layout()
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            plt.savefig(
                output_path, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white"
            )
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

    # seaborn rendering for neuron-level matrices
    fig, ax = plt.subplots(figsize=figsize)
    mask = plot_matrix.values == 0
    sns.heatmap(
        plot_matrix.values,
        cmap=purple_cmap,
        vmin=color_min,
        vmax=color_max,
        square=True,
        cbar=False,
        xticklabels=False,
        yticklabels=False,
        ax=ax,
        linewidths=0,
        rasterized=True,
        mask=mask,
    )

    for cell_type, (start_pos, end_pos) in plot_boundaries.items():
        width = height = end_pos - start_pos
        rect = Rectangle(
            (start_pos, start_pos),
            width,
            height,
            linewidth=2,
            edgecolor="black",
            facecolor="none",
        )
        ax.add_patch(rect)

    ax.set_xlabel("Postsynaptic Neuron", fontsize=AXIS_LABEL_FONTSIZE, labelpad=40)
    ax.set_ylabel("Presynaptic Neuron", fontsize=AXIS_LABEL_FONTSIZE, labelpad=40)
    ax.set_xlim(-0.5, len(plot_matrix) + 0.5)
    ax.set_ylim(-0.5, len(plot_matrix) + 0.5)

    plt.tight_layout()

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")

    # create matplotlib colorbar from seaborn mappable
    mappable = ax.collections[0] if ax.collections else None
    if mappable is not None:
        cbar = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label.lower(), fontsize=COLORBAR_LABEL_FONTSIZE, fontweight="bold")
        cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

    type_names = list(plot_boundaries.keys())
    type_boundary_positions = []

    for cell_type in type_names:
        start_pos, end_pos = plot_boundaries[cell_type]
        if start_pos > 0:
            boundary_position = start_pos - 0.5
            type_boundary_positions.append(boundary_position)
            ax.axhline(boundary_position, color="white", linewidth=2, alpha=1.0)
            ax.axvline(boundary_position, color="white", linewidth=2, alpha=1.0)

    matrix_size = len(plot_matrix)
    ax.axhline(-0.5, color="black", linewidth=3, alpha=1.0)
    ax.axhline(matrix_size - 0.5, color="black", linewidth=3, alpha=1.0)
    ax.axvline(-0.5, color="black", linewidth=3, alpha=1.0)
    ax.axvline(matrix_size - 0.5, color="black", linewidth=3, alpha=1.0)

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

    # create colorbar from seaborn mappable (if available)
    mappable = ax.collections[0] if ax.collections else None
    if mappable is not None:
        cbar = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
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

    result_df = calculate_relative_weights(result_df)

    result_df["from"] = result_df["type.from"]
    result_df["to"] = result_df["type.to"]
    result_df["name.from"] = result_df["type.from"]
    result_df["name.to"] = result_df["type.to"]

    return result_df.reset_index(drop=True)
