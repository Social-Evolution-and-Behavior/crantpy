# -*- coding: utf-8 -*-
"""
Pure functions for ordering neurons along a connectivity matrix axis.

An axis is a sequence of cell type blocks, each holding a group of neurons.
:class:`NeuronOrder` describes both levels: ``types`` orders the blocks,
``within`` orders the neurons inside each one. Neither can move a neuron across
a block boundary.

Pipeline: :func:`resolve_relevant_annotations` -> :func:`resolve_type_order` ->
:func:`build_ordered_neurons`, glued by :func:`build_axis_ordering`.

Dataset-specific rules are data, not logic: the CRANT ellipsoid-body column
labels live in :mod:`crantpy.utils.config`; this module binds them into
:data:`DEFAULT_WITHIN_TYPE_ORDER`.

Examples
--------
>>> NeuronOrder(types="size", within="id")
>>> NeuronOrder(within={**DEFAULT_WITHIN_TYPE_ORDER, "PEN": ["R1", "L1"]})
"""

from __future__ import annotations

import logging
import re
from collections.abc import (
    Callable,
    Iterable,
    KeysView,
    Mapping,
    Set as AbstractSet,
)
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd

from crantpy.utils.config import EB_COLUMN_LABELS, EB_COLUMNAR_CELL_TYPES

__all__ = [
    "AxisOrdering",
    "COLUMN_LABEL_PATTERN",
    "ColumnOrderRule",
    "DEFAULT_ORDER",
    "DEFAULT_WITHIN_TYPE_ORDER",
    "EB_COLUMN_ORDER",
    "NeuronOrder",
    "NeuronOrderLike",
    "ResolvedAnnotations",
    "TypeRule",
    "TypeSorter",
    "WithinTypeOrder",
    "WithinTypeRule",
    "WithinTypeSorter",
    "as_neuron_order",
    "build_axis_ordering",
    "build_ordered_neurons",
    "resolve_relevant_annotations",
    "resolve_type_order",
]

logger = logging.getLogger(__name__)

#: Matches a trailing ``L``/``R`` plus digits, e.g. ``"EPG/PEG_R1"`` -> ``"R1"``.
COLUMN_LABEL_PATTERN = re.compile(r"([LR]\d+)\s*$")

# ---------------------------------------------------------------------------
# Neuron ID normalization
# ---------------------------------------------------------------------------


def _is_missing_scalar(value: Any) -> bool:
    """Return True when *value* is a scalar null (None, NaN, NaT)."""
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def _stringify_id_value(value: Any) -> str:
    """Normalize a neuron ID or label to a string; ``1``, ``1.0``, ``"1"`` -> ``"1"``."""
    if isinstance(value, (bool, np.bool_)):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value_float = float(value)
        if np.isfinite(value_float) and value_float.is_integer():
            return str(int(value_float))
    return str(value)


def _stringify_id_axis(values: Iterable[Any], axis_name: str) -> pd.Index:
    """Normalize an axis of neuron IDs, rejecting nulls."""
    ids: list[str] = []
    for value in values:
        if _is_missing_scalar(value):
            raise ValueError(f"{axis_name} contains null neuron IDs")
        ids.append(_stringify_id_value(value))
    return pd.Index(ids)


def _normalize_id_values(values: Iterable[Any]) -> list[str]:
    """Normalize neuron IDs, silently dropping nulls."""
    return [
        _stringify_id_value(value) for value in values if not _is_missing_scalar(value)
    ]


def _find_duplicates(values: Iterable[str]) -> list[str]:
    """Return the values that appear more than once, in first-repeat order."""
    seen: set[str] = set()
    duplicates: list[str] = []
    duplicate_seen: set[str] = set()
    for value in values:
        if value in seen and value not in duplicate_seen:
            duplicates.append(value)
            duplicate_seen.add(value)
        seen.add(value)
    return duplicates


# ---------------------------------------------------------------------------
# Within-type ordering rules
# ---------------------------------------------------------------------------


def _reject_unordered(value: Any, what: str) -> Any:
    """Reject rule iterables that cannot work: sets and bytes.

    Sets iterate in an order that varies between interpreter runs. Key views
    (``dict.keys()``, and so ``matrix.type_boundaries.keys()``) are ``Set``
    instances but iterate in insertion order, so they are allowed. Iterating
    bytes yields integers, so no string label would ever match.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(
            f"{what} must be an iterable of string labels, not "
            f"{type(value).__name__}; iterating {value!r} yields integers"
        )
    if isinstance(value, AbstractSet) and not isinstance(value, KeysView):
        raise TypeError(
            f"{what} must be an ordered iterable; set and frozenset are not "
            "supported because their order is not stable"
        )
    return value


def _reject_bytes_like(value: Any, what: str) -> Any:
    """Reject bytes-like labels, which could only silently never match."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(
            f"{what} must be a string, not {type(value).__name__}: "
            f"{value!r} can never match a label"
        )
    return value


@dataclass(frozen=True)
class ColumnOrderRule:
    """Order neurons within a cell type by a ranked column label.

    A neuron's label comes from the first of ``label_columns`` that yields a
    label both matching ``pattern`` and present in ``order``; a column whose
    value matches the pattern but is not in ``order`` is skipped and the next
    column tried. Neurons with no resolvable label keep annotation row order
    and land after the ranked ones, each logged as a warning.

    Parameters
    ----------
    order : Iterable[str]
        Column labels, first to last, e.g. ``["R1", "L8", "R2", ...]``.
    label_columns : Iterable[str], default ("cell_instance", "cell_subtype")
        Annotation columns to search, in priority order.
    pattern : re.Pattern[str] or str, optional
        Regex whose first group is the label; a plain string is compiled.
        Defaults to a trailing ``L``/``R`` plus digits, so ``"EPG/PEG_R1"``
        gives ``"R1"``.

    Examples
    --------
    >>> ColumnOrderRule(order=["R1", "L1", "R2", "L2"])
    """

    order: tuple[str, ...]
    label_columns: tuple[str, ...] = ("cell_instance", "cell_subtype")
    pattern: re.Pattern[str] = COLUMN_LABEL_PATTERN

    def __post_init__(self) -> None:
        for field_name, value in (
            ("order", self.order),
            ("label_columns", self.label_columns),
        ):
            if isinstance(value, str):
                # tuple("R1") == ("R", "1"): the label would silently never rank.
                raise TypeError(
                    f"ColumnOrderRule.{field_name} takes an iterable of labels, "
                    f"not a bare string; wrap it: [{value!r}]"
                )
            _reject_unordered(value, f"ColumnOrderRule.{field_name}")
        if isinstance(self.pattern, str):
            object.__setattr__(self, "pattern", re.compile(self.pattern))
        if not isinstance(self.pattern, re.Pattern) or not isinstance(
            self.pattern.pattern, str
        ):
            raise TypeError(
                "ColumnOrderRule.pattern must be a str regex or a compiled "
                f"str pattern (a bytes pattern cannot match); got {self.pattern!r}"
            )
        if self.pattern.groups < 1:
            raise ValueError(
                "ColumnOrderRule.pattern needs a capture group for the label; "
                f"{self.pattern.pattern!r} has none"
            )
        object.__setattr__(self, "order", tuple(self.order))
        object.__setattr__(self, "label_columns", tuple(self.label_columns))
        for field_name in ("order", "label_columns"):
            for entry in getattr(self, field_name):
                if not isinstance(entry, str):
                    raise TypeError(
                        f"ColumnOrderRule.{field_name} entries must be "
                        f"strings; got {type(entry).__name__}: {entry!r}"
                    )


#: ``(type_name, type_rows, neuron_id_column) -> ordered neuron IDs``. The
#: frame arrives with its ID and cell type columns already normalized by
#: :func:`_stringify_id_value`, and the returned IDs must be that same set.
WithinTypeSorter = Callable[[str, pd.DataFrame, str], list[Any]]

#: How to order the neurons inside a single cell type block.
WithinTypeRule = (
    Literal["annotation", "id"] | ColumnOrderRule | Iterable[str] | WithinTypeSorter
)

#: ``NeuronOrder.within``: one rule for every type, a ``{cell_type: rule}``
#: mapping overlaid on :data:`DEFAULT_WITHIN_TYPE_ORDER` (so a type listed in
#: neither keeps annotation order), or ``None`` for annotation order throughout.
WithinTypeOrder = WithinTypeRule | Mapping[Any, WithinTypeRule] | None

#: ``(type_names, typed_annotations, type_col) -> ordered type names``.
TypeSorter = Callable[[list[str], pd.DataFrame, str], list[Any]]

#: ``NeuronOrder.types``: ``"label"``, ``"size"``, a sequence of preferred
#: labels, or a :data:`TypeSorter` callable.
TypeRule = Literal["label", "size"] | Iterable[Any] | TypeSorter | None


def _order_by_annotation(
    type_name: str, type_rows: pd.DataFrame, neuron_id_column: str
) -> list[Any]:
    """Keep the resolved annotation row order."""
    return type_rows[neuron_id_column].tolist()


def _order_by_id(
    type_name: str, type_rows: pd.DataFrame, neuron_id_column: str
) -> list[Any]:
    """Sort by neuron ID, numerically when every ID is a non-negative integer.

    IDs are normalized with :func:`_stringify_id_value` first, so a float column
    holding ``10.0`` sorts as ``10`` rather than lexicographically as ``"10.0"``.
    Anything else -- a fractional value, a negative sign, a non-numeric label --
    falls back to a plain string sort, since the numeric fast path keys on the
    normalized form being all decimal digits.
    """
    ids = [_stringify_id_value(value) for value in type_rows[neuron_id_column]]
    if ids and all(value.isdecimal() for value in ids):
        return sorted(ids, key=int)
    return sorted(ids)


def _extract_ranked_label(
    row: pd.Series,
    rule: ColumnOrderRule,
    neuron_id_column: str,
    rank: Mapping[str, int],
) -> str | None:
    for column in rule.label_columns:
        if column not in row.index:
            continue

        value = row[column]
        if _is_missing_scalar(value):
            continue

        match = rule.pattern.search(str(value).strip())
        if not match:
            continue

        label = match.group(1)
        if label in rank:
            return label

    logger.warning(
        "Could not resolve a ranked column label for neuron %s from columns %s; "
        "it will be ordered after ranked neurons",
        row.get(neuron_id_column, "<unknown>"),
        rule.label_columns,
    )
    return None


def _order_by_column_rule(rule: ColumnOrderRule) -> WithinTypeSorter:
    """Build a sorter that ranks neurons by *rule*'s column labels."""

    def sorter(
        type_name: str, type_rows: pd.DataFrame, neuron_id_column: str
    ) -> list[Any]:
        if type_rows.empty:
            # DataFrame.apply(axis=1) on an empty frame still calls the function
            # once, on an all-NaN dummy row, which would warn about "neuron nan".
            return []

        rank = {label: i for i, label in enumerate(rule.order)}
        sorted_rows = type_rows.copy()
        sorted_rows["__group_order__"] = np.arange(len(sorted_rows))
        sorted_rows["__column_label__"] = sorted_rows.apply(
            _extract_ranked_label,
            axis=1,
            rule=rule,
            neuron_id_column=neuron_id_column,
            rank=rank,
        )
        sorted_rows["__column_rank__"] = sorted_rows["__column_label__"].map(rank)
        sorted_rows["__has_column_rank__"] = sorted_rows["__column_rank__"].notna()
        sorted_rows["__column_rank__"] = sorted_rows["__column_rank__"].fillna(np.inf)
        sorted_rows = sorted_rows.sort_values(
            by=["__has_column_rank__", "__column_rank__", "__group_order__"],
            ascending=[False, True, True],
        )

        return sorted_rows[neuron_id_column].tolist()

    return sorter


def _resolve_within_type_rule(rule: WithinTypeRule | None) -> WithinTypeSorter:
    """Normalize one within-type rule into a sorter callable."""
    if rule is None:
        return _order_by_annotation
    if isinstance(rule, str):
        if rule == "annotation":
            return _order_by_annotation
        if rule == "id":
            return _order_by_id
        raise ValueError(
            f"order.within string must be 'annotation' or 'id', got {rule!r}"
        )
    if isinstance(rule, ColumnOrderRule):
        return _order_by_column_rule(rule)
    if callable(rule):
        return rule
    if isinstance(rule, Mapping):
        raise TypeError(
            "a per-cell-type mapping belongs at the top of order.within, not "
            f"nested inside it; got {dict(rule)!r} as one type's rule"
        )
    _reject_unordered(rule, "order.within")
    if isinstance(rule, Iterable):
        return _order_by_column_rule(ColumnOrderRule(order=rule))
    raise TypeError(
        "order.within rules must be 'annotation', 'id', a ColumnOrderRule, "
        f"a sequence of column labels, or a callable; got {type(rule).__name__}"
    )


def _resolve_within_type_order(
    within: WithinTypeOrder,
) -> Callable[[str], WithinTypeSorter]:
    """Return a lookup from cell type name to the sorter that orders it.

    A per-type mapping is *overlaid* on :data:`DEFAULT_WITHIN_TYPE_ORDER`, so
    adding a rule for one type leaves the built-in rules for the others in
    place. Override a built-in by naming it (``{"EPG/PEG": "annotation"}``);
    drop them all with ``within=None``.
    """
    if isinstance(within, Mapping):
        merged = {**DEFAULT_WITHIN_TYPE_ORDER, **within}
        by_type: dict[str, WithinTypeSorter] = {}
        seen: dict[str, Any] = {}
        for name, rule in merged.items():
            _reject_bytes_like(name, "order.within cell type keys")
            key = _stringify_id_value(name)
            if key in seen:
                raise ValueError(
                    "order.within has two keys naming the same cell type "
                    f"{key!r}: {seen[key]!r} and {name!r}"
                )
            seen[key] = name
            by_type[key] = _resolve_within_type_rule(rule)
        return lambda type_name: by_type.get(type_name, _order_by_annotation)

    sorter = _resolve_within_type_rule(within)
    return lambda type_name: sorter


def _apply_within_type_sorter(
    sorter: WithinTypeSorter,
    type_name: str,
    type_rows: pd.DataFrame,
    neuron_id_column: str,
) -> list[str]:
    """Run *sorter* and check it returned exactly the type's neurons."""
    expected = [str(value) for value in type_rows[neuron_id_column]]
    ordered = [str(value) for value in sorter(type_name, type_rows, neuron_id_column)]

    if sorted(ordered) != sorted(expected):
        missing = sorted(set(expected) - set(ordered))
        unexpected = sorted(set(ordered) - set(expected))
        raise ValueError(
            f"order.within for cell type {type_name!r} must return each of "
            f"its {len(expected)} neuron ID(s) exactly once, but returned "
            f"{len(ordered)}; missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    return ordered


# ---------------------------------------------------------------------------
# CRANT defaults
# ---------------------------------------------------------------------------

#: Ellipsoid-body column order, applied to CRANT's columnar cell types.
EB_COLUMN_ORDER = ColumnOrderRule(
    order=EB_COLUMN_LABELS,
    # cell_instance is a forward-compatible placeholder if explicit instance
    # annotations are added; current annotations fall back to cell_subtype.
    label_columns=("cell_instance", "cell_subtype"),
)

#: Default ``NeuronOrder.within``: only CRANT's columnar types get a rule.
#: A ``within`` mapping is overlaid on this, so ``{"PEN": [...]}`` adds a rule
#: and keeps the rest. ``within=None`` switches them all off.
DEFAULT_WITHIN_TYPE_ORDER: Mapping[str, WithinTypeRule] = MappingProxyType(
    {cell_type: EB_COLUMN_ORDER for cell_type in EB_COLUMNAR_CELL_TYPES}
)


# ---------------------------------------------------------------------------
# The single ordering spec
# ---------------------------------------------------------------------------


def _snapshot(rule: Any) -> Any:
    """Recursively freeze an order rule so it is deterministic and reusable."""
    if rule is None or isinstance(rule, str):
        return rule
    if callable(rule):
        return rule
    if isinstance(rule, Mapping):
        for name in rule:
            _reject_bytes_like(name, "order rule mapping keys")
        return {name: _snapshot(nested_rule) for name, nested_rule in rule.items()}
    _reject_unordered(rule, "order rules")
    if isinstance(rule, Iterable):
        entries = tuple(rule)
        for entry in entries:
            _reject_bytes_like(entry, "order rule entries")
            if isinstance(entry, Iterable) and not isinstance(entry, str):
                # A nested list/tuple would stringify to "['a']" and silently
                # never match a label.
                raise TypeError(
                    "order rule entries must be scalar labels; got "
                    f"{type(entry).__name__}: {entry!r}"
                )
        return entries
    return rule


@dataclass(frozen=True)
class NeuronOrder:
    """How one matrix axis is ordered, at both of its levels.

    ``types`` orders the cell type blocks, ``within`` orders the neurons inside
    each one. Neither can move a neuron across a block boundary.

    Parameters
    ----------
    types : {"label", "size"}, sequence or callable, default "label"
        ``"label"`` sorts by ``(ALPHA_PREFIX, numeric_suffix, label)``;
        ``"size"`` puts the largest blocks first; a sequence pulls its entries
        to the front -- each matching one type exactly, unnamed types
        following in ``"label"`` order -- so ``types=list(other.type_boundaries)``
        reproduces another matrix's block order; a callable takes
        ``(type_names, typed_annotations, type_col)`` and returns them
        reordered.
    within : rule or {cell_type: rule} mapping, default DEFAULT_WITHIN_TYPE_ORDER
        ``"annotation"``, ``"id"``, a :class:`ColumnOrderRule`, a label
        sequence, a ``(type_name, type_rows, id_column)`` callable, or a
        per-type mapping. A mapping is overlaid on
        :data:`DEFAULT_WITHIN_TYPE_ORDER`, so it adds to the built-in rules
        rather than replacing them; a non-mapping rule applies to every type.
        ``None`` means annotation row order everywhere.

    Examples
    --------
    >>> NeuronOrder(types=["EPG/PEG", "ER"], within="id")
    """

    types: TypeRule = "label"
    within: WithinTypeOrder = DEFAULT_WITHIN_TYPE_ORDER

    def __post_init__(self) -> None:
        object.__setattr__(self, "types", _snapshot(self.types))
        object.__setattr__(self, "within", _snapshot(self.within))


#: The default order. A constructor given no ``order`` builds a fresh equal
#: instance rather than sharing this one.
DEFAULT_ORDER = NeuronOrder()

#: What ``order=`` accepts: a :class:`NeuronOrder`, a
#: ``{"types": ..., "within": ...}`` mapping, a bare sequence (types-only
#: shorthand), or ``None``.
NeuronOrderLike = NeuronOrder | Mapping[str, Any] | Iterable[Any] | None


def as_neuron_order(spec: NeuronOrderLike) -> NeuronOrder:
    """Coerce an ``order=`` argument into a :class:`NeuronOrder`.

    ``None`` builds a fresh default instance rather than handing out a shared
    one, so editing one order's rules can never leak into another build.
    """
    if spec is None:
        return NeuronOrder()
    if isinstance(spec, NeuronOrder):
        return spec
    if isinstance(spec, Mapping):
        unknown = sorted(set(spec) - {"types", "within"})
        if unknown:
            raise ValueError(
                f"order mapping accepts only 'types' and 'within', got {unknown}. "
                "To set a per-cell-type rule, nest it: "
                "order={'within': {'EPG/PEG': ...}}"
            )
        default = NeuronOrder()
        return NeuronOrder(
            types=spec.get("types", default.types),
            within=spec.get("within", default.within),
        )
    if isinstance(spec, str):
        raise TypeError(
            f"order={spec!r} is ambiguous: say which level it applies to, e.g. "
            f"order={{'types': {spec!r}}} or order={{'within': {spec!r}}}"
        )
    if isinstance(spec, Iterable):
        return NeuronOrder(types=spec)
    raise TypeError(
        "order must be a NeuronOrder, a {'types': ..., 'within': ...} mapping, "
        f"a sequence of cell type labels, or None; got {type(spec).__name__}"
    )


# ---------------------------------------------------------------------------
# Cell type block ordering
# ---------------------------------------------------------------------------


def _parse_type_label(label: str) -> tuple[str, float, str]:
    match = re.match(r"^([A-Za-z]+)(\d*)(.*)$", label.strip())
    if not match:
        return (label.upper(), float("inf"), "")
    prefix, number, suffix = match.groups()
    return (prefix.upper(), int(number) if number else float("inf"), suffix.upper())


def _sort_cell_types(
    types: list[Any], preferred: Iterable[Any] | None = None
) -> list[str]:
    """Order cell type labels, honouring *preferred* where it applies.

    *preferred* must be an ordered iterable; sets are rejected.

    Without *preferred*: ``(ALPHA_PREFIX, numeric_suffix, label)``, un-numbered
    labels last within their prefix -- ``ER1, ER2, ER10, ER``.

    Each *preferred* entry matches one type exactly; unnamed types follow in
    generic order.
    """
    _reject_unordered(preferred, "_sort_cell_types(preferred=...)")
    unique = {_stringify_id_value(t) for t in types if not _is_missing_scalar(t)}

    parsed = {label: _parse_type_label(label) for label in unique}
    generic_order = sorted(unique, key=lambda x: (*parsed[x][:2], x))

    if not preferred:
        return generic_order

    result: list[str] = []
    remaining = set(generic_order)

    for preferred_value in preferred:
        if _is_missing_scalar(preferred_value):
            continue
        _reject_bytes_like(preferred_value, "preferred cell type entries")
        wanted = _stringify_id_value(preferred_value)
        for match in [t for t in generic_order if t == wanted]:
            if match in remaining:
                result.append(match)
                remaining.remove(match)

    result.extend([t for t in generic_order if t in remaining])
    return result


# ---------------------------------------------------------------------------
# Annotation resolution and axis assembly
# ---------------------------------------------------------------------------


class ResolvedAnnotations(NamedTuple):
    """Result of resolving neuron annotations against one axis' neuron IDs."""

    relevant: pd.DataFrame
    typed: pd.DataFrame
    id_map: dict[str, Any]
    untyped_ids: list[str]
    missing_ids: list[str]


@dataclass(frozen=True)
class AxisOrdering:
    """Ordered neurons and type metadata for one matrix axis.

    ``ordered_neurons`` is the typed neurons -- grouped into the contiguous
    blocks in ``type_boundaries`` -- followed by the untyped ones. Only typed
    neurons appear in ``neuron_to_type``.

    The untyped tail is neurons whose annotation row carries no cell type,
    then neurons with no annotation row at all, each group sorted by neuron ID
    as a string (so ``"30"`` precedes ``"7"``).
    """

    ordered_neurons: tuple[str, ...]
    type_boundaries: dict[str, tuple[int, int]]
    neuron_to_type: dict[str, Any]


def resolve_relevant_annotations(
    matrix_ids: set[str],
    annotations: pd.DataFrame,
    id_col: str,
    type_col: str,
) -> ResolvedAnnotations:
    """Narrow *annotations* to *matrix_ids*, de-duplicating by neuron.

    With several rows per neuron, the first row carrying a cell type wins, ties
    going to the earliest row. The caller's index is ignored, so a duplicate or
    non-unique index is fine.
    """
    valid_ids = annotations[id_col].map(lambda value: not _is_missing_scalar(value))
    # reset_index keeps the label-based .loc below well defined when the caller's
    # annotations carry a duplicate index (e.g. from pd.concat or set_index).
    normalized = annotations.loc[valid_ids].reset_index(drop=True)
    ann_ids = normalized[id_col].map(_stringify_id_value)
    relevant = normalized[ann_ids.isin(matrix_ids)].copy()
    relevant = relevant.assign(
        **{
            id_col: ann_ids.loc[relevant.index],
            "__has_type__": relevant[type_col].notna(),
            "__row_order__": np.arange(len(relevant)),
        }
    )
    relevant["__first_seen__"] = relevant.groupby(id_col)["__row_order__"].transform(
        "min"
    )
    relevant = relevant.sort_values(
        by=["__first_seen__", "__has_type__", "__row_order__"],
        ascending=[True, False, True],
    ).drop_duplicates(subset=[id_col], keep="first")

    relevant[type_col] = relevant[type_col].map(
        lambda value: (
            _stringify_id_value(value) if not _is_missing_scalar(value) else value
        )
    )

    typed = relevant[relevant[type_col].notna()].copy()
    id_map = dict(zip(typed[id_col], typed[type_col]))
    typed_ids = set(typed[id_col])
    untyped_ids = sorted(set(relevant[id_col]) - typed_ids)
    missing_ids = sorted(matrix_ids - set(relevant[id_col]))

    return ResolvedAnnotations(relevant, typed, id_map, untyped_ids, missing_ids)


def _order_types_by_label(
    type_names: list[str], typed_annotations: pd.DataFrame, type_col: str
) -> list[str]:
    """Generic label-aware sort: ``(ALPHA_PREFIX, numeric_suffix, label)``."""
    return _sort_cell_types(type_names)


def _order_types_by_size(
    type_names: list[str], typed_annotations: pd.DataFrame, type_col: str
) -> list[str]:
    """Largest blocks first, ties broken by label order."""
    counts = typed_annotations[type_col].map(_stringify_id_value).value_counts()
    by_label = _sort_cell_types(type_names)
    return sorted(
        by_label, key=lambda name: (-int(counts.get(name, 0)), by_label.index(name))
    )


def _order_types_by_preferred(preferred: Iterable[Any]) -> TypeSorter:
    """Build a sorter that pulls *preferred* to the front, rest in label order."""
    _reject_unordered(preferred, "_order_types_by_preferred(preferred=...)")
    preferred = list(preferred)

    def sorter(
        type_names: list[str], typed_annotations: pd.DataFrame, type_col: str
    ) -> list[str]:
        return _sort_cell_types(type_names, preferred=preferred)

    return sorter


def _resolve_type_rule(rule: TypeRule) -> TypeSorter:
    """Normalize a block-ordering rule into a sorter callable."""
    if rule is None:
        return _order_types_by_label
    if isinstance(rule, str):
        if rule == "label":
            return _order_types_by_label
        if rule == "size":
            return _order_types_by_size
        raise ValueError(f"order.types string must be 'label' or 'size', got {rule!r}")
    if callable(rule):
        return rule
    if isinstance(rule, Mapping):
        raise TypeError(
            "order.types takes a flat sequence of cell type labels, not a "
            f"mapping; got {dict(rule)!r}"
        )
    _reject_unordered(rule, "order.types")
    if isinstance(rule, Iterable):
        return _order_types_by_preferred(rule)
    raise TypeError(
        "order.types must be 'label', 'size', a sequence of cell type labels, "
        f"or a callable; got {type(rule).__name__}"
    )


def _apply_type_sorter(
    sorter: TypeSorter,
    type_names: list[str],
    typed_annotations: pd.DataFrame,
    type_col: str,
) -> list[str]:
    """Run *sorter* and check it returned exactly the present cell types."""
    expected = list(type_names)
    ordered = [
        str(value) for value in sorter(list(type_names), typed_annotations, type_col)
    ]

    if sorted(ordered) != sorted(expected):
        missing = sorted(set(expected) - set(ordered))
        unexpected = sorted(set(ordered) - set(expected))
        raise ValueError(
            f"order.types must return each of the {len(expected)} present cell "
            f"type(s) exactly once, but returned {len(ordered)}; "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    return ordered


def resolve_type_order(
    typed_annotations: pd.DataFrame,
    type_col: str,
    types_rule: TypeRule = "label",
) -> list[str]:
    """Order the cell types present in *typed_annotations*."""
    present_types: list[str] = []
    seen: set[str] = set()
    for cell_type in typed_annotations[type_col]:
        if _is_missing_scalar(cell_type):
            continue
        name = _stringify_id_value(cell_type)
        if name not in seen:
            seen.add(name)
            present_types.append(name)
    return _apply_type_sorter(
        _resolve_type_rule(types_rule), present_types, typed_annotations, type_col
    )


def build_ordered_neurons(
    typed_annotations: pd.DataFrame,
    type_col: str,
    sorted_types: list[str],
    neuron_id_column: str,
    within: WithinTypeOrder = DEFAULT_WITHIN_TYPE_ORDER,
) -> tuple[list[str], dict[str, tuple[int, int]]]:
    """Lay the typed neurons out block by block, returning order and boundaries.

    *sorted_types* holds normalized names (as :func:`resolve_type_order` returns
    them), so rows are matched on the normalized type column rather than on the
    raw values. The neuron ID column is normalized the same way, so the returned
    IDs are always in the form the matrix axes use.
    """
    sorter_for = _resolve_within_type_order(within)
    normalized_types = typed_annotations[type_col].map(_stringify_id_value)
    typed_annotations = typed_annotations.assign(
        **{
            type_col: normalized_types,
            neuron_id_column: typed_annotations[neuron_id_column].map(
                _stringify_id_value
            ),
        }
    )
    ordered_neurons: list[str] = []
    boundaries: dict[str, tuple[int, int]] = {}
    current_pos = 0

    for c_type in sorted_types:
        type_name = _stringify_id_value(c_type)
        type_rows = typed_annotations[normalized_types == type_name]
        final_group = _apply_within_type_sorter(
            sorter_for(type_name), type_name, type_rows, neuron_id_column
        )
        if not final_group:
            continue

        ordered_neurons.extend(final_group)
        boundaries[type_name] = (current_pos, current_pos + len(final_group))
        current_pos += len(final_group)

    return ordered_neurons, boundaries


def build_axis_ordering(
    axis_ids: set[str],
    annotations: pd.DataFrame,
    id_col: str,
    type_col: str,
    order: NeuronOrderLike = None,
) -> AxisOrdering:
    """Resolve annotations and lay out one matrix axis end to end."""
    order = as_neuron_order(order)
    resolved = resolve_relevant_annotations(
        matrix_ids=axis_ids,
        annotations=annotations,
        id_col=id_col,
        type_col=type_col,
    )
    sorted_types = resolve_type_order(
        typed_annotations=resolved.typed,
        type_col=type_col,
        types_rule=order.types,
    )
    ordered_neurons, boundaries = build_ordered_neurons(
        typed_annotations=resolved.typed,
        type_col=type_col,
        sorted_types=sorted_types,
        neuron_id_column=id_col,
        within=order.within,
    )
    if resolved.untyped_ids:
        logger.warning(
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
        logger.warning(
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
    return AxisOrdering(
        ordered_neurons=tuple(ordered_neurons),
        type_boundaries=boundaries,
        neuron_to_type=resolved.id_map,
    )
