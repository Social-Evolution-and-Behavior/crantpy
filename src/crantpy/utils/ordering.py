# -*- coding: utf-8 -*-
"""Private axis-ordering engine for :class:`~crantpy.queries.nested_connectivity_matrices.NestedMatrix`.

The public surface is ``NestedMatrix.order`` / ``NestedMatrix.by`` (and the
same names on ``DirectedNestedMatrix``). This module is not a user-facing
toolkit.
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

__all__ = [
    "AxisOrdering",
    "By",
    "MatrixOrder",
    "MatrixOrderLike",
    "ResolvedAnnotations",
    "as_matrix_order",
    "build_axis_ordering",
    "build_ordered_neurons",
    "by",
    "order",
    "resolve_relevant_annotations",
    "resolve_type_order",
]

logger = logging.getLogger(__name__)

_NA_POLICIES = ("last", "first", "raise")
_TYPE_STRINGS = ("label", "size")
_WITHIN_STRINGS = ("annotation", "id")


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


def _freeze_label_sequence(value: Any, what: str) -> tuple[str, ...]:
    """Materialize an ordered sequence of string labels."""
    if isinstance(value, str):
        raise TypeError(
            f"{what} takes an iterable of labels, not a bare string; "
            f"wrap it: [{value!r}]"
        )
    _reject_unordered(value, what)
    if not isinstance(value, Iterable):
        raise TypeError(f"{what} must be an iterable of string labels")
    entries = tuple(value)
    for entry in entries:
        if not isinstance(entry, str):
            raise TypeError(
                f"{what} entries must be strings; got {type(entry).__name__}: "
                f"{entry!r}"
            )
        _reject_bytes_like(entry, what)
    return entries


# ---------------------------------------------------------------------------
# by(): sort or rank a block by annotation column(s)
# ---------------------------------------------------------------------------

#: Splits a string into digit and non-digit runs, for :func:`by`.
_NATURAL_CHUNKS = re.compile(r"(\d+)")

#: Sort key standing in for a null value, never compared against a real key.
_MISSING_SORT_KEY: tuple[Any, ...] = ()


def _natural_sort_key(value: Any) -> tuple[tuple[int, int, str], ...]:
    """Key that sorts digit runs numerically: ``_g2`` before ``_g10``."""
    text = "" if _is_missing_scalar(value) else _stringify_id_value(value)
    return tuple(
        (1, int(chunk), "") if index % 2 else (0, 0, chunk.casefold())
        for index, chunk in enumerate(_NATURAL_CHUNKS.split(text))
    )


def _compile_extract(extract: Any) -> re.Pattern[str]:
    """Validate and compile a ``by(..., extract=)`` pattern."""
    if isinstance(extract, str):
        extract = re.compile(extract)
    if not isinstance(extract, re.Pattern) or not isinstance(extract.pattern, str):
        raise TypeError(
            "by(extract=...) must be a str regex or a compiled str pattern "
            f"(a bytes pattern cannot match); got {extract!r}"
        )
    if extract.groups < 1:
        raise ValueError(
            "by(extract=...) needs a capture group for the label; "
            f"{extract.pattern!r} has none"
        )
    return extract


def _leftmost_rank_label(text: str, rank: tuple[str, ...]) -> str | None:
    """Return the leftmost *rank* label in *text*.

    A label that ends in a digit is not allowed to match as a prefix of a
    longer digit run, so ``"R1"`` does not rank ``"EPG_R10"``. At the same
    start index, the longer label wins.
    """
    best_start: int | None = None
    best_label: str | None = None
    for label in rank:
        if not label:
            continue
        start = 0
        while True:
            idx = text.find(label, start)
            if idx < 0:
                break
            end = idx + len(label)
            if label[-1].isdigit() and end < len(text) and text[end].isdigit():
                start = idx + 1
                continue
            if (
                best_start is None
                or idx < best_start
                or (idx == best_start and len(label) > len(best_label or ""))
            ):
                best_start = idx
                best_label = label
            start = idx + 1
    return best_label


@dataclass(frozen=True)
class By:
    """The rule :func:`by` returns. Construct it through :func:`by`."""

    columns: tuple[str, ...]
    rank: tuple[str, ...] | None = None
    extract: re.Pattern[str] | None = None
    key: Callable[[Any], Any] | None = None
    na: Literal["last", "first", "raise"] = "last"

    def __repr__(self) -> str:
        parts = [repr(column) for column in self.columns]
        if self.rank is not None:
            parts.append(f"rank={list(self.rank)!r}")
        if self.extract is not None:
            parts.append(f"extract={self.extract.pattern!r}")
        if self.key is not None:
            parts.append(f"key={self.key!r}")
        if self.na != "last":
            parts.append(f"na={self.na!r}")
        return f"by({', '.join(parts)})"

    def __call__(
        self, type_name: str, type_rows: pd.DataFrame, neuron_id_column: str
    ) -> list[Any]:
        if type_rows.empty:
            return []

        present = [column for column in self.columns if column in type_rows.columns]
        if not present:
            raise ValueError(
                f"order.within for cell type {type_name!r} reads its labels from "
                f"{self.columns}, but the annotations have none of those "
                f"columns; available columns are "
                f"{[c for c in type_rows.columns if not c.startswith('__')]}"
            )

        rank_index = (
            {label: i for i, label in enumerate(self.rank)}
            if self.rank is not None
            else None
        )
        sort_key = self.key if self.key is not None else _natural_sort_key
        labels: list[Any] = []
        for _, row in type_rows.iterrows():
            labels.append(self._label_for(row, present, rank_index, neuron_id_column))

        missing_rank = 1 if self.na != "first" else 0
        ids = type_rows[neuron_id_column].tolist()
        keys: list[Any] = []
        missing = [label is None for label in labels]
        for label in labels:
            if label is None:
                keys.append(_MISSING_SORT_KEY)
            elif rank_index is not None:
                keys.append(rank_index[label])
            else:
                keys.append(sort_key(label))

        positions = sorted(
            range(len(ids)),
            key=lambda i: (
                missing_rank if missing[i] else 1 - missing_rank,
                keys[i],
                i,
            ),
        )
        return [ids[i] for i in positions]

    def _label_for(
        self,
        row: pd.Series,
        present: list[str],
        rank_index: dict[str, int] | None,
        neuron_id_column: str,
    ) -> Any:
        seen: dict[str, Any] = {}
        for column in present:
            value = row[column]
            seen[column] = value
            if _is_missing_scalar(value):
                continue
            text = str(value).strip()
            if self.extract is not None:
                match = self.extract.search(text)
                if not match:
                    continue
                label = match.group(1)
                if rank_index is None or label in rank_index:
                    return label
                continue
            if rank_index is not None:
                label = _leftmost_rank_label(text, self.rank or ())
                if label is not None:
                    return label
                continue
            return value

        if self.na == "raise":
            raise ValueError(
                f"Could not resolve a ranked column label for neuron "
                f"{row.get(neuron_id_column, '<unknown>')!r} from columns "
                f"{self.columns}: {seen!r}."
            )
        return None


def by(
    *columns: str,
    rank: Iterable[str] | None = None,
    extract: str | re.Pattern[str] | None = None,
    key: Callable[[Any], Any] | None = None,
    na: Literal["last", "first", "raise"] = "last",
) -> By:
    """Sort a cell type's neurons by annotation column(s).

    A string at the order layer is only a named rule (``"id"``,
    ``"annotation"``, ``"label"``, ``"size"``). Column names live here.

    Parameters
    ----------
    columns : str
        Annotation columns to read, in priority order. The first column that
        yields a usable value wins.
    rank : sequence of str, optional
        Explicit label order. Without *extract*, the leftmost *rank* label
        that appears in the cell is used, so ``"EPG/PEG_R1"`` and
        ``"Δ7_L8R1R9"`` both rank without a regex.
    extract : str or compiled pattern, optional
        Regex whose first group is the label. A plain string is compiled.
    key : callable, optional
        ``value -> sort key`` when *rank* is omitted. Defaults to a natural
        sort, so ``"ER_g2"`` precedes ``"ER_g10"``.
    na : {"last", "first", "raise"}, default "last"
        What an unrankable or null value costs. ``"raise"`` errors; the
        others put those neurons after or before the ranked ones, keeping
        annotation row order inside that group.

    Examples
    --------
    >>> NestedMatrix.by("cell_instance")
    >>> NestedMatrix.by("cell_instance", "cell_subtype", rank=EB_RING)
    >>> NestedMatrix.by("cell_instance", extract=r"([LR]\\d+)", rank=PB)
    """
    if not columns:
        raise TypeError("by() needs at least one annotation column name")
    frozen_columns: list[str] = []
    for column in columns:
        _reject_bytes_like(column, "by() columns")
        if not isinstance(column, str):
            raise TypeError(
                f"by() columns must be column name strings, got "
                f"{type(column).__name__}: {column!r}"
            )
        frozen_columns.append(column)
    if na not in _NA_POLICIES:
        raise ValueError(
            f"by(na=...) must be 'last', 'first', or 'raise', got {na!r}"
        )
    if key is not None and not callable(key):
        raise TypeError(f"by(key=...) must be callable, got {type(key).__name__}")
    if rank is not None and key is not None:
        raise TypeError("by(key=...) cannot be combined with rank=")
    frozen_rank = None if rank is None else _freeze_label_sequence(rank, "by(rank=...)")
    compiled = None if extract is None else _compile_extract(extract)
    return By(
        columns=tuple(frozen_columns),
        rank=frozen_rank,
        extract=compiled,
        key=key,
        na=na,
    )


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


def _snapshot_scalar_sequence(rule: Any, what: str) -> tuple[Any, ...]:
    _reject_unordered(rule, what)
    entries = tuple(rule)
    for entry in entries:
        _reject_bytes_like(entry, f"{what} entries")
        if isinstance(entry, Iterable) and not isinstance(entry, str):
            raise TypeError(
                "order rule entries must be scalar labels; got "
                f"{type(entry).__name__}: {entry!r}"
            )
    return entries


def _snapshot_type_rule(rule: Any) -> Any:
    if rule is None or isinstance(rule, str):
        if isinstance(rule, str) and rule not in _TYPE_STRINGS:
            raise ValueError(
                f"order.types string must be 'label' or 'size', got {rule!r}"
            )
        return rule
    if callable(rule):
        return rule
    if isinstance(rule, Mapping):
        raise TypeError(
            "order.types takes a flat sequence of cell type labels, not a "
            f"mapping; got {dict(rule)!r}"
        )
    return _snapshot_scalar_sequence(rule, "order.types")


def _snapshot_within_rule(rule: Any, what: str) -> Any:
    if rule is None or isinstance(rule, str):
        if isinstance(rule, str) and rule not in _WITHIN_STRINGS:
            raise ValueError(
                f"{what} string must be 'annotation' or 'id', got {rule!r}"
            )
        return rule
    if isinstance(rule, By):
        return rule
    if callable(rule):
        return rule
    if isinstance(rule, Mapping):
        raise TypeError(
            f"a per-cell-type mapping belongs on order.within, not as {what}; "
            f"got {dict(rule)!r}"
        )
    _reject_unordered(rule, what)
    if isinstance(rule, Iterable):
        raise TypeError(
            f"{what} no longer accepts a bare label sequence; wrap it: "
            f"NestedMatrix.by('cell_instance', rank={list(rule)!r})"
        )
    raise TypeError(
        f"{what} must be 'annotation', 'id', NestedMatrix.by(...), "
        f"or a callable; got {type(rule).__name__}"
    )


def _snapshot_within_mapping(within: Any) -> MappingProxyType | None:
    if within is None:
        return None
    if not isinstance(within, Mapping):
        raise TypeError(
            "order.within must be a {{cell_type: rule}} mapping; a rule that "
            "applies to every type belongs in default= (or .default(...)); "
            f"got {type(within).__name__}"
        )
    snapshotted: dict[Any, Any] = {}
    seen: dict[str, Any] = {}
    for name, rule in within.items():
        _reject_bytes_like(name, "order.within cell type keys")
        key = _stringify_id_value(name)
        if key in seen:
            raise ValueError(
                "order.within has two keys naming the same cell type "
                f"{key!r}: {seen[key]!r} and {name!r}"
            )
        seen[key] = name
        snapshotted[name] = _snapshot_within_rule(
            rule, f"order.within[{name!r}]"
        )
    return MappingProxyType(snapshotted)


def _thaw(rule: Any) -> Any:
    """Undo read-only wrappers, which do not pickle."""
    if isinstance(rule, Mapping):
        return {name: _thaw(nested) for name, nested in rule.items()}
    return rule


# ---------------------------------------------------------------------------
# MatrixOrder: nested call + builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatrixOrder:
    """How one matrix axis is ordered, at both of its levels.

    Construct through :func:`order` (``NestedMatrix.order``). ``types``
    orders the cell type blocks; ``default`` orders neurons inside a block
    that ``within`` does not name; ``within`` is a per-type override.
    Neither can move a neuron across a block boundary.
    """

    type_rule: Any = "label"
    default_rule: Any = None
    within_rules: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "type_rule", _snapshot_type_rule(self.type_rule))
        object.__setattr__(
            self,
            "default_rule",
            _snapshot_within_rule(self.default_rule, "order.default"),
        )
        object.__setattr__(
            self, "within_rules", _snapshot_within_mapping(self.within_rules)
        )

    def types(self, rule: Any) -> MatrixOrder:
        """Return a copy with a new block-order rule."""
        return MatrixOrder(
            type_rule=rule,
            default_rule=self.default_rule,
            within_rules=self.within_rules,
        )

    def default(self, rule: Any) -> MatrixOrder:
        """Return a copy whose unnamed types use *rule*."""
        return MatrixOrder(
            type_rule=self.type_rule,
            default_rule=rule,
            within_rules=self.within_rules,
        )

    def within(self, cell_type: Any, rule: Any = None) -> MatrixOrder:
        """Return a copy with a per-type override.

        ``.within("EPG/PEG", NestedMatrix.by(...))`` sets one type.
        ``.within({"EPG/PEG": ..., "delta7": ...})`` merges a mapping.
        """
        if rule is None:
            if not isinstance(cell_type, Mapping):
                raise TypeError(
                    "within(cell_type, rule) sets one type; "
                    "within({cell_type: rule, ...}) merges a mapping"
                )
            merged = dict(self.within_rules or {})
            merged.update(cell_type)
        else:
            merged = dict(self.within_rules or {})
            merged[cell_type] = rule
        return MatrixOrder(
            type_rule=self.type_rule,
            default_rule=self.default_rule,
            within_rules=merged,
        )

    def __repr__(self) -> str:
        within = None if self.within_rules is None else dict(self.within_rules)
        return (
            f"MatrixOrder(types={self.type_rule!r}, default={self.default_rule!r}, "
            f"within={within!r})"
        )

    def __getstate__(self) -> dict[str, Any]:
        return {
            "type_rule": _thaw(self.type_rule),
            "default_rule": _thaw(self.default_rule),
            "within_rules": _thaw(self.within_rules),
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        object.__setattr__(
            self, "type_rule", _snapshot_type_rule(state["type_rule"])
        )
        object.__setattr__(
            self,
            "default_rule",
            _snapshot_within_rule(state["default_rule"], "order.default"),
        )
        object.__setattr__(
            self, "within_rules", _snapshot_within_mapping(state["within_rules"])
        )


def order(
    types: Any = "label",
    default: Any = None,
    within: Any = None,
) -> MatrixOrder:
    """Build a reusable neuron order for a nested matrix axis.

    Nested call::

        NestedMatrix.order(
            types=["ER2", "EPG/PEG", "delta7"],
            default="id",
            within={"EPG/PEG": NestedMatrix.by("cell_instance", rank=EB_RING)},
        )

    Builder::

        NestedMatrix.order().types(["ER2", "EPG/PEG"]).default("id").within(
            "EPG/PEG", NestedMatrix.by("cell_instance", rank=EB_RING)
        )

    A bare sequence passed as ``order=`` to a matrix constructor is still
    types-only shorthand for ``NestedMatrix.order(types=...)``.
    """
    return MatrixOrder(type_rule=types, default_rule=default, within_rules=within)


MatrixOrderLike = MatrixOrder | Iterable[Any] | None


def as_matrix_order(spec: MatrixOrderLike) -> MatrixOrder:
    """Coerce an ``order=`` argument into a :class:`MatrixOrder`."""
    if spec is None:
        return MatrixOrder()
    if isinstance(spec, MatrixOrder):
        return spec
    if isinstance(spec, Mapping):
        raise TypeError(
            "order= no longer accepts a mapping; use NestedMatrix.order("
            "types=..., default=..., within=...) or the builder "
            "NestedMatrix.order().types(...).default(...).within(...)"
        )
    if isinstance(spec, str):
        raise TypeError(
            f"order={spec!r} is ambiguous: say which level it applies to, e.g. "
            f"order=NestedMatrix.order(types={spec!r}) or "
            f"order=NestedMatrix.order(default={spec!r})"
        )
    if isinstance(spec, Iterable):
        return MatrixOrder(type_rule=spec)
    raise TypeError(
        "order must be NestedMatrix.order(...), a sequence of cell type "
        f"labels, or None; got {type(spec).__name__}"
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
    """Order cell type labels, honouring *preferred* where it applies."""
    _reject_unordered(preferred, "_sort_cell_types(preferred=...)")
    unique = {_stringify_id_value(t) for t in types if not _is_missing_scalar(t)}

    parsed = {label: _parse_type_label(label) for label in unique}
    generic_order = sorted(unique, key=lambda x: (*parsed[x][:2], x))

    if not preferred:
        return generic_order

    result: list[str] = []
    remaining = set(generic_order)
    unmatched: list[str] = []

    for preferred_value in preferred:
        if _is_missing_scalar(preferred_value):
            continue
        _reject_bytes_like(preferred_value, "preferred cell type entries")
        wanted = _stringify_id_value(preferred_value)
        matches = [t for t in generic_order if t == wanted]
        if not matches:
            unmatched.append(wanted)
        for match in matches:
            if match in remaining:
                result.append(match)
                remaining.remove(match)

    if unmatched:
        logger.warning(
            "order.types names %d cell type(s) that no neuron on this axis "
            "has: %s, so those entries did nothing; matching is exact and "
            "case-sensitive. Present types: %s",
            len(unmatched),
            unmatched,
            generic_order,
        )

    result.extend([t for t in generic_order if t in remaining])
    return result


def _order_types_by_label(
    type_names: list[str], typed_annotations: pd.DataFrame, type_col: str
) -> list[str]:
    return _sort_cell_types(type_names)


def _order_types_by_size(
    type_names: list[str], typed_annotations: pd.DataFrame, type_col: str
) -> list[str]:
    counts = typed_annotations[type_col].map(_stringify_id_value).value_counts()
    by_label = _sort_cell_types(type_names)
    return sorted(
        by_label, key=lambda name: (-int(counts.get(name, 0)), by_label.index(name))
    )


def _order_types_by_preferred(preferred: Iterable[Any]):
    _reject_unordered(preferred, "_order_types_by_preferred(preferred=...)")
    preferred = list(preferred)

    def sorter(
        type_names: list[str], typed_annotations: pd.DataFrame, type_col: str
    ) -> list[str]:
        return _sort_cell_types(type_names, preferred=preferred)

    return sorter


def _resolve_type_rule(rule: Any):
    if rule is None or rule == "label":
        return _order_types_by_label
    if rule == "size":
        return _order_types_by_size
    if isinstance(rule, str):
        raise ValueError(f"order.types string must be 'label' or 'size', got {rule!r}")
    if callable(rule):
        def sorter(
            type_names: list[str], typed_annotations: pd.DataFrame, type_col: str
        ) -> list[Any]:
            return list(rule(list(type_names)))

        return sorter
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
    sorter,
    type_names: list[str],
    typed_annotations: pd.DataFrame,
    type_col: str,
) -> list[str]:
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


# ---------------------------------------------------------------------------
# Within-type ordering
# ---------------------------------------------------------------------------


def _order_by_annotation(
    type_name: str, type_rows: pd.DataFrame, neuron_id_column: str
) -> list[Any]:
    return type_rows[neuron_id_column].tolist()


def _order_by_id(
    type_name: str, type_rows: pd.DataFrame, neuron_id_column: str
) -> list[Any]:
    """Sort by neuron ID, numerically when every ID is a non-negative integer."""
    ids = [_stringify_id_value(value) for value in type_rows[neuron_id_column]]
    if ids and all(value.isdecimal() for value in ids):
        return sorted(ids, key=int)
    return sorted(ids)


def _adapt_within_callable(rule: Callable[[pd.DataFrame], Any]):
    def sorter(
        type_name: str, type_rows: pd.DataFrame, neuron_id_column: str
    ) -> list[Any]:
        result = rule(type_rows)
        if isinstance(result, pd.DataFrame):
            return result[neuron_id_column].tolist()
        return list(result)

    return sorter


def _resolve_within_type_rule(rule: Any):
    if rule is None or rule == "annotation":
        return _order_by_annotation
    if rule == "id":
        return _order_by_id
    if isinstance(rule, str):
        raise ValueError(
            f"order.within string must be 'annotation' or 'id', got {rule!r}"
        )
    if isinstance(rule, By):
        return rule
    if callable(rule):
        return _adapt_within_callable(rule)
    if isinstance(rule, Mapping):
        raise TypeError(
            "a per-cell-type mapping belongs at the top of order.within, not "
            f"nested inside it; got {dict(rule)!r} as one type's rule"
        )
    _reject_unordered(rule, "order.within")
    if isinstance(rule, Iterable):
        raise TypeError(
            "order.within no longer accepts a bare label sequence; wrap it: "
            f"NestedMatrix.by('cell_instance', rank={list(rule)!r})"
        )
    raise TypeError(
        "order.within rules must be 'annotation', 'id', NestedMatrix.by(...), "
        f"or a callable; got {type(rule).__name__}"
    )


def _within_type_keys(within: Mapping[Any, Any]) -> set[str]:
    return {_stringify_id_value(name) for name in within}


def _resolve_within_type_order(
    within: Any, default: Any = None
) -> Callable[[str], Any]:
    default_sorter = _resolve_within_type_rule(default)
    if within is None:
        return lambda type_name: default_sorter
    if not isinstance(within, Mapping):
        raise TypeError(
            "order.within must be a {cell_type: rule} mapping; a rule that "
            "applies to every type belongs in default="
        )
    by_type: dict[str, Any] = {}
    seen: dict[str, Any] = {}
    for name, rule in within.items():
        _reject_bytes_like(name, "order.within cell type keys")
        key = _stringify_id_value(name)
        if key in seen:
            raise ValueError(
                "order.within has two keys naming the same cell type "
                f"{key!r}: {seen[key]!r} and {name!r}"
            )
        seen[key] = name
        by_type[key] = _resolve_within_type_rule(rule)
    return lambda type_name: by_type.get(type_name, default_sorter)


def _apply_within_type_sorter(
    sorter,
    type_name: str,
    type_rows: pd.DataFrame,
    neuron_id_column: str,
) -> list[str]:
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
    """Ordered neurons and type metadata for one matrix axis."""

    ordered_neurons: tuple[str, ...]
    type_boundaries: dict[str, tuple[int, int]]
    neuron_to_type: dict[str, Any]


def resolve_relevant_annotations(
    matrix_ids: set[str],
    annotations: pd.DataFrame,
    id_col: str,
    type_col: str,
) -> ResolvedAnnotations:
    """Narrow *annotations* to *matrix_ids*, de-duplicating by neuron."""
    valid_ids = annotations[id_col].map(lambda value: not _is_missing_scalar(value))
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


def resolve_type_order(
    typed_annotations: pd.DataFrame,
    type_col: str,
    types_rule: Any = "label",
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
    within: Any = None,
    default: Any = None,
) -> tuple[list[str], dict[str, tuple[int, int]]]:
    """Lay the typed neurons out block by block, returning order and boundaries."""
    sorter_for = _resolve_within_type_order(within, default=default)

    if isinstance(within, Mapping):
        present_types = {_stringify_id_value(t) for t in sorted_types}
        unused = sorted(_within_type_keys(within) - present_types)
        if unused:
            logger.warning(
                "order.within names %s, which match no cell type in this matrix, "
                "so those rules were not applied; matching is exact and "
                "case-sensitive. Present types: %s",
                unused,
                sorted(present_types),
            )

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
    order: MatrixOrderLike = None,
) -> AxisOrdering:
    """Resolve annotations and lay out one matrix axis end to end."""
    order = as_matrix_order(order)
    resolved = resolve_relevant_annotations(
        matrix_ids=axis_ids,
        annotations=annotations,
        id_col=id_col,
        type_col=type_col,
    )
    sorted_types = resolve_type_order(
        typed_annotations=resolved.typed,
        type_col=type_col,
        types_rule=order.type_rule,
    )
    ordered_neurons, boundaries = build_ordered_neurons(
        typed_annotations=resolved.typed,
        type_col=type_col,
        sorted_types=sorted_types,
        neuron_id_column=id_col,
        within=order.within_rules,
        default=order.default_rule,
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
