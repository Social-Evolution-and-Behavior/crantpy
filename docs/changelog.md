# Changelog

All notable changes to CRANTpy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial changelog file
- Backward compatibility for `parse_neuroncriteria` import from `crantpy.queries.neurons`
- Offline unit tests for `crantpy.viz.skeletonize` (`tests/test_skeletonize.py`) covering the SWC dict-to-`TreeNeuron` conversion, soma detection, and `get_skeletons` deduplication (no CAVE credentials required)
- `dataset` parameter on `skeletonize_neuron` and `skeletonize_neurons_parallel` so the on-demand mesh fetch uses the same dataset as the rest of the call
- `crantpy.utils.ordering`: a functional toolkit for ordering neurons along a matrix axis, with `NeuronOrder`, `ColumnOrderRule`, `EB_COLUMN_ORDER`, `DEFAULT_ORDER` and `DEFAULT_WITHIN_TYPE_ORDER` re-exported from the top level
  - The order objects are immutable, hashable, picklable and deep-copyable, so they work as dict keys, `lru_cache` arguments, and across process boundaries
  - Rules must be ordered iterables: `set`/`frozenset` are rejected because their iteration order is not stable, while key views such as `other_matrix.type_boundaries.keys()` are accepted. Note a sequence is a *preferred* order, not an exact one -- an upper-case entry also prefix-matches, so to reproduce another matrix's block order exactly, pass a callable
  - One-shot iterables (generators, iterators) are materialized once per call and shared, so passing one to both axes, or reusing it across every ROI of `from_synapses_by_neuropil`, behaves the same as passing a list
  - An `order.within` mapping naming the same cell type twice (e.g. `1` and `"1"`) is rejected rather than letting one shadow the other
  - `resolve_type_order`, `build_ordered_neurons` and `_order_types_by_size` normalize cell type names identically, and `build_ordered_neurons` / `_order_by_id` normalize the neuron ID column the same way, so the pipeline works on annotation frames that are not already stringified -- including integer labels and the whole-number floats pandas produces when such a column contains a NaN
- `EB_COLUMN_LABELS` and `EB_COLUMNAR_CELL_TYPES` in `crantpy.utils.config`, moving the ellipsoid-body column order out of the matrix module as plain data
- `NestedMatrix.typed_neurons` / `.untyped_neurons` and the four `source_`/`target_` equivalents on `DirectedNestedMatrix`, making the untyped tail of an axis visible instead of silently appended

### Changed

- **BREAKING**: `parse_neuroncriteria` has been moved from `crantpy.queries.neurons` to `crantpy.utils.decorators`
  - Backward compatibility maintained with deprecation warning
  - Users should update imports: `from crantpy.utils.decorators import parse_neuroncriteria`
  - Old import location will be removed in a future version
- `get_skeletons` now deduplicates `root_ids` before fetching and iterates fetches as they complete (`as_completed`), reporting the failing root id on error
- `skeletonize_neuron` treats `save_to` as an output directory when given multiple root IDs, writing one `<root_id>.swc` per neuron instead of overwriting a single file
- **BREAKING**: `type_order` on every `NestedMatrix` / `DirectedNestedMatrix` constructor is replaced by a single `order` argument, which now controls both the cell type blocks and the neurons inside them
  - `order` takes a `NeuronOrder`, a `{"types": ..., "within": ...}` mapping, or a bare sequence of cell type labels as types-only shorthand
  - `order.types` accepts `"label"`, `"size"`, a preferred sequence, or a callable; `order.within` accepts `"annotation"`, `"id"`, a `ColumnOrderRule`, a label sequence, a callable, or a `{cell_type: rule}` mapping. Within-type ordering was previously not configurable at all
  - On `DirectedNestedMatrix`, `source_type_order` / `target_type_order` become `source_order` / `target_order`
  - Migration: `type_order=X` becomes `order=X`; `source_type_order=X` becomes `source_order=X` (likewise for `target_`)
  - A custom sorter that drops or invents neurons now raises, naming the missing/unexpected IDs, instead of corrupting the axis
- **BREAKING**: the `source_neurons` / `target_neurons` *selectors* on `DirectedNestedMatrix` constructors are renamed `source_ids` / `target_ids`
  - The `.source_neurons` / `.target_neurons` properties are unchanged and still hold the resolved axis order; the old names collided with them
  - The `DirectedNestedMatrix(...)` constructor arguments keep their names, since they always meant the resolved order
- The "could not resolve a ranked column label", "missing `cell_type`" and "missing from annotations" warnings now come from the `crantpy.utils.ordering` logger instead of `crantpy.queries.nested_connectivity_matrices`, following the code that emits them. Per-module logging configuration targeting the matrix module needs updating
- The ellipsoid-body ordering rule for `"EPG/PEG"` is no longer applied from a private module table. It is the default value of `order.within`, and a `{cell_type: rule}` mapping is overlaid on it, so adding one type's rule keeps the built-in ones

### Deprecated

- Importing `parse_neuroncriteria` from `crantpy.queries.neurons` (use `crantpy.utils.decorators` instead)

### Fixed

- Neuron annotations with a non-unique index (from `pd.concat` or `set_index(drop=False)`) are accepted by the nested-matrix constructors; they previously failed with an opaque pandas `cannot reindex on an axis with duplicate labels`
- Import error in tests for `parse_neuroncriteria` function
- **Skeleton topology**: `_create_node_info_dict` now rebuilds parent pointers from undirected connectivity, fixing inverted roots and dropped branch-point children (meshparty emits `[child, parent]` edges) that produced fragmented skeletons in both the `pcg_skel` and precomputed-fetch paths
- **Wrong-dataset meshes**: the skeletor fallback in `skeletonize_neuron` now fetches the CloudVolume for the client's dataset instead of always the default dataset
- **Soma detection**: `detect_soma_skeleton` no longer skips a segment whose only large-radius node is at position 0; `detect_soma_mesh` neighbour counting is now vectorized (was O(n²))
- `skeletonize_neurons_parallel` colour generation aligns colours to the returned neurons and uses `plt.get_cmap` (compatible with current matplotlib); the precomputed-fetch failure reason in `get_skeletons` is now logged instead of silently swallowed


## Support

- 📖 **Documentation**: [crantpy.readthedocs.io](https://social-evolution-and-behavior.github.io/crantpy/)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/Social-Evolution-and-Behavior/crantpy/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/Social-Evolution-and-Behavior/crantpy/discussions)
- 📧 **Email**: [crantpy-dev@example.com](mailto:crantpy-dev@example.com)