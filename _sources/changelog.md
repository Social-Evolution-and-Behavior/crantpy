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

### Changed

- **BREAKING**: `parse_neuroncriteria` has been moved from `crantpy.queries.neurons` to `crantpy.utils.decorators`
  - Backward compatibility maintained with deprecation warning
  - Users should update imports: `from crantpy.utils.decorators import parse_neuroncriteria`
  - Old import location will be removed in a future version
- `get_skeletons` now deduplicates `root_ids` before fetching and iterates fetches as they complete (`as_completed`), reporting the failing root id on error
- `skeletonize_neuron` treats `save_to` as an output directory when given multiple root IDs, writing one `<root_id>.swc` per neuron instead of overwriting a single file

### Deprecated

- Importing `parse_neuroncriteria` from `crantpy.queries.neurons` (use `crantpy.utils.decorators` instead)

### Fixed

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