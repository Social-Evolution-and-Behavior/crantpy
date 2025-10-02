# Changelog

All notable changes to CRANTpy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial changelog file
- Backward compatibility for `parse_neuroncriteria` import from `crantpy.queries.neurons`

### Changed

- **BREAKING**: `parse_neuroncriteria` has been moved from `crantpy.queries.neurons` to `crantpy.utils.decorators`
  - Backward compatibility maintained with deprecation warning
  - Users should update imports: `from crantpy.utils.decorators import parse_neuroncriteria`
  - Old import location will be removed in a future version

### Deprecated

- Importing `parse_neuroncriteria` from `crantpy.queries.neurons` (use `crantpy.utils.decorators` instead)

### Fixed

- Import error in tests for `parse_neuroncriteria` function


## Support

- 📖 **Documentation**: [crantpy.readthedocs.io](https://social-evolution-and-behavior.github.io/crantpy/)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/Social-Evolution-and-Behavior/crantpy/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/Social-Evolution-and-Behavior/crantpy/discussions)
- 📧 **Email**: [crantpy-dev@example.com](mailto:crantpy-dev@example.com)