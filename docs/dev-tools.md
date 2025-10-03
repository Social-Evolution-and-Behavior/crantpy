# CRANTpy Development Tools

This document provides a quick reference for the development tools available in CRANTpy.

## 🔧 Scripts Overview

| Script          | Purpose                                              | Usage                                  |
| --------------- | ---------------------------------------------------- | -------------------------------------- |
| `setup_dev.sh`  | Set up development environment (safe mode by default) | `./setup_dev.sh --install`             |
| `format.sh`     | Manual code formatting and linting                   | `./format.sh`                          |
| `deploy.sh`     | Package deployment to PyPI                           | `./deploy.sh --bump patch test-deploy` |
| `api_token.sh`  | Configure PyPI API tokens                            | `./api_token.sh --setup`               |
| `build_docs.sh` | Build documentation                                  | `./build_docs.sh --clean`              |

## 📋 Migration Status

🚨 **Pre-commit hooks are currently DISABLED by default** to prevent merge conflicts.

**Migration Timeline:**
- **Phase 1 (Current)**: Pre-commit ready but disabled, manual formatting available
- **Phase 2 (Next Week)**: Enable pre-commit project-wide after branch merges
- **Phase 3 (Ongoing)**: Consistent formatting for all new development

See [Pre-commit Migration Guide](pre-commit-migration.md) for details.

## 🚀 Quick Start for New Contributors

⚠️ **IMPORTANT**: Pre-commit hooks are currently **disabled by default** to prevent merge conflicts with active branches.

1. **Clone and setup**:

   ```bash
   git clone https://github.com/Social-Evolution-and-Behavior/crantpy.git
   cd crantpy
   poetry install
   ```

2. **Set up development environment** (safe mode):

   ```bash
   ./setup_dev.sh --install
   # Choose "No" when prompted (recommended until migration day)
   ```

3. **Start developing**:

   ```bash
   git checkout -b feature/my-feature
   # Make your changes...
   ./format.sh  # Manual formatting until pre-commit is enabled
   git add .
   git commit -m "Add new feature"
   ```

### Advanced: Force Pre-commit Installation

⚠️ **WARNING**: This may cause merge conflicts with existing branches!

```bash
./setup_dev.sh --install --force
```

## 🔍 Pre-commit Hooks

### What They Do

- **Format code** with Black (88 character line length)
- **Lint code** with Ruff (fix common issues automatically)
- **Check files** for trailing whitespace, proper endings
- **Validate syntax** of YAML, JSON, TOML files
- **Format notebooks** (Jupyter notebooks)
- **Lint documentation** (Markdown files)

### Manual Usage

```bash
# Run all hooks on all files
poetry run pre-commit run --all-files

# Run specific hook
poetry run pre-commit run black

# Skip hooks for a commit
git commit --no-verify -m "Skip hooks"

# Skip specific hooks
SKIP=black,ruff git commit -m "Skip formatting"
```

## 🎨 Code Formatting

### Automatic (Recommended)

Pre-commit hooks handle this automatically. Just commit your code!

### Manual

```bash
# Quick format everything
./format.sh

# Individual tools
poetry run black src/ tests/
poetry run ruff check src/ tests/ --fix
poetry run ruff format src/ tests/
```

## 📦 Package Deployment

### First Time Setup

```bash
# Configure PyPI tokens
./api_token.sh --setup
```

### Deploy Process

```bash
# Test deployment
./deploy.sh --bump patch test-deploy

# Production deployment
./deploy.sh --bump minor deploy
```

## 🔧 Development Environment

### Check Setup Status

```bash
./setup_dev.sh --check
```

### Troubleshooting

**Pre-commit hooks not installed (by design):**

```bash
# This is currently expected behavior
./setup_dev.sh --check
# Shows that hooks are not installed

# To force install (may cause merge conflicts):
./setup_dev.sh --install --force
```

**Pre-commit hooks fail:**

```bash
# Re-install hooks
./setup_dev.sh --install --force

# Update hook repositories
poetry run pre-commit autoupdate
```

**Formatting issues:**

```bash
# Manual format to see what's happening
./format.sh

# Check what Ruff wants to fix
poetry run ruff check src/ tests/
```

**Import errors after adding new code:**

```bash
# Update lazy imports
poetry run mkinit --lazy_loader src/crantpy --recursive --inplace
```

## 🚦 Workflow Integration

### With Deploy Script

The deploy script automatically runs pre-commit hooks if they're set up, falling back to manual formatting if not.

### With CI/CD

Pre-commit hooks ensure code quality before it reaches CI, making builds faster and more reliable.

### With VS Code

Install the pre-commit VS Code extension to run hooks in your editor.

## 📋 Configuration Files

- `.pre-commit-config.yaml` - Pre-commit hook configuration
- `pyproject.toml` - Python project configuration (includes Black and Ruff settings)
- `setup_dev.sh` - Development environment setup script
- `format.sh` - Manual formatting script

## 🆘 Getting Help

If you run into issues:

1. **Check setup status**: `./setup_dev.sh --check`
2. **Re-run setup**: `./setup_dev.sh --install`
3. **Manual format**: `./format.sh`
4. **Check documentation**: See `docs/contribute.md`
5. **Open an issue**: [GitHub Issues](https://github.com/Social-Evolution-and-Behavior/crantpy/issues)
