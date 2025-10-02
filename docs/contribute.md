# Contributing to CRANTpy

We welcome contributions to CRANTpy! This guide will help you get started with contributing to the project.

## 🎯 Ways to Contribute

- 🐛 **Report bugs** and suggest fixes
- 💡 **Propose new features** and enhancements  
- 📖 **Improve documentation** and examples
- 🧪 **Write tests** and improve code coverage
- 🔧 **Submit code** improvements and bug fixes
- 💬 **Help other users** in discussions

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- **Poetry** (required for dependency management) - Install from [python-poetry.org](https://python-poetry.org/)

### Development Setup

1. **Fork and clone the repository**

```bash
git clone https://github.com/your-username/crantpy.git
cd crantpy
```

2. **Install development dependencies with Poetry**

```bash
# Poetry is REQUIRED for this project
poetry install
```

This will:
- Create a virtual environment
- Install all dependencies and development tools
- Set up the project for development

3. **Set up pre-commit hooks**

```bash
poetry run pre-commit install
```

4. **Verify installation**

```bash
poetry run pytest tests/
./build_docs.sh --clean
```

## 📝 Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number
```

### 2. Make Changes

Follow these coding standards:

- **Start each file with UTF-8 encoding and detailed docstring**:
  ```python
  # -*- coding: utf-8 -*-
  """
  This module provides XXX functionality for YYY.
  
  Detailed description of what this module does,
  its main classes, functions, and usage patterns.
  """
  ```

- **Use type hints** for all function parameters and return values
- **Follow existing code structure** and patterns
- **Add tests** for new functionality
- **Update documentation** as needed
- **Add `__init__.py`** files to new folders

### 3. Update Module Imports (Required)

After adding new code, **always** run:

```bash
poetry run mkinit --lazy_loader src/crantpy --recursive --inplace
```

This updates the `__init__.py` files with lazy loading for efficient imports.

### 4. Test Your Changes

```bash
# Run tests
poetry run pytest tests/

# Check code style
poetry run black src/ tests/
poetry run ruff check src/ tests/

# Build and test documentation (REQUIRED before commit)
./build_docs.sh --clean
```

### 5. Pre-Commit Requirements

**Before every commit**, ensure you run:

```bash
# 1. Update lazy imports
poetry run mkinit --lazy_loader src/crantpy --recursive --inplace

# 2. Build documentation
./build_docs.sh --clean

# 3. Run tests
poetry run pytest tests/

# 4. Check code formatting
poetry run black src/ tests/
poetry run ruff check src/ tests/
```

### 6. Commit and Push

```bash
git add .
git commit -m "feat: add new feature description"
git push origin feature/your-feature-name
```

### 7. Create Pull Request

- Open a pull request against the `main` branch
- Fill out the pull request template
- Link any related issues

## 🧪 Testing

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_queries.py

# Run with coverage
poetry run pytest --cov=crantpy tests/
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies (CAVE API calls)

Example test:

```python
import pytest
from crantpy import NeuronCriteria

def test_neuron_criteria_creation():
    """Test that NeuronCriteria can be created with valid parameters."""
    criteria = NeuronCriteria(cell_class='kenyon_cell')
    assert criteria.cell_class == 'kenyon_cell'

def test_invalid_cell_class():
    """Test that invalid cell class raises appropriate error."""
    with pytest.raises(ValueError):
        NeuronCriteria(cell_class='invalid_class')
```

## 📖 Documentation

### Building Documentation (Required)

**Always use the provided build script:**

```bash
# Build and test documentation
./build_docs.sh --clean

# Build and deploy (for maintainers)
./build_docs.sh --clean --deploy
```

**Never use `jupyter-book build` directly** - use `build_docs.sh` which handles:
- Dependency compatibility fixes
- Proper error handling
- Consistent build environment

### Documentation Guidelines

- Use clear, concise language
- Include executable code examples
- Add comprehensive docstrings to all public functions
- Update relevant tutorials when adding features
- **Test documentation build before every commit**

### Docstring Format

Use Google-style docstrings:

```python
def get_skeletons(root_ids, dataset='latest'):
    """Retrieve neuron skeletons for given root IDs.
    
    Args:
        root_ids: List of neuron root IDs or single root ID
        dataset: Dataset version to use ('latest' or 'sandbox')
        
    Returns:
        List of TreeNeuron objects or single TreeNeuron
        
    Raises:
        ValueError: If root_ids is empty or invalid
        ConnectionError: If CAVE service is unavailable
        
    Example:
        >>> skeleton = cp.get_skeletons(123456789)
        >>> skeletons = cp.get_skeletons([123456789, 987654321])
    """
```

## 🎨 Code Style & Standards

### File Structure Requirements

**Every new Python file must start with:**

```python
# -*- coding: utf-8 -*-
"""
This module provides [detailed description of functionality].

[Comprehensive explanation of what this module does, its main classes,
functions, and usage patterns. Be as detailed as possible.]
"""
```

### Type Hints (Mandatory)

Use type hints for **all** function parameters and return values:

```python
from typing import List, Optional, Union, Dict, Any
from crantpy.utils.types import NeuronID, RootID

def get_skeletons(
    root_ids: Union[NeuronID, List[NeuronID]], 
    dataset: str = 'latest',
    simplify: bool = False
) -> Union[TreeNeuron, List[TreeNeuron]]:
    """Retrieve neuron skeletons with full type specification."""
    pass
```

**Benefits of type hints:**
- Easier debugging with type assertion
- Clear input/output expectations
- Better IDE support and autocompletion
- See `crantpy/utils/types.py` for custom compound types

### Module Organization

**For new folders:**
1. Always create `__init__.py` files
2. After adding code, run the lazy import updater:

```bash
poetry run mkinit --lazy_loader src/crantpy --recursive --inplace
```

### Code Formatting

- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [Ruff](https://ruff.rs/) for linting  
- Maximum line length: 88 characters
- **Poetry is required** - no pip installations

### Project Structure

```
src/crantpy/
├── __init__.py          # Main public API
├── queries/             # Neuron querying functionality
├── utils/              # Utility functions
│   ├── cave/           # CAVE-specific utilities
│   └── config.py       # Configuration management
└── viz/                # Visualization tools
```

### Naming Conventions

- **Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_snake_case`

## 🚨 Issue Guidelines

### Reporting Bugs

Use the bug report template and include:

- **Environment details** (Python version, OS, CRANTpy version)
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Code example** that demonstrates the bug
- **Error messages** (full traceback)

### Feature Requests

Use the feature request template and include:

- **Clear description** of the proposed feature
- **Use case** and motivation
- **Possible implementation** approach
- **Examples** of how it would be used

## 📋 Pull Request Guidelines

### Before Submitting (Checklist)

- [ ] **Poetry used** for all dependency management
- [ ] **UTF-8 header and docstring** added to new files
- [ ] **Type hints** added to all functions
- [ ] **`__init__.py`** files created for new folders
- [ ] **Lazy imports updated**: `poetry run mkinit --lazy_loader src/crantpy --recursive --inplace`
- [ ] **Documentation built successfully**: `./build_docs.sh --clean`
- [ ] **Tests pass locally**: `poetry run pytest tests/`
- [ ] **Code formatted**: `poetry run black src/ tests/`
- [ ] **Linting passes**: `poetry run ruff check src/ tests/`
- [ ] **Changelog updated** (for significant changes)
- [ ] **Commit messages follow conventional format**

### Pull Request Template

Your PR should include:

- **Description** of changes made
- **Type of change** (bug fix, feature, docs, etc.)
- **Testing** performed
- **Related issues** (if any)

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **Code review** by maintainers
3. **Testing** on different environments
4. **Documentation review** if applicable
5. **Final approval** and merge

## 🏷️ Release Process

### Version Numbers

We follow [Semantic Versioning](https://semver.org/):

- `MAJOR.MINOR.PATCH` (e.g., 1.2.3)
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

feat: add new visualization backend
fix: resolve authentication token refresh
docs: update installation guide
test: add tests for neuron queries
```

## 💬 Community

### Communication Channels

- **GitHub Discussions**: General questions and ideas
- **GitHub Issues**: Bug reports and feature requests
- **Email**: [crantpy-dev@example.com](mailto:crantpy-dev@example.com)

### Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md). We are committed to providing a welcoming and inclusive environment for all contributors.

## 🙏 Recognition

Contributors are recognized in:

- **README.md** acknowledgments
- **Release notes** for significant contributions
- **Documentation** author credits

## 📦 Package Deployment (Maintainers Only)

### Prerequisites for Deployment

1. **PyPI Account**: You need accounts on both [PyPI](https://pypi.org/) and [TestPyPI](https://test.pypi.org/)
2. **API Tokens**: Create API tokens for both services
3. **Poetry Configuration**: Configure Poetry with your API tokens

### Create API Tokens

**TestPyPI Token:**
- Go to: https://test.pypi.org/manage/account/token/
- Click 'Add API token'
- Token name: 'crantpy-dev' (or any name you prefer)
- Scope: 'Entire account' (or limit to specific project)
- Copy the generated token (starts with 'pypi-')

**PyPI Token:**
- Go to: https://pypi.org/manage/account/token/
- Click 'Add API token'
- Token name: 'crantpy-dev' (or any name you prefer)
- Scope: 'Entire account' (or limit to specific project)
- Copy the generated token (starts with 'pypi-')

### Quick Setup with API Token Script

We provide an automated script to set up your PyPI tokens:

```bash
# Run the interactive token setup
./api_token.sh --setup

# Check configuration status
./api_token.sh --status

# Test if tokens are working
./api_token.sh --test

# Clean up configuration (if needed)
./api_token.sh --cleanup
```

The script will:
- Guide you through getting API tokens from PyPI and TestPyPI
- Automatically configure your shell environment (.bashrc/.zshrc)
- Set up Poetry with the correct tokens and repositories
- Test the configuration to ensure everything works

### Manual Setup (Alternative)

If you prefer to set up tokens manually:

#### Configure Environment Variables

Add to your shell configuration file (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
# CRANTpy PyPI Tokens
export POETRY_PYPI_TOKEN_TESTPYPI="pypi-your-testpypi-token-here"
export POETRY_PYPI_TOKEN_PYPI="pypi-your-pypi-token-here"
```

Reload your shell:

```bash
source ~/.bashrc  # or source ~/.zshrc
```

### Security Best Practices

- **Never commit tokens to git**: Tokens are automatically excluded by `.gitignore`
- **Use environment variables**: The deployment script supports both Poetry config and environment variables
- **Rotate tokens regularly**: Create new tokens periodically and update your configuration
- **Limit token scope**: When possible, limit tokens to specific projects rather than entire account

### Troubleshooting Token Issues

**Script won't run:**
```bash
chmod +x api_token.sh
./api_token.sh --setup
```

**Tokens not working:**
```bash
# Check current status
./api_token.sh --status

# Clean and reconfigure
./api_token.sh --cleanup
./api_token.sh --setup
```

**Environment variables not loading:**
```bash
# Restart terminal or reload shell config
source ~/.bashrc  # or ~/.zshrc
```

### Deployment Script Usage

The project includes a comprehensive deployment script (`deploy.sh`) that automates the entire release process:

**Available Commands:**
- `test-deploy`: Deploy to TestPyPI for testing
- `deploy`: Deploy to production PyPI
- `build-only`: Build package without deploying

**Version Bumping:**
- `--bump patch`: Bug fixes (0.1.0 → 0.1.1)
- `--bump minor`: New features (0.1.0 → 0.2.0)
- `--bump major`: Breaking changes (0.1.0 → 1.0.0)

**Skip Options:**
- `--skip-tests`: Skip running pytest
- `--skip-docs`: Skip building documentation
- `--skip-checks`: Skip code quality checks

### Example Deployment Workflows

**First Release:**
```bash
# 1. Test the release process
./deploy.sh --bump patch test-deploy

# 2. Verify the test package
pip install -i https://test.pypi.org/simple/ crantpy==0.1.1

# 3. If everything works, deploy to production
./deploy.sh deploy
```

**Regular Updates:**
```bash
# For bug fixes
./deploy.sh --bump patch deploy

# For new features  
./deploy.sh --bump minor deploy

# For breaking changes
./deploy.sh --bump major deploy
```

**Development Build:**
```bash
# Just build without deploying
./deploy.sh build-only
```

### What the Deployment Script Does

1. **Pre-flight Checks:**
   - Verifies Poetry installation
   - Checks project structure
   - Validates git status

2. **Quality Assurance:**
   - Updates lazy imports with mkinit
   - Formats code with Black
   - Lints code with Ruff
   - Runs tests with pytest
   - Builds documentation

3. **Package Building:**
   - Cleans previous builds
   - Builds with Poetry
   - Validates with twine

4. **Deployment:**
   - Deploys to TestPyPI or PyPI
   - Creates git tags for releases
   - Pushes tags to remote

### Troubleshooting Deployment

**Authentication Issues:**
```bash
# Check Poetry configuration
poetry config --list

# Reconfigure tokens
poetry config pypi-token.pypi your-new-token
```

**Build Failures:**
- Check that all tests pass: `poetry run pytest`
- Verify code quality: `poetry run ruff check src/`
- Ensure documentation builds: `./build_docs.sh --clean`

**Version Conflicts:**
- Check existing versions: `pip index versions crantpy`
- Use appropriate version bump: `--bump major|minor|patch`

### Deployment Best Practices

1. **Always test first**: Use `test-deploy` before `deploy`
2. **Follow semantic versioning**: Use appropriate bump levels
3. **Keep git clean**: Commit changes before deploying
4. **Document changes**: Update CHANGELOG.md
5. **Tag releases**: The script automatically creates git tags

### Security Notes

- Store API tokens securely
- Never commit tokens to git
- Use environment variables for CI/CD
- Regularly rotate tokens

Thank you for contributing to CRANTpy! 🎉