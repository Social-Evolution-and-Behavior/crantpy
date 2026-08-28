# Installation & Setup

This guide will help you install CRANTpy and set up the necessary authentication to access the CRANT datasets.

## Requirements

- **Python**: 3.10 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: At least 8GB RAM recommended for large-scale analyses

## Installation Methods

### Option 1: Install from PyPI (Recommended)

```bash
pip install crantpy
```

### Option 2: Install from Source

For the latest development version or if you want to contribute:

```bash
git clone https://github.com/Social-Evolution-and-Behavior/crantpy.git
cd crantpy
pip install -e .
```

### Option 3: Using Poetry (For Development)

If you're planning to contribute or need a development environment:

```bash
git clone https://github.com/Social-Evolution-and-Behavior/crantpy.git
cd crantpy
poetry install
```

## Authentication Setup

CRANTpy requires authentication to access the CAVE (Connectome Annotation Versioning Engine) service and Seatable annotations.

### CAVE Token Setup

1. **Generate a CAVE token** (interactive):

   ```python
   import crantpy as cp
   cp.generate_cave_token(save=True)
   ```
   
   This will:
   - Open a browser window for authentication
   - Save the token securely for future use
   - Set up the token for the current session

2. **Manual token setup** (if you already have a token):

   ```python
   import crantpy as cp
   cp.set_cave_token("your_token_here")
   ```

### Environment Variables (Optional)

You can set up environment variables for automatic configuration:

```bash
# Set default dataset
export CRANT_DEFAULT_DATASET=latest

# Set Seatable API token (if available)
export CRANT_SEATABLE_API_TOKEN=your_seatable_token
```

## Verification

Test your installation with a simple query:

```python
import crantpy as cp

# Set logging to see progress
cp.set_logging_level("INFO")

# Test CAVE connection
client = cp.get_cave_client()
print(f"Connected to datastack: {client.datastack_name}")

# Test data access
try:
    # Get a small sample of neurons
    sample_neurons = cp.NeuronCriteria(side='L').get_roots()[:5]
    print(f"Successfully retrieved {len(sample_neurons)} neuron IDs")
    print("Installation successful! ✅")
except Exception as e:
    print(f"Error: {e}")
    print("Please check your authentication setup")
```

## Configuration Options

### Default Dataset

CRANTpy supports multiple dataset versions:

- **`latest`**: The most recent stable dataset (default)
- **`sandbox`**: Development/testing dataset

Set the default dataset:

```python
import crantpy as cp
cp.set_default_dataset("latest")  # or "sandbox"
```

### Logging

Control the verbosity of CRANTpy operations:

```python
import crantpy as cp

# Set logging level
cp.set_logging_level("INFO")    # INFO, DEBUG, WARNING, ERROR
```

### Caching

CRANTpy uses intelligent caching to improve performance. Cache location:

- **Linux/macOS**: `~/.crantpy/`
- **Windows**: `%USERPROFILE%/.crantpy/`

Clear caches if needed:

```python
import crantpy as cp

# Clear all caches
cp.clear_all_caches()

# Clear specific caches
cp.clear_cave_client_cache()
cp.clear_cloudvolume_cache()
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**

   ```text
   Error: No valid token found
   ```

   **Solution**: Run `cp.generate_cave_token(save=True)` to authenticate

2. **Network Timeouts**

   ```text
   Error: Request timeout
   ```

   **Solution**: Check internet connection and try again. Some operations may take time with large datasets.

3. **Memory Errors**

   ```text
   MemoryError: Unable to allocate array
   ```

   **Solution**: Process data in smaller batches or increase available memory

4. **Version Conflicts**

   ```text
   Error: Incompatible dependency versions
   ```

   **Solution**: Create a fresh environment and reinstall

### Getting Help

If you encounter issues:

1. **Check the logs**: Set `cp.set_logging_level("DEBUG")` for detailed output
2. **Update CRANTpy**: `pip install --upgrade crantpy`
3. **Clear caches**: `cp.clear_all_caches()`
4. **Report bugs**: Create an issue on [GitHub](https://github.com/Social-Evolution-and-Behavior/crantpy/issues)

### System-Specific Notes

#### macOS

- You may need to install Xcode command line tools: `xcode-select --install`

#### Windows

- Consider using Windows Subsystem for Linux (WSL) for better compatibility

#### High-Performance Computing (HPC)

- CRANTpy works well on HPC systems
- Consider using `--user` flag for pip installations
- Set appropriate cache directories in shared filesystems

## Next Steps

Once installation is complete, check out:

- [Quick Start Guide](quickstart.md) - Basic usage examples
- [User Guide](user-guide.md) - Comprehensive tutorials
- [API Reference](api/modules.rst) - Detailed function documentation