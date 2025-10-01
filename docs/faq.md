# Frequently Asked Questions

## General Questions

### What is CRANTpy?

CRANTpy is a Python library for accessing and analyzing the Clonal Raider ANT Brain (CRANTb) connectome dataset. It provides tools for querying neurons, accessing morphological data, and analyzing neural circuits in the ant brain.

### What does CRANT stand for?

CRANT stands for **Clonal Raider ANT**. The project focuses on mapping the neural connectome of the clonal raider ant (*Ooceraea biroi*), a species known for its unique reproductive strategy and social behavior.

### Who should use CRANTpy?

CRANTpy is designed for:
- Neuroscientists studying insect brains and behavior
- Researchers interested in connectomics and neural circuits
- Students learning about computational neuroscience
- Developers building tools for neural data analysis

## Installation and Setup

### What are the system requirements?

- **Python**: 3.10 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: At least 8GB RAM recommended
- **Storage**: ~1GB for cache and temporary files
- **Internet**: Required for accessing CAVE datasets

### How do I install CRANTpy?

The easiest way is using pip:

```bash
pip install crantpy
```

For development:

```bash
git clone https://github.com/Social-Evolution-and-Behavior/crantpy.git
cd crantpy
poetry install
```

### Why do I need authentication?

CRANTpy accesses data through the CAVE (Connectome Annotation Versioning Engine) service, which requires authentication to ensure data security and track usage. The authentication is free but required.

### How do I get a CAVE token?

Run this code and follow the browser prompts:

```python
import crantpy as cp
cp.generate_cave_token(save=True)
```

### My token expired. What do I do?

Simply generate a new token:

```python
import crantpy as cp
cp.generate_cave_token(save=True)
```

## Data and Datasets

### What datasets are available?

CRANTpy provides access to two main datasets:

- **`latest`**: The most recent stable release (recommended)
- **`sandbox`**: Development version with newest annotations (may be unstable)

### How many neurons are in the dataset?

The exact number varies by dataset version, but typically contains thousands of annotated neurons across different brain regions and cell types.

### What information is available for each neuron?

Each neuron includes:
- **Morphology**: 3D mesh and skeleton reconstructions
- **Annotations**: Cell type, brain region, functional properties
- **Connectivity**: Synaptic connections (when available)
- **Metadata**: Proofreading status, quality metrics

### Can I access raw imaging data?

CRANTpy focuses on processed annotations and reconstructions. For raw imaging data, you would need to access the original CAVE datasets directly.

## Usage Questions

### How do I find neurons of a specific type?

Use the `NeuronCriteria` class:

```python
import crantpy as cp

# Find Kenyon cells
kenyon_cells = cp.NeuronCriteria(cell_class='kenyon_cell')
kc_ids = kenyon_cells.get_roots()
```

### How do I get the 3D shape of a neuron?

You can get both meshes and skeletons:

```python
# Get mesh (detailed surface)
mesh = cp.get_mesh_neuron(neuron_id)

# Get skeleton (centerline)
skeleton = cp.get_skeletons(neuron_id)
```

### How do I visualize neurons?

CRANTpy integrates with navis for 3D visualization:

```python
# Visualize skeleton
skeleton.plot3d(backend='plotly')

# Visualize mesh
mesh.plot3d(backend='plotly')
```

### How do I analyze multiple neurons at once?

Most functions accept lists of IDs:

```python
# Get multiple skeletons
neuron_ids = [id1, id2, id3]
skeletons = cp.get_skeletons(neuron_ids)

# Analyze population
for skel in skeletons:
    print(f"Neuron {skel.id}: {skel.cable_length:.2f} nm")
```

### How do I filter by multiple criteria?

Use multiple parameters in `NeuronCriteria`:

```python
# Multiple criteria (AND logic)
left_pns = cp.NeuronCriteria(
    cell_class='projection_neuron',
    side='L',
    region='antennal_lobe'
)
```

## Performance and Optimization

### Why is my first query slow?

CRANTpy downloads and caches data on first access. Subsequent queries will be much faster thanks to intelligent caching.

### How can I speed up large analyses?

1. **Use caching**: Don't clear caches unnecessarily
2. **Batch processing**: Process neurons in groups
3. **Selective queries**: Only get the data you need
4. **Memory management**: Clear large objects when done

```python
# Example of efficient batch processing
all_ids = cp.get_proofread_neurons()
batch_size = 100

for i in range(0, len(all_ids), batch_size):
    batch = all_ids[i:i+batch_size]
    skeletons = cp.get_skeletons(batch)
    # Process batch...
```

### Where is data cached?

- **Linux/macOS**: `~/.crantpy/`
- **Windows**: `%USERPROFILE%/.crantpy/`

### How do I clear the cache?

```python
import crantpy as cp

# Clear all caches
cp.clear_all_caches()

# Clear specific caches
cp.clear_cave_client_cache()
```

### My analysis is using too much memory. What can I do?

1. Process neurons one at a time for large meshes
2. Use skeletons instead of meshes when possible
3. Clear objects from memory when done
4. Increase your system's virtual memory

## Troubleshooting

### I get "No valid token found" error

You need to authenticate first:

```python
import crantpy as cp
cp.generate_cave_token(save=True)
```

### I get "Connection timeout" errors

This usually indicates network issues. Try:
1. Check your internet connection
2. Wait and retry (servers may be busy)
3. Use smaller batch sizes

### My neuron IDs seem to be outdated

Neuron IDs can change due to proofreading. Update them:

```python
# Check if IDs are current
current_status = cp.is_latest_roots(your_ids)

# Update a DataFrame with new IDs
updated_df = cp.update_ids(your_dataframe)
```

### I get "MemoryError" when loading large neurons

Try:
1. Process neurons individually instead of in batches
2. Use skeletons instead of meshes
3. Increase system memory or use a machine with more RAM

### CRANTpy is not finding neurons I expect

Check:
1. Your spelling of annotation fields
2. Case sensitivity (use `case=True` if needed)
3. Dataset version (`latest` vs `sandbox`)
4. Whether you need regex matching

```python
# Debug your query
criteria = cp.NeuronCriteria(your_criteria)
annotations = cp.get_annotations(criteria)
print(f"Found {len(annotations)} neurons")
```

## Advanced Usage

### How do I use regular expressions in queries?

Set `regex=True`:

```python
# Match multiple cell types
kc_subtypes = cp.NeuronCriteria(
    cell_type='KC_.*',  # Matches KC_a, KC_b, etc.
    regex=True
)
```

### How do I work with different dataset versions?

Specify the dataset:

```python
# Latest dataset
latest_neurons = cp.NeuronCriteria(
    cell_class='kenyon_cell',
    dataset='latest'
)

# Sandbox dataset
sandbox_neurons = cp.NeuronCriteria(
    cell_class='kenyon_cell',
    dataset='sandbox'
)
```

### How do I access raw CAVE client functionality?

Get the client directly:

```python
client = cp.get_cave_client()
# Now you can use any CAVE client methods
```

### Can I contribute to CRANTpy?

Yes! CRANTpy is open source. See the [GitHub repository](https://github.com/Social-Evolution-and-Behavior/crantpy) for contribution guidelines.

## Data Interpretation

### What do the confidence scores mean in ID updates?

Confidence scores (0-1) indicate how certain the system is that an old ID maps to a new ID:
- **1.0**: Perfect match (100% supervoxel overlap)
- **0.5**: Partial match (50% overlap)
- **0.0**: No overlap (completely different neuron)

### What does "proofread" mean?

A proofread neuron has been manually reviewed and corrected by expert annotators. Proofread neurons are generally more reliable for analysis.

### What are the different proofreading statuses?

Common statuses include:
- `BACKBONE_PROOFREAD`: Main structure reviewed
- `PRELIM_PROOFREAD`: Initial review completed
- `COMPLETED`: Fully proofread and finalized

### How accurate are the morphological measurements?

Measurements depend on:
- **Proofreading status**: Proofread neurons are more accurate
- **Resolution**: Limited by imaging resolution (~4nm)
- **Reconstruction quality**: Varies by brain region

## Getting Help

### Where can I find more documentation?

- [User Guide](user-guide.md): Comprehensive tutorials
- [API Reference](api/modules.rst): Detailed function documentation
- [Tutorial Notebook](tutorial.ipynb): Interactive examples

### How do I report bugs or request features?

Use the [GitHub Issues](https://github.com/Social-Evolution-and-Behavior/crantpy/issues) page.

### How do I get help with my analysis?

1. Check the documentation and examples
2. Search existing GitHub issues
3. Post a question on GitHub Discussions
4. Contact the development team

### Is there a mailing list or forum?

Use [GitHub Discussions](https://github.com/Social-Evolution-and-Behavior/crantpy/discussions) for community support and questions.

## Citation and Usage

### How do I cite CRANTpy in my research?

```bibtex
@software{crantpy,
  title={CRANTpy: Python Access to Clonal Raider ANT Brain Datasets},
  author={CRANTb Community},
  url={https://github.com/Social-Evolution-and-Behavior/crantpy},
  year={2024}
}
```

### Are there any usage restrictions?

CRANTpy is open source under the GNU GPL v3.0 license. The data access follows CAVE usage policies. Always cite both CRANTpy and the original CRANT dataset in publications.

### Can I use CRANTpy for commercial projects?

Check the license terms and data usage policies. For commercial use, contact the dataset providers directly.