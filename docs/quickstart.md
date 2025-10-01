# Quick Start Guide

This guide will get you up and running with CRANTpy in just a few minutes. If you haven't installed CRANTpy yet, please see the [Installation Guide](installation.md) first.

## Prerequisites

- CRANTpy installed (`pip install crantpy`)
- Internet connection for accessing CAVE datasets
- Python 3.10 or higher

## Basic Setup

```python
import crantpy as cp

# Set up logging to see progress
cp.set_logging_level("INFO")

# Generate and save authentication token (one-time setup)
cp.generate_cave_token(save=True)
```

## Your First Query

Let's start by finding some olfactory projection neurons:

```python
# Query neurons by cell class
ol_neurons = cp.NeuronCriteria(cell_class='olfactory_projection_neuron')

# Get the root IDs
root_ids = ol_neurons.get_roots()
print(f"Found {len(root_ids)} olfactory projection neurons")

# Get detailed annotations
annotations = cp.get_annotations(ol_neurons)
print(annotations.head())
```

## Working with Specific Neurons

```python
# Get neurons from a specific brain region
mb_neurons = cp.NeuronCriteria(region='mushroom_body')

# Combine multiple criteria (AND logic)
left_mb_pns = cp.NeuronCriteria(
    region='mushroom_body',
    side='L',
    cell_class='projection_neuron'
)

# Use regex for complex matching
kenyon_cells = cp.NeuronCriteria(
    cell_type='KC_.*',  # Matches KC_a, KC_b, etc.
    regex=True
)
```

## Accessing Morphological Data

### Getting Neuron Meshes

```python
# Get a single neuron mesh
root_id = root_ids[0]
mesh = cp.get_mesh_neuron(root_id)

# The mesh is a navis.MeshNeuron object
print(f"Mesh has {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")

# Visualize (requires plotly)
mesh.plot3d(backend='plotly')
```

### Getting Neuron Skeletons

```python
# Get skeleton for a neuron
skeleton = cp.get_skeletons(root_id)

# The skeleton is a navis.TreeNeuron object
print(f"Skeleton has {len(skeleton.nodes)} nodes")

# Basic morphological measurements
print(f"Cable length: {skeleton.cable_length:.2f} nm")
print(f"Number of branch points: {skeleton.n_branch_points}")

# Visualize skeleton
skeleton.plot3d(backend='plotly')
```

## Working with Neuron Sets

```python
# Get multiple neurons at once
sample_ids = root_ids[:5]  # First 5 neurons

# Batch processing for efficiency
skeletons = cp.get_skeletons(sample_ids)

# Process each skeleton
for i, skel in enumerate(skeletons):
    print(f"Neuron {i+1}: {skel.cable_length:.0f} nm cable length")
```

## Data Validation and Updates

```python
# Check if root IDs are current
current_df = cp.get_all_seatable_annotations()

# Update outdated IDs to current versions
updated_df = cp.update_ids(current_df, dataset="latest")

# Check which IDs changed
changed = updated_df[updated_df['changed'] == True]
print(f"{len(changed)} neurons had ID updates")
```

## Exploring the Dataset

### Available Annotation Fields

```python
# See all available fields for filtering
fields = cp.NeuronCriteria.available_fields()
print("Available fields:")
for field in fields[:10]:  # Show first 10
    print(f"  - {field}")
```

### Dataset Statistics

```python
# Get all proofread neurons
proofread = cp.get_proofread_neurons()
print(f"Total proofread neurons: {len(proofread)}")

# Explore cell types
all_annotations = cp.get_all_seatable_annotations()
cell_types = all_annotations['cell_type'].value_counts()
print("Top cell types:")
print(cell_types.head(10))
```

### Brain Region Analysis

```python
# Neurons by brain region
by_region = all_annotations['region'].value_counts()
print("Neurons by region:")
print(by_region.head())

# Left vs right side
by_side = all_annotations['side'].value_counts()
print("Neurons by side:")
print(by_side)
```

## Common Workflows

### Workflow 1: Morphological Analysis

```python
# 1. Query neurons of interest
pns = cp.NeuronCriteria(cell_class='projection_neuron', side='L')

# 2. Get their IDs and sample a few
pn_ids = pns.get_roots()[:10]

# 3. Get morphological data
skeletons = cp.get_skeletons(pn_ids)

# 4. Analyze morphology
import pandas as pd

morphology_data = []
for skel in skeletons:
    morphology_data.append({
        'neuron_id': skel.id,
        'cable_length': skel.cable_length,
        'n_branches': skel.n_branch_points,
        'n_endpoints': skel.n_endpoints
    })

morphology_df = pd.DataFrame(morphology_data)
print(morphology_df.describe())
```

### Workflow 2: Connectivity Analysis

```python
# 1. Get neurons from a pathway
malt_neurons = cp.NeuronCriteria(tract='mALT')
malt_ids = malt_neurons.get_roots()

# 2. Get their annotations
malt_annotations = cp.get_annotations(malt_neurons)

# 3. Analyze their properties
print("mALT tract neurons:")
print(f"Count: {len(malt_ids)}")
print(f"Cell types: {malt_annotations['cell_type'].unique()}")
print(f"Regions: {malt_annotations['region'].unique()}")
```

### Workflow 3: Cross-Dataset Comparison

```python
# Compare latest vs sandbox datasets
latest_neurons = cp.NeuronCriteria(
    cell_class='kenyon_cell',
    dataset='latest'
).get_roots()

sandbox_neurons = cp.NeuronCriteria(
    cell_class='kenyon_cell', 
    dataset='sandbox'
).get_roots()

print(f"Latest dataset: {len(latest_neurons)} Kenyon cells")
print(f"Sandbox dataset: {len(sandbox_neurons)} Kenyon cells")
```

## Performance Tips

### Caching

```python
# CRANTpy automatically caches results
# Clear cache if needed
cp.clear_all_caches()

# Or clear specific caches
cp.clear_cave_client_cache()
```

### Batch Processing

```python
# For large datasets, process in batches
all_ids = cp.get_proofread_neurons()

batch_size = 100
for i in range(0, len(all_ids), batch_size):
    batch = all_ids[i:i+batch_size]
    
    # Process batch
    results = cp.get_skeletons(batch)
    
    # Save results incrementally
    # ... your processing code here
```

### Memory Management

```python
# For large meshes, work with one at a time
for root_id in root_ids:
    mesh = cp.get_mesh_neuron(root_id)
    
    # Process mesh
    # ... your analysis code
    
    # Clear from memory
    del mesh
```

## Next Steps

Now that you've learned the basics, explore:

- **[User Guide](user-guide.md)**: Detailed tutorials and advanced usage
- **[API Reference](api/modules.rst)**: Complete function documentation  
- **[Examples](examples/)**: Real-world analysis examples
- **[FAQ](faq.md)**: Common questions and solutions

## Getting Help

- **Check logs**: Use `cp.set_logging_level("DEBUG")` for detailed output
- **Clear cache**: Try `cp.clear_all_caches()` if you encounter issues
- **GitHub Issues**: Report bugs at [github.com/Social-Evolution-and-Behavior/crantpy](https://github.com/Social-Evolution-and-Behavior/crantpy/issues)