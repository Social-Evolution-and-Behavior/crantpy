# User Guide

This comprehensive guide covers all major features of CRANTpy and provides detailed examples for common use cases.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Authentication & Setup](#authentication--setup)
3. [Querying Neurons](#querying-neurons)
4. [Working with Morphology](#working-with-morphology)
5. [Data Management](#data-management)
6. [Visualization](#visualization)
7. [Performance Optimization](#performance-optimization)
8. [Advanced Topics](#advanced-topics)

## Core Concepts

### Datasets

CRANTpy provides access to multiple dataset versions:

- **`latest`**: The most recent stable release (recommended for most users)
- **`sandbox`**: Development version with newest annotations (may be unstable)

```python
import crantpy as cp

# Set default dataset
cp.set_default_dataset("latest")

# Or specify per query
neurons_latest = cp.NeuronCriteria(cell_type="KC_a", dataset="latest")
neurons_sandbox = cp.NeuronCriteria(cell_type="KC_a", dataset="sandbox")
```

### Neuron Identifiers

CRANTpy works with several types of neuron identifiers:

- **Root IDs**: Primary identifiers for reconstructed neurons
- **Supervoxel IDs**: Fine-grained segmentation units
- **Nucleus IDs**: Cell body annotations

```python
# Convert between ID types
root_id = 576460752653449509
supervoxels = cp.roots_to_supervoxels(root_id)
back_to_root = cp.supervoxels_to_roots(supervoxels)

print(f"Root ID: {root_id}")
print(f"Supervoxels: {supervoxels[:5]}...")  # Show first 5
print(f"Converted back: {back_to_root}")
```

### Annotation Fields

Neurons are annotated with rich metadata stored in Seatable. Key fields include:

```python
# See all available annotation fields
fields = cp.NeuronCriteria.available_fields()
print("Available fields:")
for field in sorted(fields):
    print(f"  {field}")
```

Common annotation fields:
- `cell_type`: Specific neuron type (e.g., "KC_a", "PN_vm1")
- `cell_class`: Broad functional class (e.g., "kenyon_cell", "projection_neuron")
- `region`: Brain region (e.g., "mushroom_body", "antennal_lobe")
- `side`: Left/Right hemisphere ("L", "R", or "M" for midline)
- `tract`: Anatomical tract (e.g., "mALT", "PCT")
- `status`: Proofreading status
- `proofread`: Boolean indicating if neuron is proofread

## Authentication & Setup

### Initial Setup

```python
import crantpy as cp

# Set logging level (optional)
cp.set_logging_level("INFO")

# Generate authentication token (one-time setup)
cp.generate_cave_token(save=True)
```

### Managing Tokens

```python
# Check current token
current_token = cp.get_current_cave_token()
print(f"Current token: {current_token[:20]}...")

# Set a specific token
cp.set_cave_token("your_token_here")

# Generate new token without saving
new_token = cp.generate_cave_token(save=False)
```

### CAVE Client Management

```python
# Get CAVE client
client = cp.get_cave_client()
print(f"Connected to: {client.datastack_name}")

# Get client for specific dataset
client_sandbox = cp.get_cave_client(dataset="sandbox")

# Clear client cache (forces reconnection)
cp.clear_cave_client_cache()

# Validate client connection
cp.validate_cave_client(client)
```

## Querying Neurons

### Basic Queries

The `NeuronCriteria` class is the primary interface for querying neurons:

```python
# Query by single criterion
kenyon_cells = cp.NeuronCriteria(cell_class="kenyon_cell")

# Query by multiple criteria (AND logic)
left_pns = cp.NeuronCriteria(
    cell_class="projection_neuron",
    side="L"
)

# Get root IDs
kc_ids = kenyon_cells.get_roots()
print(f"Found {len(kc_ids)} Kenyon cells")
```

### Advanced Filtering

```python
# Use regex patterns
pn_types = cp.NeuronCriteria(
    cell_type="PN_.*",  # Matches PN_vm1, PN_dm1, etc.
    regex=True
)

# Case-sensitive matching
specific_type = cp.NeuronCriteria(
    cell_type="kc_a",  # Will NOT match "KC_a"
    case=True,
    regex=True
)

# Match multiple values (OR logic within field)
multiple_regions = cp.NeuronCriteria(
    region=["mushroom_body", "antennal_lobe"]
)

# Exact vs substring matching
exact_match = cp.NeuronCriteria(
    cell_type="KC_a",
    exact=True  # Only exact matches
)

partial_match = cp.NeuronCriteria(
    cell_type="KC",
    exact=False  # Matches KC_a, KC_b, etc.
)
```

### Working with Lists in Annotations

Some annotation fields contain lists (e.g., multiple statuses). Control matching behavior:

```python
# Match ANY status in the list (default)
any_status = cp.NeuronCriteria(
    status=["BACKBONE_PROOFREAD", "PRELIM_PROOFREAD"],
    match_all=False  # default
)

# Match ALL statuses in the list
all_statuses = cp.NeuronCriteria(
    status=["BACKBONE_PROOFREAD", "COMPLETED"],
    match_all=True
)
```

### Querying Proofread Neurons

```python
# Get all proofread neurons (pre-filtered)
proofread_ids = cp.get_proofread_neurons()

# Query proofread neurons with additional criteria
proofread_kcs = cp.NeuronCriteria(
    cell_class="kenyon_cell",
    proofread_only=True
)

# Check if specific neurons are proofread
kc_criteria = cp.NeuronCriteria(cell_class="kenyon_cell")
is_proofread = cp.is_proofread(kc_criteria)
print(f"Proofread status: {is_proofread}")
```

### Getting Annotations

```python
# Get full annotation table
all_annotations = cp.get_all_seatable_annotations()
print(f"Total annotations: {len(all_annotations)}")

# Get annotations for specific criteria
pn_criteria = cp.NeuronCriteria(cell_class="projection_neuron")
pn_annotations = cp.get_annotations(pn_criteria)

# Specify which fields to include
specific_fields = cp.get_annotations(
    pn_criteria,
    fields=["root_id", "cell_type", "region", "side"]
)
```

## Working with Morphology

### Neuron Meshes

```python
# Get single neuron mesh
root_id = 576460752653449509
mesh = cp.get_mesh_neuron(root_id)

# Mesh properties
print(f"Vertices: {len(mesh.vertices)}")
print(f"Faces: {len(mesh.faces)}")
print(f"Volume: {mesh.volume:.2f} cubic microns")

# Get multiple meshes
root_ids = [576460752653449509, 576460752656385772]
meshes = [cp.get_mesh_neuron(rid) for rid in root_ids]
```

### Neuron Skeletons

```python
# Get skeleton for single neuron
skeleton = cp.get_skeletons(root_id)

# Skeleton properties
print(f"Nodes: {len(skeleton.nodes)}")
print(f"Cable length: {skeleton.cable_length:.2f} nm")
print(f"Branch points: {skeleton.n_branch_points}")
print(f"End points: {skeleton.n_endpoints}")

# Get skeletons for multiple neurons
skeletons = cp.get_skeletons(root_ids)

# Access individual skeletons
for i, skel in enumerate(skeletons):
    print(f"Neuron {i}: {skel.cable_length:.0f} nm")
```

### Morphological Analysis

```python
import pandas as pd

# Analyze morphology of neuron population
kc_ids = cp.NeuronCriteria(cell_class="kenyon_cell").get_roots()[:20]
kc_skeletons = cp.get_skeletons(kc_ids)

# Collect morphological measurements
morphology_data = []
for skel in kc_skeletons:
    morphology_data.append({
        'neuron_id': skel.id,
        'cable_length': skel.cable_length,
        'n_branch_points': skel.n_branch_points,
        'n_endpoints': skel.n_endpoints,
        'tortuosity': skel.tortuosity
    })

morphology_df = pd.DataFrame(morphology_data)
print(morphology_df.describe())

# Statistical analysis
print(f"Mean cable length: {morphology_df['cable_length'].mean():.2f} nm")
print(f"Std cable length: {morphology_df['cable_length'].std():.2f} nm")
```

### Soma Detection

```python
# Detect soma from skeleton
soma_loc = cp.detect_soma(skeleton)
print(f"Soma location: {soma_loc}")

# Detect soma from mesh
mesh_soma = cp.detect_soma_mesh(mesh)
print(f"Mesh soma: {mesh_soma}")

# Get soma from annotations (if available)
annotation_soma = cp.get_soma_from_annotations(root_id)
```

### L2 Chunk Analysis

For detailed morphological analysis at the L2 resolution:

```python
# Get L2 skeleton
l2_skeleton = cp.get_l2_skeleton(root_id)

# Get L2 graph
l2_graph = cp.get_l2_graph(root_id)

# Get L2 chunk information
l2_info = cp.get_l2_info(root_id)

# Get L2 meshes
l2_meshes = cp.get_l2_meshes(root_id)
```

## Data Management

### ID Validation and Updates

Over time, neuron IDs may change due to proofreading. CRANTpy helps manage these changes:

```python
# Check if IDs are current
old_ids = [576460752653449509, 576460752656385772]
are_current = cp.is_latest_roots(old_ids)
print(f"IDs current: {are_current}")

# Update a DataFrame with potentially outdated IDs
df = cp.get_all_seatable_annotations()
updated_df = cp.update_ids(df, dataset="latest")

# Check what changed
changed = updated_df[updated_df['changed'] == True]
print(f"{len(changed)} IDs were updated")

# See the changes
print(updated_df[['old_id', 'new_id', 'confidence', 'changed']].head())
```

### Caching System

CRANTpy uses intelligent caching to improve performance:

```python
# Clear all caches
cp.clear_all_caches()

# Clear specific caches
cp.clear_cave_client_cache()
cp.clear_cloudvolume_cache()

# Force refresh of annotations
fresh_annotations = cp.get_all_seatable_annotations(clear_cache=True)

# Force refresh of specific query
fresh_query = cp.NeuronCriteria(
    cell_class="kenyon_cell",
    clear_cache=True
)
```

### Working with CloudVolume

```python
# Get CloudVolume object
volume = cp.get_cloudvolume(dataset="latest")

# Access segmentation directly
seg_id = 576460752653449509
segmentation = volume[seg_id]

# Get raw mesh data
raw_mesh = volume.mesh.get(seg_id)[seg_id]
```

## Visualization

### Basic Plotting

```python
import plotly.graph_objects as go

# Plot skeleton
skeleton = cp.get_skeletons(root_id)
skeleton.plot3d(backend='plotly')

# Plot mesh  
mesh = cp.get_mesh_neuron(root_id)
mesh.plot3d(backend='plotly')

# Plot multiple neurons
skeletons = cp.get_skeletons(root_ids[:3])
for skel in skeletons:
    skel.plot3d(backend='plotly', inline=False)
```

### Custom Visualization

```python
# Create custom plot
fig = go.Figure()

# Add skeleton
skel_coords = skeleton.nodes[['x', 'y', 'z']].values
fig.add_trace(go.Scatter3d(
    x=skel_coords[:, 0],
    y=skel_coords[:, 1], 
    z=skel_coords[:, 2],
    mode='markers',
    name='Skeleton nodes'
))

# Add soma
soma = cp.detect_soma(skeleton)
if soma is not None:
    fig.add_trace(go.Scatter3d(
        x=[soma[0]],
        y=[soma[1]],
        z=[soma[2]],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Soma'
    ))

fig.show()
```

## Performance Optimization

### Batch Processing

```python
# Process large datasets in batches
all_ids = cp.get_proofread_neurons()
batch_size = 100

results = []
for i in range(0, len(all_ids), batch_size):
    batch = all_ids[i:i+batch_size]
    
    # Process batch
    batch_skeletons = cp.get_skeletons(batch)
    
    # Extract data
    for skel in batch_skeletons:
        results.append({
            'id': skel.id,
            'cable_length': skel.cable_length
        })
    
    print(f"Processed {i+len(batch)}/{len(all_ids)} neurons")

results_df = pd.DataFrame(results)
```

### Memory Management

```python
# For large meshes, process one at a time
for root_id in large_neuron_list:
    mesh = cp.get_mesh_neuron(root_id)
    
    # Process mesh
    volume = mesh.volume
    
    # Clear from memory
    del mesh
```

### HTTP Session Optimization

```python
# Use HTTP sessions for better performance with large ID lists
large_id_list = cp.get_proofread_neurons()

# More efficient for large lists
current_status = cp.is_latest_roots(
    large_id_list, 
    use_http_session=True
)
```

## Advanced Topics

### Custom SQL Queries

```python
# Create custom SQL for Seatable
sql_query = cp.create_sql_query(
    table_name="CRANTb_meta",
    fields=["root_id", "cell_type", "region"],
    condition="cell_class = 'kenyon_cell' AND side = 'L'",
    limit=100
)
print(sql_query)
```

### Working with Multiple Datasets

```python
# Compare across datasets
latest_kcs = cp.NeuronCriteria(
    cell_class="kenyon_cell",
    dataset="latest"
).get_roots()

sandbox_kcs = cp.NeuronCriteria(
    cell_class="kenyon_cell", 
    dataset="sandbox"
).get_roots()

print(f"Latest: {len(latest_kcs)} KCs")
print(f"Sandbox: {len(sandbox_kcs)} KCs")

# Find differences
latest_set = set(latest_kcs)
sandbox_set = set(sandbox_kcs)

only_in_latest = latest_set - sandbox_set
only_in_sandbox = sandbox_set - latest_set

print(f"Only in latest: {len(only_in_latest)}")
print(f"Only in sandbox: {len(only_in_sandbox)}")
```

### Error Handling

```python
from crantpy.utils.exceptions import NoMatchesError, FilteringError

try:
    # This might fail if no matches
    rare_neurons = cp.NeuronCriteria(
        cell_type="very_rare_type"
    ).get_roots()
    
except NoMatchesError as e:
    print(f"No neurons found: {e}")
    
except FilteringError as e:
    print(f"Filtering failed: {e}")
```

### Configuration Management

```python
# Check current configuration
print(f"Default dataset: {cp.CRANT_DEFAULT_DATASET}")
print(f"Valid datasets: {cp.CRANT_VALID_DATASETS}")
print(f"Cache directory: {cp.CRANT_CACHE_DIR}")

# Modify configuration
cp.set_default_dataset("sandbox")

# Access datastacks
datastacks = cp.get_cave_datastacks()
print(f"Available datastacks: {datastacks}")
```

## Best Practices

1. **Use caching**: Don't clear caches unnecessarily
2. **Batch processing**: Process large datasets in chunks
3. **Memory management**: Clear large objects when done
4. **Error handling**: Wrap queries in try-catch blocks
5. **Logging**: Use appropriate logging levels for debugging
6. **ID validation**: Check for updated IDs regularly
7. **Dataset selection**: Use "latest" for production, "sandbox" for exploration

## Common Patterns

### Pattern 1: Population Analysis

```python
# 1. Query population
population = cp.NeuronCriteria(cell_class="projection_neuron")

# 2. Get morphology
skeletons = cp.get_skeletons(population.get_roots())

# 3. Extract measurements
measurements = [
    {
        'id': s.id,
        'cable_length': s.cable_length,
        'branch_points': s.n_branch_points
    }
    for s in skeletons
]

# 4. Analyze
df = pd.DataFrame(measurements)
summary = df.describe()
```

### Pattern 2: Cross-Region Comparison

```python
# Compare neurons across regions
regions = ["mushroom_body", "antennal_lobe", "central_complex"]

region_data = {}
for region in regions:
    neurons = cp.NeuronCriteria(region=region)
    region_data[region] = {
        'count': len(neurons.get_roots()),
        'cell_types': len(cp.get_annotations(neurons)['cell_type'].unique())
    }

for region, data in region_data.items():
    print(f"{region}: {data['count']} neurons, {data['cell_types']} types")
```

### Pattern 3: Longitudinal Analysis

```python
# Track changes between dataset versions
latest_neurons = cp.get_all_seatable_annotations(dataset="latest")
sandbox_neurons = cp.get_all_seatable_annotations(dataset="sandbox")

# Compare counts
print(f"Latest: {len(latest_neurons)} neurons")
print(f"Sandbox: {len(sandbox_neurons)} neurons")
print(f"Difference: {len(sandbox_neurons) - len(latest_neurons)}")

# Find new annotations
latest_ids = set(latest_neurons['root_id'])
sandbox_ids = set(sandbox_neurons['root_id'])
new_ids = sandbox_ids - latest_ids
print(f"New in sandbox: {len(new_ids)} neurons")
```