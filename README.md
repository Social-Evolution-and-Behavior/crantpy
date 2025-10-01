# CRANTpy

## Python Access to the Clonal Raider ANT (CRANT) Brain Datasets

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://social-evolution-and-behavior.github.io/crantpy/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

CRANTpy is a Python library providing streamlined access to the Clonal Raider ANT Brain (CRANTb) connectome datasets. It offers a comprehensive interface for querying neurons, accessing morphological data, and analyzing neural circuits in the ant brain.

## 🧠 About the CRANT Project

The Clonal Raider ANT Brain (CRANTb) project is a large-scale effort to map the complete neural connectome of the clonal raider ant (*Ooceraea biroi*). This dataset represents one of the most complete invertebrate brain connectomes available, providing unprecedented insights into the neural basis of social behavior in insects.

## ✨ Key Features

- **🔍 Flexible Neuron Querying**: Filter neurons by anatomical region, cell type, tract, or custom criteria
- **🌐 CAVE Integration**: Direct access to the Connectome Annotation Versioning Engine
- **📊 Rich Metadata**: Access to comprehensive neuron annotations and properties
- **🧮 Morphological Analysis**: Tools for analyzing neuron meshes and skeletons
- **⚡ Optimized Performance**: Intelligent caching and batch processing for large-scale analyses
- **🔄 Version Control**: Support for different dataset versions (latest, sandbox)

## 🚀 Quick Start

```python
import crantpy as cp

# Set up authentication (one-time setup)
cp.generate_cave_token(save=True)

# Query neurons by cell type
ol_neurons = cp.NeuronCriteria(cell_class='olfactory_projection_neuron')
root_ids = ol_neurons.get_roots()

# Get morphological data
mesh = cp.get_mesh_neuron(root_ids[0])
skeleton = cp.get_skeletons(root_ids[0])

# Access annotations
annotations = cp.get_annotations(ol_neurons)
```

## 🛠️ Installation

### Requirements

- Python 3.10 or higher
- Internet connection for accessing CAVE datasets

### Install from PyPI

```bash
pip install crantpy
```

### Install from Source

```bash
git clone https://github.com/Social-Evolution-and-Behavior/crantpy.git
cd crantpy
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/Social-Evolution-and-Behavior/crantpy.git
cd crantpy
poetry install
```

## 📖 Documentation

- **[Installation Guide](https://social-evolution-and-behavior.github.io/crantpy/installation.html)**: Detailed setup instructions
- **[Quick Start](https://social-evolution-and-behavior.github.io/crantpy/quickstart.html)**: Get up and running in minutes
- **[User Guide](https://social-evolution-and-behavior.github.io/crantpy/user-guide.html)**: Comprehensive tutorials and examples
- **[Tutorial Notebook](https://social-evolution-and-behavior.github.io/crantpy/tutorial.html)**: Interactive Jupyter notebook
- **[API Reference](https://social-evolution-and-behavior.github.io/crantpy/api/modules.html)**: Complete function documentation
- **[FAQ](https://social-evolution-and-behavior.github.io/crantpy/faq.html)**: Common questions and solutions

## 💡 Usage Examples

### Query Neurons by Criteria

```python
import crantpy as cp

# Find Kenyon cells in the mushroom body
kenyon_cells = cp.NeuronCriteria(
    cell_class='kenyon_cell',
    region='mushroom_body'
)

# Get projection neurons from the left side
left_pns = cp.NeuronCriteria(
    cell_class='projection_neuron',
    side='L'
)

# Use regex for complex matching
pn_subtypes = cp.NeuronCriteria(
    cell_type='PN_.*',  # Matches PN_vm1, PN_dm1, etc.
    regex=True
)
```

### Morphological Analysis

```python
# Get neuron skeletons for morphological analysis
neuron_ids = kenyon_cells.get_roots()[:10]
skeletons = cp.get_skeletons(neuron_ids)

# Analyze morphology
for skel in skeletons:
    print(f"Neuron {skel.id}:")
    print(f"  Cable length: {skel.cable_length:.2f} nm")
    print(f"  Branch points: {skel.n_branch_points}")
    print(f"  End points: {skel.n_endpoints}")
```

### 3D Visualization

```python
# Visualize neurons in 3D
skeleton = cp.get_skeletons(neuron_ids[0])
skeleton.plot3d(backend='plotly')

# Visualize mesh
mesh = cp.get_mesh_neuron(neuron_ids[0])
mesh.plot3d(backend='plotly')
```

### Population Analysis

```python
# Compare different neuron populations
cell_classes = ['kenyon_cell', 'projection_neuron', 'local_neuron']

for cell_class in cell_classes:
    neurons = cp.NeuronCriteria(cell_class=cell_class)
    count = len(neurons.get_roots())
    print(f"{cell_class}: {count} neurons")
```

## 🔧 Authentication

CRANTpy requires authentication to access the CAVE service:

1. **Generate token** (interactive, one-time setup):

   ```python
   import crantpy as cp
   cp.generate_cave_token(save=True)
   ```

2. **Verify connection**:

   ```python
   client = cp.get_cave_client()
   print(f"Connected to: {client.datastack_name}")
   ```

## 🧪 Dataset Versions

CRANTpy supports multiple dataset versions:

- **`latest`**: Most recent stable release (recommended)
- **`sandbox`**: Development version with newest annotations

```python
# Use specific dataset version
latest_neurons = cp.NeuronCriteria(cell_type='KC_a', dataset='latest')
sandbox_neurons = cp.NeuronCriteria(cell_type='KC_a', dataset='sandbox')
```

## 🤝 Contributing

We welcome contributions! Please see our [contribution guidelines](CONTRIBUTING.md) for details on:

- Reporting bugs
- Suggesting enhancements
- Submitting pull requests
- Code style and testing

## 📄 Citation

If you use CRANTpy in your research, please cite:

```bibtex
@software{crantpy,
  title={CRANTpy: Python Access to Clonal Raider ANT Brain Datasets},
  author={CRANTb Community},
  url={https://github.com/Social-Evolution-and-Behavior/crantpy},
  year={2024}
}
```

## 📞 Support

- **Documentation**: [https://social-evolution-and-behavior.github.io/crantpy/](https://social-evolution-and-behavior.github.io/crantpy/)
- **Issues**: [GitHub Issues](https://github.com/Social-Evolution-and-Behavior/crantpy/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Social-Evolution-and-Behavior/crantpy/discussions)

## 📜 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The CRANT project team for providing the connectome data
- The CAVE development team for the infrastructure
- The navis team for morphological analysis tools
- All contributors to the CRANTpy project
