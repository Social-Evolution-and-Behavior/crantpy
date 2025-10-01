# CRANTpy

> **Python Access to the Clonal Raider ANT Brain Connectome**

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://social-evolution-and-behavior.github.io/crantpy/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A comprehensive Python interface for exploring the complete neural connectome of the clonal raider ant (*Ooceraea biroi*).

**[Documentation](https://social-evolution-and-behavior.github.io/crantpy/)** • **[Quick Start](#-quick-start)** • **[Examples](#-examples)** • **[Support](#-support)**

---

## 🧠 About

The Clonal Raider ANT Brain (CRANTb) project represents one of the most complete invertebrate brain connectomes available, providing unprecedented insights into the neural basis of social behavior in insects. CRANTpy makes this rich dataset accessible through an intuitive Python interface.

## ✨ Features

- **🔍 Intelligent Querying** - Filter neurons by anatomy, cell type, or custom criteria
- **🌐 CAVE Integration** - Direct access to the Connectome Annotation Versioning Engine  
- **📊 Rich Metadata** - Comprehensive neuron annotations and properties
- **🧮 Morphological Tools** - Analyze neuron meshes, skeletons, and connectivity
- **⚡ High Performance** - Optimized caching and batch processing
- **🔄 Version Support** - Work with latest stable or development datasets

## � Installation

```bash
# Install from PyPI
pip install crantpy

# Or install from source
git clone https://github.com/Social-Evolution-and-Behavior/crantpy.git
cd crantpy
pip install -e .
```

**Requirements:** Python 3.10+, internet connection for CAVE access

## � Quick Start

```python
import crantpy as cp

# One-time authentication setup
cp.generate_cave_token(save=True)

# Query neurons by cell type
neurons = cp.NeuronCriteria(cell_class='kenyon_cell')
root_ids = neurons.get_roots()

# Get morphological data  
skeleton = cp.get_skeletons(root_ids[0])
mesh = cp.get_mesh_neuron(root_ids[0])

# Analyze properties
print(f"Cable length: {skeleton.cable_length:.0f} nm")
print(f"Branch points: {skeleton.n_branch_points}")
```

## 💡 Examples

### Find Specific Neurons

```python
# Kenyon cells in mushroom body
kenyon_cells = cp.NeuronCriteria(
    cell_class='kenyon_cell',
    region='mushroom_body'
)

# Left-side projection neurons  
left_pns = cp.NeuronCriteria(
    cell_class='projection_neuron',
    side='L'
)

# Pattern matching with regex
pn_subtypes = cp.NeuronCriteria(
    cell_type='PN_.*',  # Matches PN_vm1, PN_dm1, etc.
    regex=True
)
```

### Morphological Analysis

```python
# Batch analysis of neuron populations
neuron_ids = kenyon_cells.get_roots()[:10]
skeletons = cp.get_skeletons(neuron_ids)

for skel in skeletons:
    print(f"Neuron {skel.id}: {skel.cable_length:.0f} nm cable")
```

### 3D Visualization

```python
# Interactive 3D plots
skeleton.plot3d(backend='plotly')
mesh.plot3d(backend='plotly')
```

## � Documentation

| Resource | Description |
|----------|-------------|
| [**Installation Guide**](https://social-evolution-and-behavior.github.io/crantpy/installation.html) | Detailed setup instructions |
| [**Quick Start**](https://social-evolution-and-behavior.github.io/crantpy/quickstart.html) | Get up and running in minutes |
| [**Tutorial Notebook**](https://social-evolution-and-behavior.github.io/crantpy/tutorial.html) | Interactive Jupyter examples |
| [**API Reference**](https://social-evolution-and-behavior.github.io/crantpy/api/modules.html) | Complete function documentation |
| [**FAQ**](https://social-evolution-and-behavior.github.io/crantpy/faq.html) | Common questions and solutions |

## 🔐 Authentication

CRANTpy requires one-time authentication setup to access CAVE:

```python
import crantpy as cp

# Interactive setup (saves token automatically)
cp.generate_cave_token(save=True)

# Verify connection
client = cp.get_cave_client()
print(f"Connected to: {client.datastack_name}")
```

## 📊 Dataset Versions

Choose between dataset versions for your analysis:

- **`latest`** - Stable release (recommended for research)
- **`sandbox`** - Development version with newest annotations

```python
# Specify version in queries
stable_neurons = cp.NeuronCriteria(cell_type='KC_a', dataset='latest')
dev_neurons = cp.NeuronCriteria(cell_type='KC_a', dataset='sandbox')
```

## 🤝 Contributing

We welcome contributions! Here's how to get involved:

- 🐛 **Report bugs** via [GitHub Issues](https://github.com/Social-Evolution-and-Behavior/crantpy/issues)
- 💡 **Suggest features** in [GitHub Discussions](https://github.com/Social-Evolution-and-Behavior/crantpy/discussions)  
- 🔧 **Submit pull requests** with improvements
- 📖 **Improve documentation** and examples

See our [contribution guidelines](CONTRIBUTING.md) for detailed information.

## � Citation

If you use CRANTpy in your research, please cite:

```bibtex
@software{crantpy,
  title={CRANTpy: Python Access to Clonal Raider ANT Brain Datasets},
  author={CRANTb Community},
  url={https://github.com/Social-Evolution-and-Behavior/crantpy},
  year={2024}
}
```

## � Support

| Need Help? | Resource |
|------------|----------|
| 📖 **Documentation** | [social-evolution-and-behavior.github.io/crantpy](https://social-evolution-and-behavior.github.io/crantpy/) |
| 🐛 **Bug Reports** | [GitHub Issues](https://github.com/Social-Evolution-and-Behavior/crantpy/issues) |
| 💭 **Questions** | [GitHub Discussions](https://github.com/Social-Evolution-and-Behavior/crantpy/discussions) |

## � License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Special thanks to:

- The CRANT project team for the connectome data
- The CAVE development team for infrastructure  
- The navis team for morphological analysis tools
- All CRANTpy contributors and users

---

Made with ❤️ by the CRANTb Community
