# CRANTpy

> **Python Access to the Clonal Raider ANT Brain Connectome**

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://social-evolution-and-behavior.github.io/crantpy/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A comprehensive Python interface for exploring the complete neural connectome of the clonal raider ant (*Ooceraea biroi*).

**[📖 Documentation](https://social-evolution-and-behavior.github.io/crantpy/)** • **[🚀 Quick Start](https://social-evolution-and-behavior.github.io/crantpy/quickstart.html)** • **[💡 Examples](https://social-evolution-and-behavior.github.io/crantpy/tutorial.html)**

---

## ✨ Features

- **🔍 Intelligent Querying** - Filter neurons by anatomy, cell type, or custom criteria
- **🌐 CAVE Integration** - Direct access to the Connectome Annotation Versioning Engine  
- **📊 Rich Metadata** - Comprehensive neuron annotations and properties
- **🧮 Morphological Tools** - Analyze neuron meshes, skeletons, and connectivity
- **⚡ High Performance** - Optimized caching and batch processing

## 📦 Installation

```bash
pip install crantpy
```

**Requirements:** Python 3.10+

## 🚀 Quick Example

```python
import crantpy as cp

# One-time setup
cp.generate_cave_token(save=True)

# Query and analyze neurons
neurons = cp.NeuronCriteria(cell_class='kenyon_cell')
skeleton = cp.get_skeletons(neurons.get_roots()[0])

print(f"Cable length: {skeleton.cable_length:.0f} nm")
skeleton.plot3d()  # Interactive 3D visualization
```

## 📚 Learn More

Visit our **[comprehensive documentation](https://social-evolution-and-behavior.github.io/crantpy/)** for:

- 📖 **[Installation Guide](https://social-evolution-and-behavior.github.io/crantpy/installation.html)** - Detailed setup
- 🚀 **[Quick Start](https://social-evolution-and-behavior.github.io/crantpy/quickstart.html)** - Get running in 5 minutes  
- 💡 **[Tutorial](https://social-evolution-and-behavior.github.io/crantpy/tutorial.html)** - Interactive examples
- 📋 **[API Reference](https://social-evolution-and-behavior.github.io/crantpy/api/modules.html)** - Complete function docs
- ❓ **[FAQ](https://social-evolution-and-behavior.github.io/crantpy/faq.html)** - Common questions

## 🤝 Contributing

We welcome contributions! See our [contribution guidelines](CONTRIBUTING.md) or visit [GitHub Discussions](https://github.com/Social-Evolution-and-Behavior/crantpy/discussions).

## 📝 Citation

```bibtex
@software{crantpy,
  title={CRANTpy: Python Access to Clonal Raider ANT Brain Datasets},
  author={CRANTb Community},
  url={https://github.com/Social-Evolution-and-Behavior/crantpy},
  year={2025}
}
```

## 📄 License

GNU General Public License v3.0 - see [LICENSE](LICENSE) for details.

---

Made with ❤️ by the CRANTb Community