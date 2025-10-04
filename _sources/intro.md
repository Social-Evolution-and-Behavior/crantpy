# CRANTpy: Python Access to Clonal Raider ANT Brain Datasets

**CRANTpy** is a Python library providing streamlined access to the Clonal Raider ANT (CRANT) brain connectome datasets. It offers a comprehensive interface for querying neurons, accessing morphological data, and analyzing neural circuits in the ant brain.

## 🧠 About the CRANT Project

The Clonal Raider ANT Brain (CRANTb) project is a large-scale effort to map the complete neural connectome of the clonal raider ant (*Ooceraea biroi*). This dataset represents one of the first comprehensive invertebrate brain connectomes available after the fruitfly, providing unprecedented insights into the neural basis of social behavior in insects.

![CRANT Brain Dataset Overview](crant.png)

## ✨ Key Features

- **🔍 Flexible Neuron Querying**: Filter neurons by anatomical region, cell type, tract, or custom criteria
- **🌐 CAVE Integration**: Direct access to the Connectome Annotation Versioning Engine
- **📊 Rich Metadata**: Access to comprehensive neuron annotations and properties
- **🧮 Morphological Analysis**: Tools for analyzing neuron meshes and skeletons
- **⚡ Optimized Performance**: Intelligent caching and batch processing for large-scale analyses
- **🔄 Version Control**: Support for different dataset versions (latest, sandbox)


## 📖 Documentation Sections

```{tableofcontents}
```

## 🛠️ Installation

CRANTpy requires Python 3.10+ and can be installed using pip:

```bash
pip install crantpy
```

For development installations:

```bash
git clone https://github.com/Social-Evolution-and-Behavior/crantpy.git
cd crantpy
poetry install
```

## 🔗 Sister Projects

For R users, check out our sister project **crantr** which provides R-based access to the same CRANT datasets: [flyconnectome.github.io/crantr](https://flyconnectome.github.io/crantr/)

## 🤝 Contributing

We welcome contributions! Please see our [GitHub repository](https://github.com/Social-Evolution-and-Behavior/crantpy) for guidelines on how to contribute to the project.

## 🙏 Acknowledgements

CRANTpy is heavily derived from the excellent work by **Dr. Philipp Schlegel** and the FlyWire project. See our [full acknowledgements](acknowledgements.md) for complete contributor information.

## 📄 Citation

If you use CRANTpy in your research, please cite:

```bibtex
@software{crantpy,
  title={CRANTpy: Python Access to Clonal Raider ANT Brain Datasets},
  author={CRANTb Community},
  url={https://github.com/Social-Evolution-and-Behavior/crantpy},
  year={2025}
}
```

**Note:** CRANTpy is built upon [fafbseg-py](https://github.com/flyconnectome/fafbseg-py) by Dr. Philipp Schlegel. Please try to also cite the [FlyWire project papers](acknowledgements.html#flywire-project-citations) when using this tool.

## 📞 Support

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/Social-Evolution-and-Behavior/crantpy/issues)
- **Discussions**: Join the community discussion on [GitHub Discussions](https://github.com/Social-Evolution-and-Behavior/crantpy/discussions)
