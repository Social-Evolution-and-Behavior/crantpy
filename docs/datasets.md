# Datasets

CRANTpy provides access to multiple versions of the Clonal Raider ANT Brain connectome dataset.

## Available Datasets

### Latest (Recommended)
- **Version**: Latest release
- **Status**: Work in progress
- **Use case**: Research and analysis
- **Updates**: Regular updates with new annotations and corrections

```python
import crantpy as cp

# Use latest dataset (default)
neurons = cp.NeuronCriteria(cell_class='spiny_kenyon_cell', dataset='latest')
```

### Sandbox
- **Version**: Training Version
- **Status**: Proofreading Training and Testing
- **Use case**: Training new proofreaders and annotators
- **Updates**: Frequent updates but quality may vary

```python
# Use sandbox dataset
neurons = cp.NeuronCriteria(cell_class='spiny_kenyon_cell', dataset='sandbox')
```

### Static Releases
- **Version**: Specific past releases (e.g., v1.0, v1.5, v2.0)
- **Status**: Stable, no further updates
- **Use case**: Reproducible research requiring fixed datasets
- **Updates**: None

```python
# Use specific static release
neurons = cp.NeuronCriteria(cell_class='spiny_kenyon_cell', dataset='v1.0')
```

```{note}
There are NO static releases yet. This is a placeholder for future versions.
```

## Data Quality

### Proofreading Status

```{note}
Neurons in the latest dataset are at various stages of proofreading. Some neurons may be backbone proofread, while others are partially or not proofread at all. Always check the proofreading status of neurons before analysis.
```

### Synapse Annotations

Synapse annotations are continuously being improved. The latest dataset contains the most up-to-date synapse information, but some synapses may still be missing or incorrectly annotated.

## Version History

### Latest
- **Release date**: Continuous updates
- **Changes**: Ongoing proofreading and annotation

### Stable Releases
- **v1.0**: Initial public release is planned for 2026

### Previous Versions
No deprecated releases yet.

## Citation

This will be updated once we have a stable release.

<!-- ```bibtex
@software{crantpy,
  title={CRANTpy: Python Access to Clonal Raider ANT Brain Datasets},
  author={CRANTb Community},
  url={https://social-evolution-and-behavior.github.io/crantpy/},
  year={2026}
}
``` -->

## Support

For dataset-related questions:
- 📧 Email: [crantb-support@example.com](mailto:crantb-support@example.com)
- 💬 Discussions: [GitHub Discussions](https://github.com/Social-Evolution-and-Behavior/crantpy/discussions)
- 🐛 Issues: [GitHub Issues](https://github.com/Social-Evolution-and-Behavior/crantpy/issues)