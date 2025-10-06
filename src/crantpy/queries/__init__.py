import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'connections',
        'neurons',
    },
    submod_attrs={
        'connections': [
            'get_adjacency',
            'get_connectivity',
            'get_synapse_counts',
            'get_synapses',
        ],
        'neurons': [
            'NeuronCriteria',
            'get_annotations',
            'is_proofread',
            'parse_neuroncriteria',
        ],
    },
)

__all__ = ['NeuronCriteria', 'connections', 'get_adjacency', 'get_annotations',
           'get_connectivity', 'get_synapse_counts', 'get_synapses',
           'is_proofread', 'neurons', 'parse_neuroncriteria']
