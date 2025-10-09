import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'connections',
        'neurons',
    },
    submod_attrs={
        'connections': [
            'attach_synapses',
            'get_adjacency',
            'get_connectivity',
            'get_synapse_counts',
            'get_synapses',
            'logger',
        ],
        'neurons': [
            'NeuronCriteria',
            'get_annotations',
            'is_proofread',
            'parse_neuroncriteria',
        ],
    },
)

__all__ = ['NeuronCriteria', 'attach_synapses', 'connections', 'get_adjacency',
           'get_annotations', 'get_connectivity', 'get_synapse_counts',
           'get_synapses', 'is_proofread', 'logger', 'neurons',
           'parse_neuroncriteria']
