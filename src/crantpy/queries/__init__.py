import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'connections',
        'neurons',
        'neuropils',
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
        'neuropils': [
            'count_synapses_in_mesh',
            'get_synapses_in_mesh',
            'logger',
        ],
    },
)

__all__ = ['NeuronCriteria', 'attach_synapses', 'connections',
           'count_synapses_in_mesh', 'get_adjacency', 'get_annotations',
           'get_connectivity', 'get_synapse_counts', 'get_synapses',
           'get_synapses_in_mesh', 'is_proofread', 'logger', 'neurons',
           'neuropils', 'parse_neuroncriteria']
