import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'connections',
        'get_adjacency',
        'neurons',
    },
    submod_attrs={
        'connections': [
            'get_adjacency',
            'get_synapses',
        ],
        'get_adjacency': [
            'get_adjacency',
        ],
        'neurons': [
            'F',
            'NeuronCriteria',
            'T',
            'get_annotations',
        ],
    },
)

__all__ = ['F', 'NeuronCriteria', 'T', 'connections', 'get_adjacency',
           'get_annotations', 'get_synapses', 'neurons']
