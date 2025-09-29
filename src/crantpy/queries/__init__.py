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
        'neurons': [
            'NeuronCriteria',
            'get_annotations',
            'is_proofread',
        ],
    },
)
