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
<<<<<<< HEAD
=======
        'get_adjacency': [
            'get_adjacency',
        ],
>>>>>>> 57865ba6a93806b5c525642ab5ddaa145a275f89
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
