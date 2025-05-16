import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'fetch_connections',
        'fetch_neurons',
    },
    submod_attrs={
        'fetch_neurons': [
            'F',
            'NeuronCriteria',
            'T',
            'get_annotations',
            'parse_neuroncriteria',
        ],
    },
)

__all__ = ['F', 'NeuronCriteria', 'T', 'fetch_connections', 'fetch_neurons',
           'get_annotations', 'parse_neuroncriteria']
