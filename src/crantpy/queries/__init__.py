import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'fetch_connections',
        'fetch_neurons',
    },
    submod_attrs={
        'fetch_neurons': [
            'NeuronCriteria',
            'get_annotations',
        ],
    },
)

__all__ = ['NeuronCriteria', 'fetch_connections', 'fetch_neurons',
           'get_annotations']
