import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'connections',
        'neurons',
    },
    submod_attrs={
        'neurons': [
            'F',
            'NeuronCriteria',
            'T',
            'get_annotations',
        ],
    },
)

__all__ = ['F', 'NeuronCriteria', 'T', 'connections', 'get_annotations',
           'neurons']
