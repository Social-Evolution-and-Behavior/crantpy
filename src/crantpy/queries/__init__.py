import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'connections',
        'neurons',
    },
    submod_attrs={
        'neurons': [
            'NeuronCriteria',
            'get_annotations',
        ],
    },
)

__all__ = ['NeuronCriteria', 'connections', 'get_annotations', 'neurons']
