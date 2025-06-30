import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={},
    submod_attrs={
        'skeletonize': [
            'skeletonize_neuron',
            'skeletonize_neurons_parallel',
            'get_skeletons',
        ],
    },
)

__all__ = ['skeletonize_neuron', 'skeletonize_neurons_parallel', 'get_skeletons']
