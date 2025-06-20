import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'visualization',
    },
    submod_attrs={
        'visualization': [
            'find_anchor_loc',
            'get_l2_chunk_info',
            'get_l2_info',
        ],
    },
)

__all__ = ['find_anchor_loc', 'get_l2_chunk_info', 'get_l2_info',
           'visualization']
