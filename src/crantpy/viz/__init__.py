import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'l2',
    },
    submod_attrs={
        'l2': [
            'chunks_to_nm',
            'find_anchor_loc',
            'get_l2_chunk_info',
            'get_l2_dotprops',
            'get_l2_graph',
            'get_l2_info',
            'get_l2_meshes',
            'get_l2_skeleton',
        ],
    },
)

__all__ = ['chunks_to_nm', 'find_anchor_loc', 'get_l2_chunk_info',
           'get_l2_dotprops', 'get_l2_graph', 'get_l2_info', 'get_l2_meshes',
           'get_l2_skeleton', 'l2']
