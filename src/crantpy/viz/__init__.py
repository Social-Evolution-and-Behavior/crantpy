import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'l2',
        'mesh',
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
        'mesh': [
            'detect_soma',
            'get_mesh_neuron',
        ],
    },
)

__all__ = ['chunks_to_nm', 'detect_soma', 'find_anchor_loc',
           'get_l2_chunk_info', 'get_l2_dotprops', 'get_l2_graph',
           'get_l2_info', 'get_l2_meshes', 'get_l2_skeleton',
           'get_mesh_neuron', 'l2', 'mesh']
