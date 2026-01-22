import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'l1',
        'l2',
        'mesh',
        'skeletonize',
    },
    submod_attrs={
        'l1': [
            'dotprops_from_mesh',
            'get_l1_dotprops',
            'get_skeleton',
        ],
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
            'get_brain_mesh_scene',
            'get_mesh_neuron',
            'load_neuropil_mesh',
            'load_whole_brain_mesh',
        ],
        'skeletonize': [
            '_create_node_info_dict',
            '_preprocess_mesh',
            '_remove_soma_hairball',
            '_shave_skeleton',
            '_swc_dict_to_dataframe',
            '_worker_wrapper',
            'chunks_to_nm',
            'configure_urllib3_warning_suppression',
            'detect_soma_mesh',
            'detect_soma_skeleton',
            'divide_local_neighbourhood',
            'get_skeletons',
            'get_soma_from_annotations',
            'skeletonize_neuron',
            'skeletonize_neurons_parallel',
            'suppress_urllib3_connectionpool_warnings',
        ],
    },
)

__all__ = ['_create_node_info_dict', '_preprocess_mesh',
           '_remove_soma_hairball', '_shave_skeleton',
           '_swc_dict_to_dataframe', '_worker_wrapper', 'chunks_to_nm',
           'configure_urllib3_warning_suppression', 'detect_soma',
           'detect_soma_mesh', 'detect_soma_skeleton',
           'divide_local_neighbourhood', 'dotprops_from_mesh',
           'find_anchor_loc', 'get_brain_mesh_scene', 'get_l1_dotprops',
           'get_l2_chunk_info', 'get_l2_dotprops', 'get_l2_graph',
           'get_l2_info', 'get_l2_meshes', 'get_l2_skeleton',
           'get_mesh_neuron', 'get_skeleton', 'get_skeletons',
           'get_soma_from_annotations', 'l1', 'l2', 'load_neuropil_mesh',
           'load_whole_brain_mesh', 'mesh', 'skeletonize',
           'skeletonize_neuron', 'skeletonize_neurons_parallel',
           'suppress_urllib3_connectionpool_warnings']
