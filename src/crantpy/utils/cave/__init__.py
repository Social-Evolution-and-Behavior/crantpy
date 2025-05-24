import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'auth',
        'helpers',
        'load',
        'segmentation',
    },
    submod_attrs={
        'auth': [
            'generate_cave_token',
            'get_current_cave_token',
            'set_cave_token',
        ],
        'helpers': [
            'parse_root_ids',
            'parse_timestamp',
        ],
        'load': [
            'clear_all_caches',
            'clear_cave_client_cache',
            'clear_cloudvolume_cache',
            'get_cave_client',
            'get_cave_datastacks',
            'get_cloudvolume',
            'get_dataset_segmentation_source',
            'get_datastack_segmentation_source',
            'validate_cave_client',
        ],
        'segmentation': [
            'roots_to_supervoxels',
            'supervoxels_to_roots',
        ],
    },
)

__all__ = ['auth', 'clear_all_caches', 'clear_cave_client_cache',
           'clear_cloudvolume_cache', 'generate_cave_token', 'get_cave_client',
           'get_cave_datastacks', 'get_cloudvolume', 'get_current_cave_token',
           'get_dataset_segmentation_source',
           'get_datastack_segmentation_source', 'helpers', 'load',
           'parse_root_ids', 'parse_timestamp', 'roots_to_supervoxels',
           'segmentation', 'set_cave_token', 'supervoxels_to_roots',
           'validate_cave_client']
