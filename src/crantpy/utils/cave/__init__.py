import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        "auth",
        "helpers",
        "load",
        "segmentation",
    },
    submod_attrs={
        "auth": [
            "generate_cave_token",
            "get_current_cave_token",
            "set_cave_token",
        ],
        "helpers": [
            "is_latest_roots",
            "is_valid_root",
            "is_valid_supervoxel",
            "parse_root_ids",
        ],
        "load": [
            "clear_all_caches",
            "clear_cave_client_cache",
            "clear_cloudvolume_cache",
            "get_cave_client",
            "get_cave_datastacks",
            "get_cloudvolume",
            "get_dataset_segmentation_source",
            "get_datastack_segmentation_source",
            "validate_cave_client",
        ],
        "segmentation": [
            "roots_to_supervoxels",
            "supervoxels_to_roots",
            "update_ids",
        ],
    },
)

__all__ = [
    "auth",
    "clear_all_caches",
    "clear_cave_client_cache",
    "clear_cloudvolume_cache",
    "generate_cave_token",
    "get_cave_client",
    "get_cave_datastacks",
    "get_cloudvolume",
    "get_current_cave_token",
    "get_dataset_segmentation_source",
    "get_datastack_segmentation_source",
    "helpers",
    "is_latest_roots",
    "is_valid_root",
    "is_valid_supervoxel",
    "load",
    "parse_root_ids",
    "roots_to_supervoxels",
    "segmentation",
    "set_cave_token",
    "supervoxels_to_roots",
    "update_ids",
    "validate_cave_client",
]
