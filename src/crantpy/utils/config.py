# -*- coding: utf-8 -*-
"""
This module contains configuration settings for CRANTpy.
It includes the default dataset, CRANT data stacks, and Seatable server details.
It also provides a decorator to inject the current default dataset into functions.
"""
import os

CRANT_VALID_DATASETS = ['latest', 'sandbox']
CRANT_DEFAULT_DATASET = os.environ.get('CRANT_DEFAULT_DATASET', 'latest')
if CRANT_DEFAULT_DATASET not in CRANT_VALID_DATASETS:
    raise ValueError(f"Invalid CRANT_DEFAULT_DATASET: {CRANT_DEFAULT_DATASET}. "
                     f"Accepted values are: {CRANT_VALID_DATASETS}")


CRANT_CAVE_SERVER_URL = "https://proofreading.zetta.ai"
CRANT_CAVE_DATASTACKS = {
    'latest': 'kronauer_ant',
    'sandbox': 'kronauer_ant_clone_x1',
}

CRANT_SEATABLE_SERVER_URL = "https://cloud.seatable.io/"
CRANT_SEATABLE_WORKSPACE_ID = "62919"
CRANT_SEATABLE_BASENAME = "CRANTb"

CRANT_SEATABLE_ANNOTATIONS_TABLES = {
    'latest': 'CRANTb_meta',
    'sandbox': 'CRANTb_meta',
}

MAXIMUM_CACHE_DURATION = 2 * 60 * 60 # 2 hours 

ALL_ANNOTATION_FIELDS = [
    "root_id",
    "root_id_processed",
    "supervoxel_id",
    "position",
    "nucleus_id",
    "nucleus_position",
    "root_position",
    "cave_table",
    "proofread",
    "status",
    "region",
    "proofreader_notes",
    "side",
    "nerve",
    "tract",
    "hemilineage",
    "flow",
    "super_class",
    "cell_class",
    "cell_type",
    "cell_subtype",
    "cell_instance",
    "known_nt",
    "known_nt_source",
    "alternative_names",
    "annotator_notes",
    "user_annotator",
    "user_proofreader",
    "ngl_link",
    "date_proofread",
]

SEARCH_EXCLUDED_ANNOTATION_FIELDS = [
    "root_id_processed",
    "supervoxel_id",
    "position",
    "nucleus_position",
    "root_position",
]

SCALE_X, SCALE_Y, SCALE_Z = 8, 8, 42