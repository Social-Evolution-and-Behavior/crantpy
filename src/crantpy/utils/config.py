# -*- coding: utf-8 -*-
"""
This module contains configuration settings for CRANTpy.
It includes the default dataset, CRANT data stacks, and Seatable server details.
It also provides a decorator to inject the current default dataset into functions.
"""
import os

CRANT_VALID_DATASETS = ["latest", "sandbox"]
CRANT_DEFAULT_DATASET = os.environ.get("CRANT_DEFAULT_DATASET", "latest")
if CRANT_DEFAULT_DATASET not in CRANT_VALID_DATASETS:
    raise ValueError(
        f"Invalid CRANT_DEFAULT_DATASET: {CRANT_DEFAULT_DATASET}. "
        f"Accepted values are: {CRANT_VALID_DATASETS}"
    )


# function to forcefully set the default dataset to a specific value
def set_default_dataset(dataset: str):
    global CRANT_DEFAULT_DATASET
    # Check if the dataset is valid
    if dataset not in CRANT_VALID_DATASETS:
        raise ValueError(
            f"Invalid CRANT_DEFAULT_DATASET: {dataset}. "
            f"Accepted values are: {CRANT_VALID_DATASETS}"
        )
    CRANT_DEFAULT_DATASET = dataset
    return CRANT_DEFAULT_DATASET


CRANT_CAVE_SERVER_URL = "https://proofreading.zetta.ai"
CRANT_NGL_DATASTACKS = {
    "latest": "kronauer_ant_x1",
    "sandbox": "kronauer_ant_sandbox_x1",
}
CRANT_CAVE_DATASTACKS = {
    "latest": "kronauer_ant",
    "sandbox": "kronauer_ant_sandbox",
}

CRANT_SEATABLE_SERVER_URL = "https://cloud.seatable.io/"
CRANT_SEATABLE_WORKSPACE_ID = "62919"
CRANT_SEATABLE_BASENAME = "CRANTb"

CRANT_SEATABLE_ANNOTATIONS_TABLES = {
    "latest": "CRANTb_meta",
    "sandbox": "CRANTb_meta",
}

CRANT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".crantpy")
MAXIMUM_CACHE_DURATION = 2 * 60 * 60  # 2 hours

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

# Voxel resolution in nm
SCALE_X, SCALE_Y, SCALE_Z = 8, 8, 42

# Voxel offset kept for backward compatibility. Neuropil meshes are already aligned.
VOXEL_OFFSET = (0, 0, 0)

# Neuroglancer URL for the whole brain tissue mesh
WHOLE_BRAIN_TISSUE_MESH_URL = (
    "https://www.googleapis.com/storage/v1/b/"
    "dkronauer-ant-001-alignment-final/o/tissue_mesh%2F"
    "mesh%2Ftissue_mesh.frag?alt=media"
    "&neuroglancer=a2b0cf07baf8c501891d6c683cc7e24a"
)

# URL for the precomputed aligned EM data
ALIGNED_EM_URL = "precomputed://gs://dkronauer-ant-001-alignment-final/aligned"

# URL for precomputed neuropil meshes
NEUROPIL_MESH_URL = "precomputed://https://raw.githubusercontent.com/yigityargili991/haberkernlab_mesh_repo/6d007140ff8b15dd20b4110b8c22b556e76c6f14/"

# Neuropil Mesh Dict mapping source label IDs to names
NEUROPIL_MESH_DICT = {
    1: "mushroom_body_pedunculus_and_lobes_left",
    2: "mushroom_body_medial_calyx_left",
    3: "mushroom_body_lateral_calyx_left",
    4: "mushroom_body_pedunculus_and_lobes_right",
    5: "mushroom_body_medial_calyx_right",
    6: "mushroom_body_lateral_calyx_right",
    7: "antennal_lobe_left",
    8: "antennal_lobe_right",
    9: "fan_shaped_body",
    10: "ellipsoid_body",
    11: "protocerebral_bridge",
    12: "nodulus_left",
    13: "optic_lobe_left",
    14: "optic_lobe_right",
    15: "lateral_horn_left",
    16: "lateral_horn_right",
    17: "pb_glomerulus_L5",
    18: "pb_glomerulus_L6",
    19: "pb_glomerulus_L7",
    20: "pb_glomerulus_L8",
    21: "pb_glomerulus_R1",
    22: "pb_glomerulus_R2",
    23: "pb_glomerulus_R3",
    24: "pb_glomerulus_R5",
    25: "pb_glomerulus_R7",
    26: "pb_glomerulus_R8",
    27: "gall_left",
    28: "gall_right",
    29: "bulb_left",
    30: "bulb_right",
    31: "posterior_optic_tubercle_left",
    32: "posterior_optic_tubercle_right",
    33: "NOc_r",
    34: "NOm1_r",
    35: "NOm2_r",
    36: "NOs_r",
}

# Compatibility aliases for labels no longer present as single source segments.
NEUROPIL_MESH_ALIASES = {
    "nodulus_right": ["NOc_r", "NOm1_r", "NOm2_r", "NOs_r"],
}
