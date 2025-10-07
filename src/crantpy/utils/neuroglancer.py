# -*- coding: utf-8 -*-
"""
Neuroglancer scene generation and URL encoding/decoding for CRANT datasets.

This module provides tools to create, manipulate, and share neuroglancer scenes
for visualizing CRANT neurons, annotations, and connectivity data.

Key Features:
- Create neuroglancer URLs with selected segments, annotations, and skeletons
- Decode existing neuroglancer URLs to extract information
- Build custom scenes with different layer combinations
- Add annotations (points, lines, ellipsoids) to scenes
- Color and group segments for better visualization

Examples
--------
>>> import crantpy as crt
>>> from crantpy.utils.neuroglancer import encode_url, decode_url, construct_scene
>>>
>>> # Create a simple scene with some neurons
>>> url = encode_url(segments=[720575940621039145, 720575940621039146])
>>>
>>> # Decode an existing URL
>>> info = decode_url(url, format='brief')
>>> print(info['selected'])
>>>
>>> # Create a custom scene with specific layers
>>> scene = construct_scene(image=True, segmentation=True, brain_mesh=True)
>>> url = encode_url(scene=scene, segments=[720575940621039145])
"""

import copy
import json
import uuid
import webbrowser
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from urllib.parse import parse_qs, quote, unquote, urlparse

import matplotlib.colors as mcl
import navis
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import pyperclip

    HAS_PYPERCLIP = True
except ImportError:
    HAS_PYPERCLIP = False

from .config import (
    CRANT_NGL_DATASTACKS,
    CRANT_DEFAULT_DATASET,
    SCALE_X,
    SCALE_Y,
    SCALE_Z,
)
from .decorators import inject_dataset
from .helpers import make_iterable
from .cave.load import get_cave_client

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

__all__ = [
    "encode_url",
    "decode_url",
    "scene_to_url",
    "construct_scene",
    "add_annotation_layer",
    "add_skeleton_layer",
    "neurons_to_url",
]


@lru_cache
def _load_ngl_scenes() -> Dict[str, Any]:
    """Load neuroglancer layers & settings from JSON configuration."""
    fp = Path(__file__).parent
    with open(fp / "ngl_scenes.json") as f:
        return json.load(f)


def _find_segmentation_layer(
    layers: List[Dict], raise_not_found: bool = True
) -> Optional[int]:
    """Find the CRANT segmentation layer among given layers.

    Parameters
    ----------
    layers : list of dict
        List of neuroglancer layers.
    raise_not_found : bool, default True
        Whether to raise an error if no segmentation layer is found.

    Returns
    -------
    int or None
        Index of the segmentation layer, or None if not found and raise_not_found=False.
    """
    poss_names = list(CRANT_NGL_DATASTACKS.values())

    for i, layer in enumerate(layers):
        if layer.get("type") == "segmentation":
            # Check if it's one of our known segmentation layers
            layer_name = layer.get("name", "")
            if layer_name in poss_names or "proofreadable" in layer_name.lower():
                return i

            # Check source URL
            source = layer.get("source", {})
            if isinstance(source, dict):
                url = source.get("url", "")
            elif isinstance(source, str):
                url = source
            else:
                continue

            if "kronauer_ant" in url or "data.proofreading.zetta.ai" in url:
                return i

    if raise_not_found:
        raise ValueError("Unable to identify CRANT segmentation layer among layers")
    return None


@inject_dataset()
def construct_scene(
    *,
    image: bool = True,
    segmentation: bool = True,
    brain_mesh: bool = True,
    merge_biased_seg: bool = False,
    nuclei: bool = False,
    base_neuroglancer: bool = False,
    layout: Literal["3d", "xy-3d", "xy", "4panel"] = "xy-3d",
    dataset: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct a basic neuroglancer scene for CRANT data.

    Parameters
    ----------
    image : bool, default True
        Whether to add the aligned EM image layer.
    segmentation : bool, default True
        Whether to add the proofreadable segmentation layer.
    brain_mesh : bool, default True
        Whether to add the brain mesh layer.
    merge_biased_seg : bool, default False
        Whether to add the merge-biased segmentation layer (for proofreading).
    nuclei : bool, default False
        Whether to add the nuclei segmentation layer.
    base_neuroglancer : bool, default False
        Whether to use base neuroglancer (affects segmentation layer format).
    layout : str, default "xy-3d"
        Layout to show. Options: "3d", "xy-3d", "xy", "4panel".
    dataset : str, optional
        Which dataset to use ("latest" or "sandbox"). If None, uses default.

    Returns
    -------
    dict
        Neuroglancer scene dictionary with requested layers.

    Examples
    --------
    >>> # Create a minimal visualization scene
    >>> scene = construct_scene(image=True, segmentation=True, brain_mesh=True)
    >>>
    >>> # Create a full proofreading scene
    >>> scene = construct_scene(
    ...     image=True,
    ...     segmentation=True,
    ...     brain_mesh=True,
    ...     merge_biased_seg=True,
    ...     nuclei=True
    ... )
    """
    # Load scene templates
    NGL_SCENES = copy.deepcopy(_load_ngl_scenes())

    # Start with minimal scene
    scene = copy.deepcopy(NGL_SCENES["MINIMAL_SCENE"])

    # Update layout
    if isinstance(layout, str):
        scene["layout"] = {"type": layout, "orthographicProjection": True}
    else:
        scene["layout"] = layout

    # Add image layer
    if image:
        scene["layers"].append(NGL_SCENES["CRANT_IMAGE_LAYER"])

    # Add segmentation layer
    if segmentation:
        seg_layer = copy.deepcopy(NGL_SCENES["CRANT_SEG_LAYER"])

        # Set the correct dataset
        dataset_name = CRANT_NGL_DATASTACKS.get(dataset, dataset)
        if isinstance(seg_layer["source"], dict):
            seg_layer["source"]["url"] = seg_layer["source"]["url"].format(
                dataset=dataset_name
            )

        scene["layers"].append(seg_layer)

    # Add brain mesh
    if brain_mesh:
        scene["layers"].append(NGL_SCENES["CRANT_BRAIN_MESH_LAYER"])

    # Add merge-biased segmentation (optional, for proofreading)
    if merge_biased_seg:
        scene["layers"].append(NGL_SCENES["CRANT_MERGE_BIASED_SEG_LAYER"])

    # Add nuclei (optional, for proofreading)
    if nuclei:
        scene["layers"].append(NGL_SCENES["CRANT_NUCLEI_LAYER"])

    return scene


@inject_dataset()
def encode_url(
    segments: Optional[Union[int, List[int]]] = None,
    annotations: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
    coords: Optional[Union[List, np.ndarray]] = None,
    skeletons: Optional[Union[navis.TreeNeuron, navis.NeuronList]] = None,
    skeleton_names: Optional[Union[str, List[str]]] = None,
    seg_colors: Optional[Union[str, Tuple, List, Dict, np.ndarray]] = None,
    seg_groups: Optional[Union[List, Dict]] = None,
    invis_segs: Optional[Union[int, List[int]]] = None,
    scene: Optional[Union[Dict, str]] = None,
    base_neuroglancer: bool = False,
    layout: Literal["3d", "xy-3d", "xy", "4panel"] = "xy-3d",
    open: bool = False,
    to_clipboard: bool = False,
    shorten: bool = False,
    *,
    dataset: Optional[str] = None,
) -> str:
    """Encode data as CRANT neuroglancer scene URL.

    Parameters
    ----------
    segments : int or list of int, optional
        Segment IDs (root IDs) to have selected in the scene.
    annotations : array or dict, optional
        Coordinates for annotations:
        - (N, 3) array: Point annotations at x/y/z coordinates (in voxels)
        - dict: Multiple annotation layers {name: (N, 3) array}
    coords : (3,) array, optional
        X, Y, Z coordinates (in voxels) to center the view on.
    skeletons : TreeNeuron or NeuronList, optional
        Skeleton(s) to add as annotation layer(s). Must be in nanometers.
    skeleton_names : str or list of str, optional
        Names for the skeleton(s) to display in the UI. If a single string is provided,
        it will be used for all skeletons. If a list is provided, its length must
        match the number of skeletons.
    seg_colors : str, tuple, list, dict, or array, optional
        Colors for segments:
        - str or tuple: Single color for all segments
        - list: List of colors matching segments
        - dict: Mapping of segment IDs to colors
        - array: Labels that will be converted to colors
    seg_groups : list or dict, optional
        Group segments into separate layers:
        - list: Group labels matching segments
        - dict: {group_name: [seg_id1, seg_id2, ...]}
    invis_segs : int or list, optional
        Segment IDs to select but keep invisible.
    scene : dict or str, optional
        Existing scene to modify (as dict or URL string).
    base_neuroglancer : bool, default False
        Whether to use base neuroglancer instead of CAVE Spelunker.
    layout : str, default "xy-3d"
        Layout to show. Options: "3d", "xy-3d", "xy", "4panel".
    open : bool, default False
        If True, opens the URL in a web browser.
    to_clipboard : bool, default False
        If True, copies the URL to clipboard (requires pyperclip).
    shorten : bool, default False
        If True, creates a shortened URL (requires state server).
    dataset : str, optional
        Which dataset to use. If None, uses default.

    Returns
    -------
    str
        Neuroglancer URL.

    Examples
    --------
    >>> # Simple scene with segments
    >>> url = encode_url(segments=[720575940621039145, 720575940621039146])
    >>>
    >>> # Scene with colored segments
    >>> url = encode_url(
    ...     segments=[720575940621039145, 720575940621039146],
    ...     seg_colors={720575940621039145: 'red', 720575940621039146: 'blue'}
    ... )
    >>>
    >>> # Scene with skeleton and centered view
    >>> import navis
    >>> skeleton = crt.viz.get_skeletons([720575940621039145])[0]
    >>> url = encode_url(
    ...     segments=[720575940621039145],
    ...     skeletons=skeleton,
    ...     coords=[24899, 14436, 3739]
    ... )
    """
    # Handle scene input
    if isinstance(scene, str):
        scene = decode_url(scene, format="json")
        seg_layer_ix = _find_segmentation_layer(scene["layers"])
    elif isinstance(scene, dict):
        scene = copy.deepcopy(scene)
        seg_layer_ix = _find_segmentation_layer(scene["layers"])
    elif scene is None:
        scene = construct_scene(
            dataset=dataset,
            segmentation=True,
            image=True,
            brain_mesh=True,
            layout=layout,
            base_neuroglancer=base_neuroglancer,
        )
        # We know the segmentation layer is at index 1 (after image)
        seg_layer_ix = 1
    else:
        raise TypeError(f"`scene` must be string, dict or None, got {type(scene)}")

    # Set layout
    if isinstance(layout, str):
        scene["layout"] = {"type": layout, "orthographicProjection": True}
    else:
        scene["layout"] = layout

    # Add segments
    if segments is not None:
        segments = make_iterable(segments, force_type=str)

        if seg_groups is None:
            # Add to existing segments
            present = scene["layers"][seg_layer_ix].get("segments", [])
            scene["layers"][seg_layer_ix]["segments"] = present + list(segments)

    # Handle segment groups
    if seg_groups is not None:
        if not isinstance(seg_groups, dict):
            # Convert to dict format
            if not hasattr(seg_groups, "__iter__"):
                raise TypeError(
                    f'`seg_groups` must be dict or iterable, got "{type(seg_groups)}"'
                )

            if len(seg_groups) != len(segments):
                raise ValueError(
                    f"Got {len(seg_groups)} groups for {len(segments)} segments."
                )

            # Convert to group labels
            seg_groups = np.asarray(seg_groups)
            if seg_groups.dtype != object:
                seg_groups = [f"group_{i}" for i in seg_groups]

            # Create dict mapping
            seg_groups = dict(zip(segments, seg_groups))

        # Check format and convert to {group: [ids]} format
        is_list = [
            isinstance(v, (list, tuple, set, np.ndarray)) for v in seg_groups.values()
        ]

        if not any(is_list):
            # Convert {id: group} to {group: [ids]}
            groups = {}
            for s, g in seg_groups.items():
                if not isinstance(g, str):
                    raise TypeError(
                        f"Expected `seg_groups` to be strings, got {type(g)}"
                    )
                groups[g] = groups.get(g, []) + [str(s)]
        elif all(is_list):
            groups = {k: [str(s) for s in v] for k, v in seg_groups.items()}
        else:
            raise ValueError("`seg_groups` appears to be a mix of formats.")

        # Create layer for each group
        for g, ids in groups.items():
            new_layer = copy.deepcopy(scene["layers"][seg_layer_ix])
            new_layer["name"] = f"{g}"
            new_layer["segments"] = ids
            new_layer["visible"] = False
            scene["layers"].append(new_layer)

    # Add invisible segments
    if invis_segs is not None:
        invis_segs = make_iterable(invis_segs, force_type=str)
        present = scene["layers"][seg_layer_ix].get("hiddenSegments", [])
        scene["layers"][seg_layer_ix]["hiddenSegments"] = present + list(invis_segs)

    # Get all visible segments for coloring
    all_segs = segments if segments is not None else []

    # Assign colors
    if seg_colors is not None and len(all_segs) > 0:
        # Parse different color input formats
        if isinstance(seg_colors, str):
            # Single color name
            seg_colors = {s: seg_colors for s in all_segs}
        elif isinstance(seg_colors, tuple) and len(seg_colors) == 3:
            # Single RGB tuple
            seg_colors = {s: seg_colors for s in all_segs}
        elif (
            isinstance(seg_colors, (np.ndarray, pd.Series, pd.Categorical))
            and seg_colors.ndim == 1
        ):
            # Array of labels
            if len(seg_colors) != len(all_segs):
                raise ValueError(
                    f"Got {len(seg_colors)} colors for {len(all_segs)} segments."
                )

            # Generate palette based on number of unique labels
            uni = np.unique(seg_colors)
            if len(uni) > 20:
                pal = sns.color_palette("hls", len(uni) + 1)
                rng = np.random.default_rng(1985)
                rng.shuffle(pal)
            elif len(uni) > 10:
                pal = sns.color_palette("tab20", len(uni))
            else:
                pal = sns.color_palette("tab10", len(uni))

            color_map = dict(zip(uni, pal))
            seg_colors = {s: color_map[l] for s, l in zip(all_segs, seg_colors)}
        elif not isinstance(seg_colors, dict):
            # List of colors
            if not hasattr(seg_colors, "__iter__"):
                raise TypeError(
                    f'`seg_colors` must be dict or iterable, got "{type(seg_colors)}"'
                )
            if len(seg_colors) < len(all_segs):
                raise ValueError(
                    f"Got {len(seg_colors)} colors for {len(all_segs)} segments."
                )
            seg_colors = dict(zip(all_segs, seg_colors))

        # Convert all colors to hex
        seg_colors = {str(s): mcl.to_hex(c) for s, c in seg_colors.items()}
        scene["layers"][seg_layer_ix]["segmentColors"] = seg_colors

        # Propagate colors to group layers
        if seg_groups is not None:
            for layer in scene["layers"]:
                if layer["name"] in groups:
                    layer["segmentColors"] = {
                        s: seg_colors.get(s, "#ffffff") for s in layer["segments"]
                    }

    # Set coordinates
    if coords is not None:
        coords = np.asarray(coords)
        if not (coords.ndim == 1 and coords.shape[0] == 3):
            raise ValueError(f"Expected coords to be (3,) array, got {coords.shape}")

        # Ensure position is set in scene
        scene["position"] = coords.round().astype(int).tolist()

    # Add annotations
    if annotations is not None:
        if isinstance(annotations, (np.ndarray, list)):
            scene = add_annotation_layer(annotations, scene)
        elif isinstance(annotations, dict):
            for layer_name, coords_array in annotations.items():
                scene = add_annotation_layer(coords_array, scene, name=layer_name)

    # Add skeletons
    if skeletons is not None:
        if isinstance(skeletons, navis.NeuronList):
            if skeleton_names is None:
                skeleton_names = [f"Neuron {i+1}" for i in range(len(skeletons))]
            elif isinstance(skeleton_names, str):
                skeleton_names = [skeleton_names] * len(skeletons)
            elif len(skeleton_names) != len(skeletons):
                raise ValueError(
                    f"Got {len(skeleton_names)} names for {len(skeletons)} skeletons."
                )
            for neuron, name in zip(skeletons, skeleton_names):
                scene = add_skeleton_layer(neuron, scene, name=name)
        else:
            if isinstance(skeleton_names, list):
                if len(skeleton_names) != 1:
                    raise ValueError(
                        f"Got {len(skeleton_names)} names for single skeleton."
                    )
            elif isinstance(skeleton_names, str):
                skeleton_names = [skeleton_names]
            else:
                skeleton_names = [f"Skeleton {uuid.uuid4().hex[:6]}"]
            scene = add_skeleton_layer(skeletons, scene, name=skeleton_names[0])

    return scene_to_url(
        scene,
        base_neuroglancer=base_neuroglancer,
        shorten=shorten,
        open=open,
        to_clipboard=to_clipboard,
    )


def scene_to_url(
    scene: Dict[str, Any],
    base_neuroglancer: bool = False,
    shorten: bool = False,
    open: bool = False,
    to_clipboard: bool = False,
) -> str:
    """Convert neuroglancer scene dictionary to URL.

    Parameters
    ----------
    scene : dict
        Neuroglancer scene dictionary.
    base_neuroglancer : bool, default False
        Whether to use base neuroglancer instead of CAVE Spelunker.
    shorten : bool, default False
        Whether to create a shortened URL (requires state server).
    open : bool, default False
        If True, opens URL in web browser.
    to_clipboard : bool, default False
        If True, copies URL to clipboard.

    Returns
    -------
    str
        Neuroglancer URL.
    """
    NGL_SCENES = copy.deepcopy(_load_ngl_scenes())

    # Check if we can shorten
    if shorten:
        state_url = NGL_SCENES.get("CRANT_STATE_URL")
        if not state_url or not HAS_REQUESTS:
            if not HAS_REQUESTS:
                print("Warning: requests module not available. Cannot shorten URL.")
            else:
                print("Warning: No state server configured. Cannot shorten URL.")
            shorten = False

    # Build URL
    if shorten:
        # Use state server to shorten URL
        try:
            url = _shorten_url(scene, state_url)
        except Exception as e:
            print(f"Warning: Failed to shorten URL: {e}. Using full URL instead.")
            shorten = False

    if not shorten:
        # Create full URL
        scene_str = (
            json.dumps(scene)
            .replace("'", '"')
            .replace("True", "true")
            .replace("False", "false")
        )

        if not base_neuroglancer:
            ngl_url = NGL_SCENES["NGL_URL_SPELUNKER"]
        else:
            ngl_url = NGL_SCENES["NGL_URL_BASIC"]

        url = f"{ngl_url}/#!{quote(scene_str)}"

    # Open in browser
    if open:
        try:
            wb = webbrowser.get("chrome")
        except Exception:
            wb = webbrowser
        wb.open_new_tab(url)

    # Copy to clipboard
    if to_clipboard:
        if not HAS_PYPERCLIP:
            print("Warning: pyperclip not installed. Cannot copy to clipboard.")
        else:
            pyperclip.copy(url)
            print("URL copied to clipboard.")

    return url


def _shorten_url(scene: Dict[str, Any], state_url: str) -> str:
    """Upload scene to state server and get shortened URL, falling back to CAVE."""
    if not HAS_REQUESTS:
        raise ImportError("requests module required for URL shortening")

    import warnings
    import requests

    NGL_SCENES = _load_ngl_scenes()
    spelunker_url = NGL_SCENES.get(
        "NGL_URL_SPELUNKER", "https://spelunker.cave-explorer.org"
    )

    client = get_cave_client()
    token = client.auth.token

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})
    session.cookies.set("middle_auth_token", token)

    payload = json.dumps(scene)
    state_url = state_url.rstrip("/")

    def _format_url(url: str) -> str:
        return url.strip().replace(" /#!", "/#!").replace(" ", "").replace("//", "/")

    def _try_endpoint(endpoint: str) -> Optional[str]:
        resp = session.post(endpoint, data=payload, timeout=10)
        if resp.ok:
            out = resp.json()
            if isinstance(out, dict):
                out = out.get("url") or out.get("id") or out.get("json_url")
            return str(out) if out else None
        resp.raise_for_status()
        return None

    for endpoint in (f"{state_url}/post", f"{state_url}/api/v1/post"):
        try:
            json_url = _try_endpoint(endpoint)
            if json_url:
                return f"{spelunker_url}/?json_url={json_url.strip()}"
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in (401, 403):
                continue
            raise

    warnings.warn(
        "Primary state server rejected authentication; falling back to CAVE upload.",
        RuntimeWarning,
        stacklevel=2,
    )

    state_id = client.state.upload_state_json(scene)
    url = client.state.build_neuroglancer_url(state_id, target_site="cave-explorer")
    return _format_url(url)


def decode_url(
    url: Union[str, List[str]],
    format: Literal["json", "brief", "dataframe"] = "json",
) -> Union[Dict[str, Any], pd.DataFrame]:
    """Decode neuroglancer URL to extract information.

    Parameters
    ----------
    url : str or list of str
        Neuroglancer URL(s) to decode.
    format : str, default "json"
        Output format:
        - "json": Full scene dictionary
        - "brief": Dict with position, selected segments, and annotations
        - "dataframe": DataFrame with segment IDs and their layers

    Returns
    -------
    dict or DataFrame
        Decoded information in requested format.

    Examples
    --------
    >>> url = "https://spelunker.cave-explorer.org/#!{...}"
    >>> info = decode_url(url, format='brief')
    >>> print(info['selected'])  # List of selected segment IDs
    >>> print(info['position'])  # [x, y, z] coordinates
    """
    if isinstance(url, list):
        if format != "dataframe":
            raise ValueError('Can only parse multiple URLs if format="dataframe"')
        return pd.concat([decode_url(u, format=format) for u in url], axis=0)

    if not isinstance(url, str):
        raise TypeError(f'`url` must be string, got "{type(url)}"')

    # Parse URL to extract scene
    if isinstance(url, str):
        query = unquote(urlparse(url).fragment)[1:]
        try:
            scene = json.loads(query)
        except json.JSONDecodeError:
            raise ValueError(f"Could not decode URL: {url}")
    else:
        scene = url

    # Return in requested format
    if format == "json":
        return scene

    elif format == "brief":
        seg_layers = [
            layer
            for layer in scene.get("layers", [])
            if "segmentation" in layer.get("type", "")
        ]
        an_layers = [
            layer
            for layer in scene.get("layers", [])
            if layer.get("type") == "annotation"
        ]

        # Try to get position from different possible locations
        position = None
        if "position" in scene:
            # New format
            position = scene["position"]
        elif "navigation" in scene:
            # Old format
            try:
                position = scene["navigation"]["pose"]["position"].get(
                    "voxelCoordinates", None
                )
            except (KeyError, TypeError):
                pass

        return {
            "position": position,
            "annotations": [
                a for layer in an_layers for a in layer.get("annotations", [])
            ],
            "selected": [s for layer in seg_layers for s in layer.get("segments", [])],
        }

    elif format == "dataframe":
        segs = []
        seg_layers = [
            layer
            for layer in scene.get("layers", [])
            if "segmentation" in layer.get("type", "")
        ]

        for layer in seg_layers:
            for s in layer.get("segments", []):
                # Handle invisible segments (marked with !)
                is_visible = not s.startswith("!")
                seg_id = int(s.replace("!", ""))
                segs.append([seg_id, layer["name"], is_visible])

        return pd.DataFrame(segs, columns=["segment", "layer", "visible"])

    else:
        raise ValueError(f'Unexpected format: "{format}"')


def add_annotation_layer(
    annotations: Union[np.ndarray, List],
    scene: Dict[str, Any],
    name: Optional[str] = None,
    connected: bool = False,
) -> Dict[str, Any]:
    """Add annotations as new layer to scene.

    Parameters
    ----------
    annotations : array or list
        Coordinates for annotations (in voxel space):
        - (N, 3): Point annotations at x/y/z coordinates
        - (N, 2, 3): Line segments with start and end points
        - (N, 4): Ellipsoids with x/y/z center and radius
    scene : dict
        Scene to add annotation layer to.
    name : str, optional
        Name for the annotation layer.
    connected : bool, default False
        If True, point annotations will be connected as a path (TODO).

    Returns
    -------
    dict
        Modified scene with annotation layer added.

    Examples
    --------
    >>> # Add point annotations
    >>> points = np.array([[100, 200, 50], [150, 250, 60]])
    >>> scene = add_annotation_layer(points, scene, name="my_points")
    >>>
    >>> # Add line annotations
    >>> lines = np.array([
    ...     [[100, 200, 50], [150, 250, 60]],
    ...     [[150, 250, 60], [200, 300, 70]]
    ... ])
    >>> scene = add_annotation_layer(lines, scene, name="my_lines")
    """
    if not isinstance(scene, dict):
        raise TypeError(f'`scene` must be dict, got "{type(scene)}"')

    scene = scene.copy()
    annotations = np.asarray(annotations)

    # Generate annotation records
    records = []

    if annotations.ndim == 2 and annotations.shape[1] == 3:
        # Point annotations
        for co in annotations.round().astype(int).tolist():
            records.append(
                {"point": co, "type": "point", "tagIds": [], "id": str(uuid.uuid4())}
            )

    elif annotations.ndim == 2 and annotations.shape[1] == 4:
        # Ellipsoid annotations
        for co in annotations.round().astype(int).tolist():
            records.append(
                {
                    "center": co[:3],
                    "radii": [co[3], co[3], co[3]],
                    "type": "ellipsoid",
                    "id": str(uuid.uuid4()),
                }
            )

    elif (
        annotations.ndim == 3
        and annotations.shape[1] == 2
        and annotations.shape[2] == 3
    ):
        # Line annotations
        for co in annotations.round().astype(int).tolist():
            start, end = co[0], co[1]
            records.append(
                {
                    "pointA": start,
                    "pointB": end,
                    "type": "line",
                    "id": str(uuid.uuid4()),
                }
            )

    else:
        raise ValueError(
            f"Expected annotations of shape (N, 3) for points, (N, 4) for ellipsoids, "
            f"or (N, 2, 3) for lines. Got {annotations.shape}"
        )

    # Generate layer name
    if not name:
        existing_an_layers = [l for l in scene["layers"] if l["type"] == "annotation"]
        name = f"annotation{len(existing_an_layers)}"

    # Create annotation layer
    an_layer = {
        "type": "annotation",
        "annotations": records,
        "annotationTags": [],
        "voxelSize": [SCALE_X, SCALE_Y, SCALE_Z],
        "name": name,
    }

    scene["layers"].append(an_layer)
    return scene


def add_skeleton_layer(
    skeleton: Union[navis.TreeNeuron, pd.DataFrame],
    scene: Dict[str, Any],
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Add skeleton as line annotation layer to scene.

    Parameters
    ----------
    skeleton : TreeNeuron or DataFrame
        Neuron skeleton to add. Coordinates must be in nanometers.
        Will be automatically converted to voxel space.
    scene : dict
        Scene to add skeleton layer to.
    name : str, optional
        Name for the skeleton layer.

    Returns
    -------
    dict
        Modified scene with skeleton layer added.

    Examples
    --------
    >>> skeleton = crt.viz.get_skeletons([720575940621039145])[0]
    >>> scene = construct_scene()
    >>> scene = add_skeleton_layer(skeleton, scene)
    """
    if not isinstance(scene, dict):
        raise TypeError(f'`scene` must be dict, got "{type(scene)}"')

    scene = scene.copy()

    if isinstance(skeleton, navis.NeuronList):
        if len(skeleton) > 1:
            raise ValueError(f"Expected a single neuron, got {len(skeleton)}")
        skeleton = skeleton[0]

    if not isinstance(skeleton, (navis.TreeNeuron, pd.DataFrame)):
        raise TypeError(f"Expected TreeNeuron or DataFrame, got {type(skeleton)}")

    # Get node table
    if isinstance(skeleton, navis.TreeNeuron):
        nodes = skeleton.nodes
        neuron_name = getattr(skeleton, "name", "skeleton")
    else:
        nodes = skeleton
        neuron_name = "skeleton"

    # Generate line segments
    not_root = nodes[nodes.parent_id >= 0]
    loc1 = not_root[["x", "y", "z"]].values
    loc2 = (
        nodes.set_index("node_id")
        .loc[not_root.parent_id.values, ["x", "y", "z"]]
        .values
    )

    # Stack into (N, 2, 3) array
    stack = np.stack([loc1, loc2], axis=1)

    # Convert from nanometers to voxels
    stack = stack / [SCALE_X, SCALE_Y, SCALE_Z]

    # Use neuron name if no name provided
    if name is None:
        name = neuron_name

    return add_annotation_layer(stack, scene, name=name)


def neurons_to_url(
    neurons: navis.NeuronList,
    include_skeleton: bool = True,
    downsample: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:
    """Create neuroglancer URLs for a list of neurons.

    Parameters
    ----------
    neurons : NeuronList
        List of neurons to create URLs for. Must have root_id attribute.
    include_skeleton : bool, default True
        Whether to include the skeleton in the URL.
    downsample : int, optional
        Factor by which to downsample skeletons before adding to scene.
    **kwargs
        Additional arguments passed to encode_url().

    Returns
    -------
    DataFrame
        DataFrame with columns: id, name, url

    Examples
    --------
    >>> neurons = crt.viz.get_skeletons([720575940621039145, 720575940621039146])
    >>> urls = neurons_to_url(neurons)
    >>> print(urls[['id', 'url']])
    """
    if not isinstance(neurons, navis.NeuronList):
        raise TypeError(f"Expected NeuronList, got {type(neurons)}")

    data = []
    for neuron in neurons:
        # Get root ID
        root_id = getattr(neuron, "id", None)
        name = getattr(neuron, "name", str(root_id))

        if root_id is None:
            print(f"Warning: Skipping neuron without ID: {name}")
            continue

        # Downsample if requested
        if downsample:
            neuron = navis.downsample_neuron(neuron, downsample)

        # Create URL
        url = encode_url(
            segments=[root_id], skeletons=neuron if include_skeleton else None, **kwargs
        )

        data.append([root_id, name, url])

    return pd.DataFrame(data, columns=["id", "name", "url"])
