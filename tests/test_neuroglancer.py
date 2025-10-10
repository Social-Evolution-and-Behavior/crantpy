"""
Tests for neuroglancer URL encoding and decoding functionality.
"""

import pytest
import numpy as np
import pandas as pd

from crantpy.utils import neuroglancer as ngl


def test_construct_scene():
    """Test basic scene construction."""
    # Minimal scene with all defaults
    scene = ngl.construct_scene()

    assert "layers" in scene
    assert "layout" in scene
    assert scene["layout"]["type"] == "xy-3d"

    # Check that we have the expected layers
    layer_names = [layer["name"] for layer in scene["layers"]]
    assert "aligned" in layer_names  # EM image
    assert "proofreadable seg" in layer_names  # Segmentation
    assert "brain mesh" in layer_names  # Brain mesh


def test_construct_scene_minimal():
    """Test scene construction with minimal layers."""
    scene = ngl.construct_scene(image=False, segmentation=True, brain_mesh=False)

    layer_names = [layer["name"] for layer in scene["layers"]]
    assert "aligned" not in layer_names
    assert "proofreadable seg" in layer_names
    assert "brain mesh" not in layer_names


def test_construct_scene_proofreading():
    """Test scene construction with proofreading layers."""
    scene = ngl.construct_scene(
        image=True,
        segmentation=True,
        brain_mesh=True,
        merge_biased_seg=True,
        nuclei=True,
    )

    layer_names = [layer["name"] for layer in scene["layers"]]
    assert "merge-biased seg" in layer_names
    assert "nuclei" in layer_names


def test_encode_url_simple():
    """Test simple URL encoding with segments."""
    url = ngl.encode_url(segments=[720575940621039145])

    assert isinstance(url, str)
    assert "spelunker.cave-explorer.org" in url or "neuroglancer" in url
    assert "#!" in url


def test_encode_url_multiple_segments():
    """Test URL encoding with multiple segments."""
    segments = [720575940621039145, 720575940621039146]
    url = ngl.encode_url(segments=segments)

    # Decode and verify
    info = ngl.decode_url(url, format="brief")
    assert len(info["selected"]) >= 2
    assert str(segments[0]) in info["selected"]
    assert str(segments[1]) in info["selected"]


def test_encode_url_with_coords():
    """Test URL encoding with coordinates."""
    coords = [24899, 14436, 3739]
    url = ngl.encode_url(segments=[720575940621039145], coords=coords)

    info = ngl.decode_url(url, format="brief")
    assert info["position"] == coords


def test_encode_url_with_colors():
    """Test URL encoding with segment colors."""
    segments = [720575940621039145, 720575940621039146]
    colors = {720575940621039145: "red", 720575940621039146: "blue"}

    url = ngl.encode_url(segments=segments, seg_colors=colors)
    scene = ngl.decode_url(url, format="json")

    # Find segmentation layer
    seg_layer = None
    for layer in scene["layers"]:
        if "segmentation" in layer.get("type", ""):
            seg_layer = layer
            break

    assert seg_layer is not None
    assert "segmentColors" in seg_layer


def test_decode_url_brief():
    """Test URL decoding in brief format."""
    segments = [720575940621039145]
    coords = [24899, 14436, 3739]
    url = ngl.encode_url(segments=segments, coords=coords)

    info = ngl.decode_url(url, format="brief")

    assert "position" in info
    assert "selected" in info
    assert "annotations" in info
    assert info["position"] == coords
    assert str(segments[0]) in info["selected"]


def test_decode_url_dataframe():
    """Test URL decoding to DataFrame."""
    segments = [720575940621039145, 720575940621039146]
    url = ngl.encode_url(segments=segments)

    df = ngl.decode_url(url, format="dataframe")

    assert isinstance(df, pd.DataFrame)
    assert "segment" in df.columns
    assert "layer" in df.columns
    assert "visible" in df.columns
    assert len(df) >= 2


def test_add_annotation_layer_points():
    """Test adding point annotations to a scene."""
    scene = ngl.construct_scene()
    points = np.array([[100, 200, 50], [150, 250, 60], [200, 300, 70]])

    scene = ngl.add_annotation_layer(points, scene, name="test_points")

    # Check that annotation layer was added
    an_layers = [l for l in scene["layers"] if l["type"] == "annotation"]
    assert len(an_layers) > 0
    assert an_layers[-1]["name"] == "test_points"
    assert len(an_layers[-1]["annotations"]) == 3


def test_add_annotation_layer_lines():
    """Test adding line annotations to a scene."""
    scene = ngl.construct_scene()
    lines = np.array(
        [[[100, 200, 50], [150, 250, 60]], [[150, 250, 60], [200, 300, 70]]]
    )

    scene = ngl.add_annotation_layer(lines, scene, name="test_lines")

    # Check that annotation layer was added
    an_layers = [l for l in scene["layers"] if l["type"] == "annotation"]
    assert len(an_layers) > 0
    assert an_layers[-1]["name"] == "test_lines"
    assert len(an_layers[-1]["annotations"]) == 2
    assert an_layers[-1]["annotations"][0]["type"] == "line"


def test_seg_groups():
    """Test segment grouping into separate layers."""
    segments = [720575940621039145, 720575940621039146, 720575940621039147]
    groups = {
        "group_A": [720575940621039145, 720575940621039146],
        "group_B": [720575940621039147],
    }

    url = ngl.encode_url(segments=segments, seg_groups=groups)
    scene = ngl.decode_url(url, format="json")

    layer_names = [l["name"] for l in scene["layers"]]
    assert "group_A" in layer_names
    assert "group_B" in layer_names


def test_layout_options():
    """Test different layout options."""
    layouts = ["3d", "xy-3d", "xy", "4panel"]

    for layout in layouts:
        url = ngl.encode_url(segments=[720575940621039145], layout=layout)
        scene = ngl.decode_url(url, format="json")
        assert scene["layout"]["type"] == layout


def test_find_segmentation_layer():
    """Test finding segmentation layer in scene."""
    scene = ngl.construct_scene()
    seg_ix = ngl._find_segmentation_layer(scene["layers"])

    assert seg_ix is not None
    assert "segmentation" in scene["layers"][seg_ix]["type"]


def test_invisible_segments():
    """Test adding invisible segments."""
    segments = [720575940621039145, 720575940621039146]
    invis = [720575940621039147]

    url = ngl.encode_url(segments=segments, invis_segs=invis)
    scene = ngl.decode_url(url, format="json")

    # Find segmentation layer
    seg_layer = None
    for layer in scene["layers"]:
        if "segmentation" in layer.get("type", ""):
            seg_layer = layer
            break

    assert seg_layer is not None
    if "hiddenSegments" in seg_layer:
        assert str(invis[0]) in seg_layer["hiddenSegments"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
