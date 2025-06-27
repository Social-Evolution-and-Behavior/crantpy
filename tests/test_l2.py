import pytest
import numpy as np
import pandas as pd
import navis
from crantpy.viz import l2
from typing import Any, Dict, List, Set, Tuple

# Example root ID for testing 
TEST_ROOT_ID = 576460752732354679

@pytest.mark.parametrize("root_id", [TEST_ROOT_ID])
def test_get_l2_info(root_id) -> None:
    """Test get_l2_info returns DataFrame with 'root_id' column.
    This verifies that get_l2_info fetches L2 info and returns a DataFrame
    with the expected columns for a given neuron root ID.
    """
    df = l2.get_l2_info(root_id)
    assert isinstance(df, pd.DataFrame)
    assert 'root_id' in df.columns

@pytest.mark.parametrize("root_id", [TEST_ROOT_ID])
def test_get_l2_chunk_info(root_id) -> None:
    """Test get_l2_chunk_info returns DataFrame with 'id' column.
    This checks that L2 chunk info is fetched and structured as expected.
    """
    df = l2.get_l2_chunk_info(root_id)
    assert isinstance(df, pd.DataFrame)
    assert 'id' in df.columns

@pytest.mark.parametrize("root_id", [TEST_ROOT_ID])
def test_find_anchor_loc(root_id) -> None:
    """Test find_anchor_loc returns DataFrame with anchor columns.
    Ensures the function returns a DataFrame with root_id and coordinates.
    """
    df = l2.find_anchor_loc(root_id)
    assert isinstance(df, pd.DataFrame)
    assert set(['root_id', 'x', 'y', 'z']).issubset(df.columns)

@pytest.mark.parametrize("root_id", [TEST_ROOT_ID])
def test_get_l2_graph(root_id) -> None:
    """Test get_l2_graph returns a graph object with nodes.
    Checks that the returned object is a graph and has nodes attribute.
    """
    G = l2.get_l2_graph(root_id)
    assert G is not None
    assert hasattr(G, 'nodes')

@pytest.mark.parametrize("root_id", [TEST_ROOT_ID])
def test_get_l2_skeleton(root_id) -> None:
    """Test get_l2_skeleton returns a skeleton or NeuronList.
    Ensures the function returns a valid skeleton object for the neuron.
    """
    skel = l2.get_l2_skeleton(root_id)
    assert skel is not None
    assert hasattr(skel, 'nodes') or isinstance(skel, navis.NeuronList)

@pytest.mark.parametrize("root_id", [TEST_ROOT_ID])
def test_get_l2_dotprops(root_id) -> None:
    """Test get_l2_dotprops returns a NeuronList of Dotprops.
    Verifies that dotprops are generated and returned in a NeuronList.
    """
    dps = l2.get_l2_dotprops(root_id)
    assert isinstance(dps, navis.NeuronList)
    assert all(isinstance(dp, navis.Dotprops) for dp in dps)

@pytest.mark.parametrize("root_id", [TEST_ROOT_ID])
def test_get_l2_meshes(root_id) -> None:
    """Test get_l2_meshes returns a NeuronList of meshes.
    Checks that mesh objects are returned and have vertices attribute.
    """
    meshes = l2.get_l2_meshes(root_id)
    assert isinstance(meshes, navis.NeuronList)
    assert all(hasattr(m, 'vertices') for m in meshes)

def test__get_l2_centroids() -> None:
    """Test _get_l2_centroids returns a dict of centroids.
    Uses get_l2_meshes to get IDs, then checks centroid computation.
    """
    meshes = l2.get_l2_meshes(TEST_ROOT_ID)
    l2_ids = [str(m.id) for m in meshes]
    vol = l2.get_cloudvolume()
    centroids = l2._get_l2_centroids(l2_ids, vol)
    assert isinstance(centroids, dict)
    assert all(isinstance(v, np.ndarray) for v in centroids.values())

def test_chunks_to_nm() -> None:
    """Test chunks_to_nm returns correct shape and type.
    Checks that chunk indices are mapped to nanometer coordinates.
    """
    vol = l2.get_cloudvolume()
    arr = np.array([[0, 0, 0], [1, 1, 1]])
    nm = l2.chunks_to_nm(arr, vol)
    assert isinstance(nm, np.ndarray)
    assert nm.shape == (2, 3)

def test_add_skeleton_layer_with_treeneuron() -> None:
    """Test add_skeleton_layer with a single navis.TreeNeuron."""
    skel = l2.get_l2_skeleton(TEST_ROOT_ID)
    if isinstance(skel, navis.NeuronList):
        skel = skel[0]
    scene = {}
    result = l2.add_skeleton_layer(skel, scene)
    assert isinstance(result, dict)
    assert result is not scene  # scene should be copied


def test_add_skeleton_layer_with_neuronlist() -> None:
    """Test add_skeleton_layer with a navis.NeuronList of length 1."""
    skel = l2.get_l2_skeleton(TEST_ROOT_ID)
    if not isinstance(skel, navis.NeuronList):
        skel = navis.NeuronList([skel])
    scene = {}
    result = l2.add_skeleton_layer(skel, scene)
    assert isinstance(result, dict)


def test_add_skeleton_layer_with_dataframe() -> None:
    """Test add_skeleton_layer with a DataFrame of nodes."""
    skel = l2.get_l2_skeleton(TEST_ROOT_ID)
    if isinstance(skel, navis.NeuronList):
        skel = skel[0]
    nodes = skel.nodes
    scene = {}
    result = l2.add_skeleton_layer(nodes, scene)
    assert isinstance(result, dict)


def test_add_skeleton_layer_invalid_scene() -> None:
    """Test add_skeleton_layer raises TypeError for non-dict scene."""
    skel = l2.get_l2_skeleton(TEST_ROOT_ID)
    if isinstance(skel, navis.NeuronList):
        skel = skel[0]
    with pytest.raises(TypeError):
        l2.add_skeleton_layer(skel, None)


def test_add_skeleton_layer_invalid_type() -> None:
    """Test add_skeleton_layer raises TypeError for unsupported skeleton type."""
    scene = {}
    with pytest.raises(TypeError):
        l2.add_skeleton_layer("not_a_skeleton", scene)


def test_add_skeleton_layer_neuronlist_multiple() -> None:
    """Test add_skeleton_layer raises ValueError for NeuronList with >1 neuron."""
    skel1 = l2.get_l2_skeleton(TEST_ROOT_ID)
    skel2 = l2.get_l2_skeleton(TEST_ROOT_ID)
    nl = navis.NeuronList([skel1, skel2]) if not isinstance(skel1, navis.NeuronList) else navis.NeuronList([skel1[0], skel2[0]])
    scene = {}
    with pytest.raises(ValueError):
        l2.add_skeleton_layer(nl, scene)

