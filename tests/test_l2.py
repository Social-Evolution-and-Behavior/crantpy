import pytest
import numpy as np
import pandas as pd
import navis
from crantpy.viz import l2
from typing import Any, Dict, List, Set, Tuple

# Example root ID for testing
TEST_ROOT_ID = 576460752732354679
TEST_ROOT_IDS = [576460752715406504, 576460752749155108]


@pytest.mark.parametrize("root_id", [TEST_ROOT_ID])
def test_get_l2_info(root_id) -> None:
    """Test get_l2_info returns DataFrame with 'root_id' column.
    This verifies that get_l2_info fetches L2 info and returns a DataFrame
    with the expected columns for a given neuron root ID.
    """
    df = l2.get_l2_info(root_id)
    assert isinstance(df, pd.DataFrame)
    assert "root_id" in df.columns


@pytest.mark.parametrize("root_id", [TEST_ROOT_ID])
def test_get_l2_chunk_info(root_id) -> None:
    """Test get_l2_chunk_info returns DataFrame with 'l2_id' and 'root_id' columns.
    Ensures the function returns a DataFrame with the expected columns for a given neuron root ID.
    """
    df = l2.get_l2_chunk_info(root_id)
    assert isinstance(df, pd.DataFrame)
    assert "l2_id" in df.columns
    assert "root_id" in df.columns


@pytest.mark.parametrize("root_id", [TEST_ROOT_ID])
def test_find_anchor_loc(root_id) -> None:
    """Test find_anchor_loc returns DataFrame with anchor columns.
    Ensures the function returns a DataFrame with root_id and coordinates.
    """
    df = l2.find_anchor_loc(root_id)
    assert isinstance(df, pd.DataFrame)
    assert set(["root_id", "x", "y", "z"]).issubset(df.columns)


@pytest.mark.parametrize("root_id", [TEST_ROOT_ID])
def test_get_l2_graph(root_id) -> None:
    """Test get_l2_graph returns a graph object with nodes.
    Checks that the returned object is a graph and has nodes attribute.
    """
    G = l2.get_l2_graph(root_id)
    assert G is not None
    assert hasattr(G, "nodes")


@pytest.mark.parametrize("root_id", [TEST_ROOT_ID])
def test_get_l2_skeleton(root_id) -> None:
    """Test get_l2_skeleton returns a skeleton or NeuronList.
    Ensures the function returns a valid skeleton object for the neuron.
    """
    skel = l2.get_l2_skeleton(root_id)
    assert skel is not None
    assert hasattr(skel, "nodes") or isinstance(skel, navis.NeuronList)


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
    assert all(hasattr(m, "vertices") for m in meshes)


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


@pytest.mark.parametrize("root_ids", [TEST_ROOT_IDS])
def test_get_l2_info_batch(root_ids) -> None:
    """Test get_l2_info returns DataFrame with 'root_id' column for multiple neurons."""
    df = l2.get_l2_info(root_ids)
    assert isinstance(df, pd.DataFrame)
    assert "root_id" in df.columns
    expected_ids = set(str(rid) for rid in root_ids)
    actual_ids = set(df["root_id"].astype(str))
    assert expected_ids.issubset(actual_ids)


@pytest.mark.parametrize("root_ids", [TEST_ROOT_IDS])
def test_get_l2_chunk_info_batch(root_ids) -> None:
    """Test get_l2_chunk_info returns DataFrame with 'l2_id' and 'root_id' columns for multiple neurons."""
    df = l2.get_l2_chunk_info(root_ids)
    assert isinstance(df, pd.DataFrame)
    assert "l2_id" in df.columns
    assert "root_id" in df.columns
    assert set([str(rid) for rid in root_ids]).issubset(set(df["root_id"].astype(str)))


@pytest.mark.parametrize("root_ids", [TEST_ROOT_IDS])
def test_find_anchor_loc_batch(root_ids) -> None:
    """Test find_anchor_loc returns DataFrame with anchor columns for multiple neurons."""
    df = l2.find_anchor_loc(root_ids)
    assert isinstance(df, pd.DataFrame)
    assert set(["root_id", "x", "y", "z"]).issubset(df.columns)
    assert set([str(rid) for rid in root_ids]).issubset(set(df["root_id"].astype(str)))


@pytest.mark.parametrize("root_ids", [TEST_ROOT_IDS])
def test_get_l2_graph_batch(root_ids) -> None:
    """Test get_l2_graph returns a list of graph objects for multiple neurons."""
    Gs = l2.get_l2_graph(root_ids)
    assert isinstance(Gs, list)
    assert all(hasattr(G, "nodes") for G in Gs)
    assert len(Gs) == len(root_ids)


@pytest.mark.parametrize("root_ids", [TEST_ROOT_IDS])
def test_get_l2_skeleton_batch(root_ids) -> None:
    """Test get_l2_skeleton returns a NeuronList for multiple neurons."""
    skels = l2.get_l2_skeleton(root_ids)
    assert isinstance(skels, navis.NeuronList)
    assert len(skels) == len(root_ids)


@pytest.mark.parametrize("root_ids", [TEST_ROOT_IDS])
def test_get_l2_dotprops_batch(root_ids) -> None:
    """Test get_l2_dotprops returns a NeuronList of Dotprops for multiple neurons."""
    dps = l2.get_l2_dotprops(root_ids)
    assert isinstance(dps, navis.NeuronList)
    assert all(isinstance(dp, navis.Dotprops) for dp in dps)
    assert len(dps) == len(root_ids)
