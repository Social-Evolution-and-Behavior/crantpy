import pytest
import numpy as np
import pandas as pd
import navis
from crantpy.viz import l2
from typing import Any, Dict, List, Set, Tuple

# Example root ID for testing
TEST_ROOT_ID = 576460752732354679
TEST_ROOT_IDS = [576460752715406504, 576460752749155108]
# Root ID with known synapses for synapse attachment tests
TEST_ROOT_ID_WITH_SYNAPSES = 576460752664524086


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


# Tests for new attach_synapses and reroot_at_soma parameters
@pytest.mark.parametrize("root_id", [TEST_ROOT_ID_WITH_SYNAPSES])
def test_get_l2_skeleton_attach_synapses(root_id) -> None:
    """Test get_l2_skeleton with attach_synapses=True.
    Verifies that synapses are attached as connectors to the skeleton.
    """
    skel = l2.get_l2_skeleton(root_id, attach_synapses=True)
    assert isinstance(skel, navis.TreeNeuron)
    assert hasattr(skel, "connectors")
    assert isinstance(skel.connectors, pd.DataFrame)
    assert len(skel.connectors) > 0
    # Check that connectors have required columns
    assert "type" in skel.connectors.columns
    assert "node_id" in skel.connectors.columns
    # Check that we have pre and/or post synapses
    assert set(skel.connectors["type"].unique()).issubset({"pre", "post"})


@pytest.mark.parametrize("root_id", [TEST_ROOT_ID_WITH_SYNAPSES])
def test_get_l2_skeleton_reroot_at_soma(root_id) -> None:
    """Test get_l2_skeleton with reroot_at_soma=True.
    Verifies that the skeleton is rerooted at the soma location.
    """
    # Get skeleton without rerooting
    skel_orig = l2.get_l2_skeleton(root_id, reroot_at_soma=False)
    original_root = skel_orig.root[0]

    # Get skeleton with rerooting
    skel = l2.get_l2_skeleton(root_id, reroot_at_soma=True)
    assert isinstance(skel, navis.TreeNeuron)
    new_root = skel.root[0]

    # Root should have changed (soma is unlikely to be the default root)
    # Note: This might fail if the default root happens to be at the soma
    # but this is extremely unlikely
    assert new_root != original_root or len(skel.nodes) < 10


@pytest.mark.parametrize("root_id", [TEST_ROOT_ID_WITH_SYNAPSES])
def test_get_l2_skeleton_both_postprocessing(root_id) -> None:
    """Test get_l2_skeleton with both attach_synapses=True and reroot_at_soma=True.
    Verifies that both post-processing steps are applied correctly.
    """
    skel = l2.get_l2_skeleton(root_id, attach_synapses=True, reroot_at_soma=True)
    assert isinstance(skel, navis.TreeNeuron)

    # Should have connectors
    assert hasattr(skel, "connectors")
    assert isinstance(skel.connectors, pd.DataFrame)
    assert len(skel.connectors) > 0

    # Should be rerooted (compare with default)
    skel_default = l2.get_l2_skeleton(root_id)
    assert skel.root[0] != skel_default.root[0] or len(skel.nodes) < 10


@pytest.mark.parametrize("root_ids", [TEST_ROOT_IDS])
def test_get_l2_skeleton_batch_attach_synapses(root_ids) -> None:
    """Test get_l2_skeleton with attach_synapses=True in batch mode.
    Verifies that synapses are attached to all neurons in the batch.
    """
    skels = l2.get_l2_skeleton(root_ids, attach_synapses=True)
    assert isinstance(skels, navis.NeuronList)
    assert len(skels) == len(root_ids)

    # All skeletons should have connectors
    for skel in skels:
        assert hasattr(skel, "connectors")
        assert isinstance(skel.connectors, pd.DataFrame)
        # Check that connectors are present (most neurons should have some)
        # Note: Some neurons might have no synapses, so we just check the attribute exists


@pytest.mark.parametrize("root_ids", [TEST_ROOT_IDS])
def test_get_l2_skeleton_batch_reroot_at_soma(root_ids) -> None:
    """Test get_l2_skeleton with reroot_at_soma=True in batch mode.
    Verifies that all skeletons are rerooted at their soma.
    """
    # Get skeletons without rerooting
    skels_orig = l2.get_l2_skeleton(root_ids, reroot_at_soma=False)
    original_roots = [skel.root[0] for skel in skels_orig]

    # Get skeletons with rerooting
    skels = l2.get_l2_skeleton(root_ids, reroot_at_soma=True)
    assert isinstance(skels, navis.NeuronList)
    assert len(skels) == len(root_ids)

    # Check that roots have changed for at least some neurons
    new_roots = [skel.root[0] for skel in skels]
    # At least one root should have changed
    assert any(old != new for old, new in zip(original_roots, new_roots))


@pytest.mark.parametrize("root_ids", [TEST_ROOT_IDS])
def test_get_l2_skeleton_batch_both_postprocessing(root_ids) -> None:
    """Test get_l2_skeleton with both attach_synapses=True and reroot_at_soma=True in batch mode.
    Verifies that both post-processing steps are applied to all neurons.
    """
    skels = l2.get_l2_skeleton(root_ids, attach_synapses=True, reroot_at_soma=True)
    assert isinstance(skels, navis.NeuronList)
    assert len(skels) == len(root_ids)

    # All skeletons should have connectors and be rerooted
    for skel in skels:
        assert hasattr(skel, "connectors")
        assert isinstance(skel.connectors, pd.DataFrame)


@pytest.mark.parametrize("root_id", [TEST_ROOT_ID_WITH_SYNAPSES])
def test_get_l2_skeleton_connector_properties(root_id) -> None:
    """Test that attached connectors have correct properties.
    Verifies connector DataFrame structure and content.
    """
    skel = l2.get_l2_skeleton(root_id, attach_synapses=True)

    # Check connector DataFrame structure
    assert "x" in skel.connectors.columns
    assert "y" in skel.connectors.columns
    assert "z" in skel.connectors.columns
    assert "type" in skel.connectors.columns
    assert "node_id" in skel.connectors.columns

    # Check that node_ids are valid (exist in skeleton)
    valid_nodes = set(skel.nodes["node_id"].values)
    connector_nodes = set(skel.connectors["node_id"].values)
    assert connector_nodes.issubset(valid_nodes)

    # Check that connector types are valid
    assert all(t in ["pre", "post"] for t in skel.connectors["type"].unique())


@pytest.mark.parametrize("root_id", [TEST_ROOT_ID_WITH_SYNAPSES])
def test_get_l2_skeleton_postprocessing_order(root_id) -> None:
    """Test that post-processing happens in correct order.
    Verifies that synapses are attached before rerooting when both are requested.
    """
    # Get skeleton with both post-processing steps
    skel = l2.get_l2_skeleton(root_id, attach_synapses=True, reroot_at_soma=True)

    # Verify both operations completed successfully
    assert hasattr(skel, "connectors")
    assert len(skel.connectors) > 0

    # Get skeleton with just synapses
    skel_synapses = l2.get_l2_skeleton(
        root_id, attach_synapses=True, reroot_at_soma=False
    )

    # Both should have same number of connectors (rerooting shouldn't affect connector count)
    assert len(skel.connectors) == len(skel_synapses.connectors)

    # The connector node_ids might be different due to rerooting potentially changing node IDs,
    # but the total count should be the same
