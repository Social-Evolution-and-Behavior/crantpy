# -*- coding: utf-8 -*-
"""Offline unit tests for crantpy.viz.skeletonize.

These cover the pure, network-free helpers -- in particular the
dict-to-TreeNeuron conversion (``_create_node_info_dict`` /
``_swc_dict_to_dataframe``), radius-based soma detection, and the
dataset/signature plumbing. They do not require CAVE credentials or any
remote access.
"""

import inspect

import numpy as np
import pandas as pd
import navis
import pytest

from crantpy.viz import skeletonize as S


# ---------------------------------------------------------------------------
# _create_node_info_dict / _swc_dict_to_dataframe
# ---------------------------------------------------------------------------
def _roots(node_info):
    return sorted(k for k, v in node_info.items() if v["Parent"] == -1)


def _parent_count(node_info):
    """Number of nodes that have a parent (Parent != -1)."""
    return sum(1 for v in node_info.values() if v["Parent"] != -1)


def _as_treeneuron(node_info):
    df = S._swc_dict_to_dataframe(node_info)
    return navis.TreeNeuron(df, units="1 nm", id=1)


def test_node_info_linear_chain_meshparty_orientation():
    """meshparty emits edges as [child, parent]; the root must be vertex 0."""
    verts = np.array([[i, 0, 0] for i in range(4)], dtype=float)
    # child -> parent edges for chain 0-1-2-3 rooted at 0
    edges = np.array([[1, 0], [2, 1], [3, 2]])
    ni = S._create_node_info_dict(verts, edges)

    assert _roots(ni) == [0]  # vertex 0 is the single root
    # exactly one parent per non-root node
    assert _parent_count(ni) == 3
    # Parent links are 1-indexed (PointNo); node i's parent is i-1
    assert ni[1]["Parent"] == 1
    assert ni[2]["Parent"] == 2
    assert ni[3]["Parent"] == 3


def test_node_info_branch_node_keeps_all_children():
    """Regression: branch points must not lose children to parent overwrites."""
    # Y shape: 0-1, 1-2, 1-3, 3-4  (vertex 1 is a branch with children 2 and 3)
    verts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 1, 0], [1, 2, 0]], float)
    edges = np.array([[1, 0], [2, 1], [3, 1], [4, 3]])  # [child, parent]
    ni = S._create_node_info_dict(verts, edges)

    assert _roots(ni) == [0]
    # children of the branch node (vertex 1 == PointNo 2) are vertices 2 and 3
    children_of_branch = sorted(k for k, v in ni.items() if v["Parent"] == 2)
    assert children_of_branch == [2, 3]
    # resulting neuron is a single connected tree with one root
    tn = _as_treeneuron(ni)
    assert tn.n_nodes == 5
    assert len(np.atleast_1d(tn.root)) == 1


def test_node_info_orientation_invariant():
    """Reversing edge orientation must yield the same rooted tree."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 1, 0], [1, 2, 0]], float)
    edges_cp = np.array([[1, 0], [2, 1], [3, 1], [4, 3]])  # [child, parent]
    edges_pc = edges_cp[:, ::-1]  # [parent, child]

    ni_cp = S._create_node_info_dict(verts, edges_cp)
    ni_pc = S._create_node_info_dict(verts, edges_pc)

    parents_cp = {k: v["Parent"] for k, v in ni_cp.items()}
    parents_pc = {k: v["Parent"] for k, v in ni_pc.items()}
    assert parents_cp == parents_pc
    assert _roots(ni_cp) == [0]


def test_node_info_duplicate_edges_are_safe():
    """Duplicate / undirected edges must not create extra parents or cycles."""
    verts = np.array([[i, 0, 0] for i in range(3)], dtype=float)
    edges = np.array([[1, 0], [0, 1], [2, 1], [2, 1]])  # duplicates + reversed
    ni = S._create_node_info_dict(verts, edges)
    assert _roots(ni) == [0]
    assert _parent_count(ni) == 2  # one parent each for nodes 1 and 2


def test_node_info_disconnected_components_get_one_root_each():
    """Each connected component must get exactly one root."""
    verts = np.array([[i, 0, 0] for i in range(5)], dtype=float)
    # component A: 0-1-2 ; component B: 3-4
    edges = np.array([[1, 0], [2, 1], [4, 3]])
    ni = S._create_node_info_dict(verts, edges)
    assert _roots(ni) == [0, 3]
    assert _parent_count(ni) == 3  # 5 nodes, 2 components -> 3 parented nodes


def test_node_info_isolated_vertex_is_its_own_root():
    verts = np.array([[0, 0, 0], [1, 0, 0], [5, 5, 5]], dtype=float)
    edges = np.array([[1, 0]])  # vertex 2 has no edges
    ni = S._create_node_info_dict(verts, edges)
    assert 2 in _roots(ni)
    assert ni[2]["Type"] == 1  # isolated node classified as root


def test_node_info_type_classification():
    """Root -> 1, leaf/endpoint -> 6, intermediate/branch -> 3."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 1, 0]], float)
    edges = np.array([[1, 0], [2, 1], [3, 1]])  # root 0, branch 1, leaves 2 & 3
    ni = S._create_node_info_dict(verts, edges)
    assert ni[0]["Type"] == 1  # root
    assert ni[1]["Type"] == 3  # branch (degree 3)
    assert ni[2]["Type"] == 6  # leaf
    assert ni[3]["Type"] == 6  # leaf


def test_node_info_mixed_orientation_within_tree():
    """Edges with inconsistent orientation within one tree still yield one root."""
    verts = np.array([[i, 0, 0] for i in range(4)], dtype=float)
    # chain 0-1-2-3 given as [parent,child], [child,parent], [parent,child]
    edges = np.array([[0, 1], [2, 1], [2, 3]])
    ni = S._create_node_info_dict(verts, edges)
    assert _roots(ni) == [0]
    assert _parent_count(ni) == 3
    tn = _as_treeneuron(ni)
    assert tn.n_nodes == 4
    assert len(np.atleast_1d(tn.root)) == 1


def test_swc_dict_to_dataframe_schema():
    verts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    edges = np.array([[1, 0], [2, 1]])
    df = S._swc_dict_to_dataframe(S._create_node_info_dict(verts, edges))
    assert list(df.columns) == ["PointNo", "Type", "X", "Y", "Z", "Radius", "Parent"]
    assert df["PointNo"].tolist() == [1, 2, 3]  # sorted by PointNo
    for col in ["X", "Y", "Z", "Radius"]:
        assert df[col].dtype == float


# ---------------------------------------------------------------------------
# detect_soma_skeleton
# ---------------------------------------------------------------------------
def _linear_neuron(radii):
    n = len(radii)
    df = pd.DataFrame(
        {
            "node_id": list(range(1, n + 1)),
            "parent_id": [-1] + list(range(1, n)),
            "x": np.arange(n, dtype=float),
            "y": np.zeros(n),
            "z": np.zeros(n),
            "radius": np.asarray(radii, dtype=float),
        }
    )
    return navis.TreeNeuron(df, units="1 nm", id=1)


def test_detect_soma_skeleton_no_radius_returns_none():
    tn = _linear_neuron([100.0] * 5)
    tn.nodes.drop(columns=["radius"], inplace=True)
    assert S.detect_soma_skeleton(tn) is None


def test_detect_soma_skeleton_no_large_nodes_returns_none():
    tn = _linear_neuron([100.0] * 6)  # all below default min_rad=800
    assert S.detect_soma_skeleton(tn) is None


def test_detect_soma_skeleton_finds_large_radius_blob():
    # three consecutive large-radius nodes -> soma candidate; the largest wins
    radii = [100, 100, 900, 1500, 1000, 100, 100]
    tn = _linear_neuron(radii)
    soma = S.detect_soma_skeleton(tn, min_rad=800, N=3)
    assert soma is not None
    # the detected node must be the largest-radius one (node_id 4, radius 1500)
    assert tn.nodes.set_index("node_id").loc[soma, "radius"] == 1500


def test_detect_soma_skeleton_large_node_at_segment_start():
    """Regression for the ``any(is_big)`` bug.

    navis orders each segment leaf-first, so a lone large node at a leaf lands
    at position 0 of its segment. The old ``not any(is_big)`` test treated that
    index value 0 as falsy and silently skipped the segment; the corrected
    ``is_big.size == 0`` check detects it.
    """
    radii = [100.0, 100.0, 100.0, 2000.0]  # the leaf (node_id 4) is the big one
    tn = _linear_neuron(radii)
    soma = S.detect_soma_skeleton(tn, min_rad=800, N=1)
    assert soma is not None
    assert tn.nodes.set_index("node_id").loc[soma, "radius"] == 2000.0


# ---------------------------------------------------------------------------
# detect_soma_mesh
# ---------------------------------------------------------------------------
def test_detect_soma_mesh_none_returns_empty():
    assert S.detect_soma_mesh(None).size == 0


def test_detect_soma_mesh_too_few_vertices_returns_empty():
    import trimesh

    m = trimesh.creation.box(extents=(1, 1, 1))  # 8 vertices < 100
    out = S.detect_soma_mesh(m)
    assert isinstance(out, np.ndarray)
    assert out.size == 0


def test_detect_soma_mesh_neighbor_count_matches_reference():
    """Lock the vectorized neighbor count against the old per-vertex loop.

    The patch replaced an O(n^2) ``tree.query(v, k=n, distance_upper_bound=4000)``
    loop with ``tree.query_ball_point(..., r=4000, return_length=True)``; both
    count vertices within 4 µm (including the vertex itself), so the per-vertex
    counts must be identical.
    """
    import trimesh
    from scipy.spatial import cKDTree

    m = trimesh.creation.icosphere(subdivisions=2, radius=3000.0)
    tree = cKDTree(m.vertices)
    n = len(m.vertices)
    reference = np.array(
        [
            int(np.isfinite(tree.query(v, k=n, distance_upper_bound=4000)[0]).sum())
            for v in m.vertices
        ]
    )
    vectorized = np.asarray(
        tree.query_ball_point(m.vertices, r=4000, return_length=True)
    )
    assert np.array_equal(vectorized, reference)


# ---------------------------------------------------------------------------
# get_skeletons -- dedup + ordering contract (offline, mocked)
# ---------------------------------------------------------------------------
def test_get_skeletons_dedups_and_preserves_order(monkeypatch):
    """Duplicate root_ids are collapsed; output is reindexed to unique order."""

    class _DummySkeletonSvc:
        def get_skeleton(self, root_id, output_format="dict"):
            return None  # force the skeletonize_neuron fallback

    class _DummyClient:
        skeleton = _DummySkeletonSvc()

    calls = []

    def _fake_skeletonize(client, root_id, progress=False, dataset=None, **kw):
        calls.append(int(root_id))
        tn = _linear_neuron([1.0, 1.0])
        tn.id = int(root_id)
        return tn

    monkeypatch.setattr(S, "create_client", lambda dataset=None: _DummyClient())
    monkeypatch.setattr(S, "skeletonize_neuron", _fake_skeletonize)

    nl = S.get_skeletons([123, 123, 456], progress=False, max_threads=1)

    assert len(nl) == 2
    assert list(nl.id) == [123, 456]
    assert sorted(calls) == [123, 456]  # each unique id fetched exactly once


# ---------------------------------------------------------------------------
# dataset / signature plumbing
# ---------------------------------------------------------------------------
def test_skeletonize_neuron_has_dataset_param():
    params = inspect.signature(S.skeletonize_neuron).parameters
    assert "dataset" in params


def test_skeletonize_neurons_parallel_has_dataset_param():
    params = inspect.signature(S.skeletonize_neurons_parallel).parameters
    assert "dataset" in params


def test_assert_id_match_warns_and_is_noop():
    tn = _linear_neuron([100.0, 100.0, 100.0])
    with pytest.warns(UserWarning):
        S._assert_id_match(tn, root_id=123, client=None)


def test_assert_id_match_rejects_zero_id():
    tn = _linear_neuron([100.0, 100.0, 100.0])
    with pytest.raises(ValueError):
        S._assert_id_match(tn, root_id=0, client=None)
