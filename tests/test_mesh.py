import pytest
import numpy as np
from crantpy.viz import mesh

# Example root IDs for testing
TEST_ROOT_ID = 576460752681552812
TEST_ROOT_IDS = [576460752662526105, 576460752730083020]


def test_get_mesh_neuron_single():
    """Test fetching a single mesh neuron."""
    n = mesh.get_mesh_neuron(TEST_ROOT_ID, dataset="latest", threads=1)
    assert hasattr(n, "trimesh"), "Returned object should have a 'trimesh' attribute."
    assert n.id == TEST_ROOT_ID or str(n.id) == str(TEST_ROOT_ID)


def test_get_mesh_neuron_batch_multiple_threads():
    """Test fetching multiple mesh neurons as a batch."""
    nlist = mesh.get_mesh_neuron(TEST_ROOT_IDS, dataset="latest", threads=2)
    assert hasattr(nlist, "__iter__"), "Returned object should be iterable."
    assert len(nlist) == len(TEST_ROOT_IDS)
    for n in nlist:
        assert hasattr(n, "trimesh")


def test_get_mesh_neuron_batch_single_thread():
    """Test fetching multiple mesh neurons as a batch with a single thread."""
    nlist = mesh.get_mesh_neuron(TEST_ROOT_IDS, dataset="latest", threads=1)
    assert hasattr(nlist, "__iter__"), "Returned object should be iterable."
    assert len(nlist) == len(TEST_ROOT_IDS)
    for n in nlist:
        assert hasattr(n, "trimesh")


def test_get_mesh_neuron_invalid():
    """Test error handling for invalid neuron input."""
    with pytest.raises(ValueError):
        mesh.get_mesh_neuron({"invalid": "input"})


def test_detect_soma_single():
    """Test soma detection for a single neuron."""
    n = mesh.get_mesh_neuron(TEST_ROOT_ID, dataset="latest", threads=1)
    coords = mesh.detect_soma(n)
    assert isinstance(coords, (list, np.ndarray))
    assert len(coords) == 3


def test_detect_soma_batch_multiple_threads():
    """Test soma detection for a batch of neurons."""
    nlist = mesh.get_mesh_neuron(TEST_ROOT_IDS, dataset="latest", threads=2)
    coords = mesh.detect_soma(nlist)
    assert isinstance(coords, np.ndarray)
    assert coords.shape == (len(TEST_ROOT_IDS), 3)


def test_detect_soma_batch_single_thread():
    """Test soma detection for a batch of neurons with a single thread."""
    nlist = mesh.get_mesh_neuron(TEST_ROOT_IDS, dataset="latest", threads=1)
    coords = mesh.detect_soma(nlist)
    assert isinstance(coords, np.ndarray)
    assert coords.shape == (len(TEST_ROOT_IDS), 3)


def test_detect_soma_no_soma():
    """Test detect_soma returns [None, None, None] for invalid input."""
    coords = mesh.detect_soma("not_a_real_id")
    assert isinstance(coords, (list, np.ndarray))
    assert len(coords) == 3
    # Accept either [None, None, None] or [0, 0, 0] as valid outputs
    # Normalize coords to a list for comparison
    coords_list = list(coords)
    assert coords_list == [None, None, None] or coords_list == [
        0,
        0,
        0,
    ], f"Expected [None, None, None] or [0, 0, 0], got {coords_list}"
