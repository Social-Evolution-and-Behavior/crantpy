import pytest
import numpy as np
import pyvista as pv
import trimesh as tm
from crantpy.viz import mesh

# Example root IDs for testing
TEST_ROOT_ID = 576460752681552812
TEST_ROOT_IDS = [576460752664524086, 576460752662516321]
# For typing compatibility with Union types
TEST_ROOT_IDS_MIXED = [576460752664524086, "576460752662516321"]


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


def test_load_whole_brain_mesh():
    """Test loading the whole brain mesh."""
    brain_mesh = mesh.load_whole_brain_mesh()

    # Check that it returns a trimesh object
    assert isinstance(brain_mesh, tm.Trimesh), "Should return a trimesh.Trimesh object"

    # Check that it has vertices and faces
    assert hasattr(brain_mesh, "vertices"), "Mesh should have vertices"
    assert hasattr(brain_mesh, "faces"), "Mesh should have faces"

    # Check that vertices and faces are not empty
    assert len(brain_mesh.vertices) > 0, "Mesh should have vertices"
    assert len(brain_mesh.faces) > 0, "Mesh should have faces"

    # Check that vertices are 3D coordinates
    assert brain_mesh.vertices.shape[1] == 3, "Vertices should be 3D coordinates"

    # Check that faces are triangles
    assert brain_mesh.faces.shape[1] == 3, "Faces should be triangles"


def test_get_brain_mesh_scene_single_neuron():
    """Test creating a brain mesh scene with a single neuron."""
    plotter = mesh.get_brain_mesh_scene(
        TEST_ROOT_ID,
        dataset=None,
        threads=1,
        progress=False,
        backend="static",  # Use static backend for testing
    )

    # Check that it returns a PyVista plotter
    assert isinstance(plotter, pv.Plotter), "Should return a pv.Plotter object"

    # Check that the plotter has actors (meshes) added
    assert len(plotter.actors) > 0, "Plotter should have actors"

    # Should have at least 2 actors: brain + 1 neuron
    assert (
        len(plotter.actors) >= 2
    ), "Should have brain mesh and at least one neuron mesh"


def test_get_brain_mesh_scene_multiple_neurons():
    """Test creating a brain mesh scene with multiple neurons."""
    # Use the mixed type list to satisfy the Union type requirements
    plotter = mesh.get_brain_mesh_scene(
        TEST_ROOT_IDS_MIXED,
        dataset=None,
        threads=1,
        progress=False,
        backend="static",  # Use static backend for testing
    )

    # Check that it returns a PyVista plotter
    assert isinstance(plotter, pv.Plotter), "Should return a pv.Plotter object"

    # Check that the plotter has actors (meshes) added
    assert len(plotter.actors) > 0, "Plotter should have actors"

    # Should have brain mesh + multiple neuron meshes
    assert (
        len(plotter.actors) >= len(TEST_ROOT_IDS_MIXED) + 1
    ), "Should have brain mesh and multiple neuron meshes"


def test_get_brain_mesh_scene_parameters():
    """Test get_brain_mesh_scene with different parameters."""
    plotter = mesh.get_brain_mesh_scene(
        TEST_ROOT_ID,
        dataset=None,
        brain_mesh_color="blue",
        brain_mesh_alpha=0.2,
        neuron_mesh_alpha=0.8,
        backend="static",
        progress=False,
    )

    # Check that it returns a PyVista plotter
    assert isinstance(plotter, pv.Plotter), "Should return a pv.Plotter object"

    # Check that actors were added
    assert len(plotter.actors) >= 2, "Should have brain mesh and neuron mesh"


def test_get_brain_mesh_scene_invalid_input():
    """Test get_brain_mesh_scene with invalid input."""
    with pytest.raises(ValueError):
        mesh.get_brain_mesh_scene("not_a_real_id_that_will_fail")


def test_get_brain_mesh_scene_no_neurons():
    """Test creating a brain mesh scene without neurons (only brain mesh)."""
    plotter = mesh.get_brain_mesh_scene(
        neurons=None,
        dataset=None,
        backend="static",
        progress=False,
    )

    # Check that it returns a PyVista plotter
    assert isinstance(plotter, pv.Plotter), "Should return a pv.Plotter object"

    # Check that the plotter has actors (meshes) added
    assert len(plotter.actors) > 0, "Plotter should have actors"

    # Should have just the brain mesh (1 actor)
    assert len(plotter.actors) >= 1, "Should have at least the brain mesh"


def test_get_brain_mesh_scene_with_neuropil_meshes():
    """Test creating a brain mesh scene with neuropil meshes."""
    plotter = mesh.get_brain_mesh_scene(
        neurons=TEST_ROOT_ID,
        neuropil_meshes=["antennal_lobe_left", "antennal_lobe_right"],
        neuropil_mesh_colors=["red", "blue"],
        neuropil_mesh_alphas=[0.4, 0.5],
        dataset=None,
        backend="static",
        progress=False,
    )

    # Check that it returns a PyVista plotter
    assert isinstance(plotter, pv.Plotter), "Should return a pv.Plotter object"

    # Check that the plotter has actors (meshes) added
    assert len(plotter.actors) > 0, "Plotter should have actors"

    # Should have brain mesh + neuron + 2 neuropil meshes (at least 4 actors)
    assert (
        len(plotter.actors) >= 4
    ), "Should have brain mesh, neuron mesh, and 2 neuropil meshes"


def test_get_brain_mesh_scene_neuropil_only():
    """Test creating a brain mesh scene with only neuropil meshes (no neurons)."""
    plotter = mesh.get_brain_mesh_scene(
        neurons=None,
        neuropil_meshes=["antennal_lobe_left", "mushroom_body_medial_calyx_left"],
        neuropil_mesh_colors=["green", "purple"],
        neuropil_mesh_alphas=0.5,  # Single value applied to all
        dataset=None,
        backend="static",
        progress=False,
    )

    # Check that it returns a PyVista plotter
    assert isinstance(plotter, pv.Plotter), "Should return a pv.Plotter object"

    # Check that the plotter has actors (meshes) added
    assert len(plotter.actors) > 0, "Plotter should have actors"

    # Should have brain mesh + 2 neuropil meshes (at least 3 actors)
    assert (
        len(plotter.actors) >= 3
    ), "Should have brain mesh and 2 neuropil meshes"


def test_get_brain_mesh_scene_single_neuropil_string():
    """Test creating a brain mesh scene with a single neuropil mesh as string."""
    plotter = mesh.get_brain_mesh_scene(
        neurons=None,
        neuropil_meshes="antennal_lobe_left",
        neuropil_mesh_colors="orange",
        neuropil_mesh_alphas=0.6,
        dataset=None,
        backend="static",
        progress=False,
    )

    # Check that it returns a PyVista plotter
    assert isinstance(plotter, pv.Plotter), "Should return a pv.Plotter object"

    # Should have brain mesh + 1 neuropil mesh
    assert len(plotter.actors) >= 2, "Should have brain mesh and 1 neuropil mesh"


def test_get_brain_mesh_scene_neuropil_trimesh_object():
    """Test creating a brain mesh scene with neuropil as trimesh object."""
    # Load a neuropil mesh first
    neuropil_trimesh = mesh.load_neuropil_mesh("antennal_lobe_left")
    
    plotter = mesh.get_brain_mesh_scene(
        neurons=None,
        neuropil_meshes=neuropil_trimesh,
        neuropil_mesh_colors="cyan",
        neuropil_mesh_alphas=0.7,
        dataset=None,
        backend="static",
        progress=False,
    )

    # Check that it returns a PyVista plotter
    assert isinstance(plotter, pv.Plotter), "Should return a pv.Plotter object"

    # Should have brain mesh + 1 neuropil mesh
    assert len(plotter.actors) >= 2, "Should have brain mesh and 1 neuropil mesh"


def test_get_brain_mesh_scene_mixed_neuropil_types():
    """Test creating a brain mesh scene with mixed neuropil types (string and trimesh)."""
    # Load a neuropil mesh first
    neuropil_trimesh = mesh.load_neuropil_mesh("antennal_lobe_left")
    
    plotter = mesh.get_brain_mesh_scene(
        neurons=TEST_ROOT_ID,
        neuropil_meshes=[neuropil_trimesh, "antennal_lobe_right"],
        neuropil_mesh_alphas=[0.3, 0.4],
        dataset=None,
        backend="static",
        progress=False,
    )

    # Check that it returns a PyVista plotter
    assert isinstance(plotter, pv.Plotter), "Should return a pv.Plotter object"

    # Should have brain + neuron + 2 neuropil meshes
    assert len(plotter.actors) >= 4, "Should have all meshes added"


def test_get_brain_mesh_scene_neuropil_alpha_mismatch():
    """Test that mismatched alpha list length raises ValueError."""
    with pytest.raises(ValueError) as excinfo:
        mesh.get_brain_mesh_scene(
            neurons=None,
            neuropil_meshes=["antennal_lobe_left", "antennal_lobe_right"],
            neuropil_mesh_alphas=[0.3],  # Only 1 alpha for 2 meshes
            backend="static",
            progress=False,
        )
    
    assert "neuropil_mesh_alphas" in str(excinfo.value)


def test_get_brain_mesh_scene_neuropil_color_mismatch():
    """Test that mismatched color list length raises ValueError."""
    with pytest.raises(ValueError) as excinfo:
        mesh.get_brain_mesh_scene(
            neurons=None,
            neuropil_meshes=["antennal_lobe_left", "antennal_lobe_right"],
            neuropil_mesh_colors=["red"],  # Only 1 color for 2 meshes
            backend="static",
            progress=False,
        )
    
    assert "neuropil_mesh_colors" in str(excinfo.value)


def test_load_neuropil_mesh_valid():
    """Test loading a valid neuropil mesh."""
    neuropil_mesh = mesh.load_neuropil_mesh("antennal_lobe_left")
    
    # Check that it returns a trimesh object
    assert isinstance(neuropil_mesh, tm.Trimesh), "Should return a trimesh.Trimesh object"
    
    # Check that it has vertices and faces
    assert hasattr(neuropil_mesh, "vertices"), "Mesh should have vertices"
    assert hasattr(neuropil_mesh, "faces"), "Mesh should have faces"
    
    # Check that vertices and faces are not empty
    assert len(neuropil_mesh.vertices) > 0, "Mesh should have vertices"
    assert len(neuropil_mesh.faces) > 0, "Mesh should have faces"
    
    # Check that vertices are 3D coordinates
    assert neuropil_mesh.vertices.shape[1] == 3, "Vertices should be 3D coordinates"
    
    # Check that faces are triangles
    assert neuropil_mesh.faces.shape[1] == 3, "Faces should be triangles"


def test_load_neuropil_mesh_multiple():
    """Test loading multiple different neuropil meshes."""
    meshes = [
        "antennal_lobe_left",
        "antennal_lobe_right",
        "mushroom_body_pedunculus_and_lobes_right"
    ]
    
    for mesh_name in meshes:
        neuropil_mesh = mesh.load_neuropil_mesh(mesh_name)
        assert isinstance(neuropil_mesh, tm.Trimesh), f"Should return a trimesh for {mesh_name}"
        assert len(neuropil_mesh.vertices) > 0, f"Mesh {mesh_name} should have vertices"
        assert len(neuropil_mesh.faces) > 0, f"Mesh {mesh_name} should have faces"


def test_load_neuropil_mesh_invalid_label():
    """Test loading a neuropil mesh with an invalid label."""
    with pytest.raises(ValueError) as excinfo:
        mesh.load_neuropil_mesh("invalid_neuropil_name")
    
    assert "Invalid neuropil label" in str(excinfo.value)
    assert "Available labels are" in str(excinfo.value)


def test_load_neuropil_mesh_wrong_type():
    """Test loading a neuropil mesh with wrong type input."""
    with pytest.raises((ValueError, TypeError, AttributeError)):
        mesh.load_neuropil_mesh(123)  # Should be a string
