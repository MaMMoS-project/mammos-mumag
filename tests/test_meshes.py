"""Test mesh module."""

import pytest

from mammos_mumag.mesh import Mesh, find_mesh


def test_mesh_no_matches():
    """Test Mesh creation with no matches in database."""
    with pytest.raises(ValueError):
        Mesh("cube131")


def test_mesh_too_many_matches():
    """Test Mesh creation with too many matches in database."""
    with pytest.raises(ValueError):
        Mesh("cube40")


def test_mesh_wrong_format():
    """Try create mesh with wrong format."""
    mesh = Mesh("cube20_singlegrain_msize2")
    with pytest.raises(ValueError):
        mesh.write("mesh.med")
    with pytest.raises(ValueError):
        mesh.write("mesh.unv")


@pytest.mark.parametrize("mesh_name", find_mesh())
def test_mesh_download_all_meshes(mesh_name, tmp_path):
    """Test that all meshes are downloadable."""
    Mesh(mesh_name).write(tmp_path / f"{mesh_name}.fly")
