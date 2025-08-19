"""Test mesh module."""

import filecmp

import pytest
import requests

from mammos_mumag.mesh import Mesh, find_mesh, get_mesh_json


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
def test_mesh_get_all_meshes(mesh_name):
    """Test that all meshes are reachable."""
    zenodo_url = get_mesh_json()["metadata"]["zenodo_url"]
    mesh_url = f"{zenodo_url}/files/{mesh_name}.fly"
    res = requests.get(mesh_url)
    assert res.status_code == 200


def test_mesh_write(tmp_path):
    """Test write method."""
    mesh = Mesh("cube20_singlegrain_msize1")
    mesh.write(tmp_path / "mesh.fly")
    assert (tmp_path / "mesh.fly").is_file()
    zenodo_url = get_mesh_json()["metadata"]["zenodo_url"]
    mesh_url = f"{zenodo_url}/files/{mesh.name}.fly"
    with open(tmp_path / "temp.fly", "wb") as f:
        f.write(requests.get(mesh_url).content)
    assert filecmp.cmp(tmp_path / "mesh.fly", tmp_path / "temp.fly")
