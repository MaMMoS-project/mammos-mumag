"""Test mesh module."""

import pathlib

import pytest
from platformdirs import user_cache_dir

from mammos_mumag.mesh import Mesh, find_mesh


def test_mesh_no_matches():
    """Test Mesh creation with no matches in database."""
    with pytest.raises(RuntimeError):
        Mesh("cube131")


def test_mesh_too_many_matches():
    """Test Mesh creation with too many matches in database."""
    with pytest.raises(RuntimeError):
        Mesh("cube40")


def test_mesh_wrong_extension(tmp_path):
    """Try create mesh with wrong extension."""
    mesh = Mesh("cube20_singlegrain_msize2")
    with pytest.warns(UserWarning):
        mesh.write(tmp_path / "mesh.med")
    with pytest.warns(UserWarning):
        mesh.write(tmp_path / "mesh.txt")


@pytest.mark.parametrize("mesh_name", find_mesh())
def test_mesh_download_all_meshes(mesh_name, tmp_path):
    """Test that all meshes are downloadable.

    Furthermore, it checks that each mesh (except for ``cube20_singlegrain_msize2``)
    is cached after download. One mesh is excluded because it is packaged with
    ``mammos_mumag`` and it will not be be downloaded to cache.
    """
    Mesh(mesh_name).write(tmp_path / f"{mesh_name}.fly")
    if mesh_name != "cube20_singlegrain_msize2":
        assert (
            pathlib.Path(user_cache_dir("mammos_mumag")) / f"{mesh_name}.fly"
        ).is_file()


def test_download_from_keeper(tmp_path):
    """Test downloading meshes from Keeper."""
    mesh = Mesh("cube20_singlegrain_msize2")
    mesh._download_from_keeper(tmp_path / "mesh.fly", extension=".fly")
    assert (tmp_path / "mesh.fly").is_file()
    mesh._download_from_keeper(tmp_path / "mesh.med", extension=".med")
    assert (tmp_path / "mesh.med").is_file()
    mesh._download_from_keeper(tmp_path / "mesh.unv", extension=".unv")
    assert (tmp_path / "mesh.unv").is_file()
    with pytest.raises(RuntimeError):
        mesh._download_from_keeper(tmp_path / "error.txt", extension=".txt")
