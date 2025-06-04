"""Check mesh conversion."""

import filecmp

from mammos_mumag.tofly import convert


def test_mesh_conversion(DATA, tmp_path):
    """Test mesh conversion."""
    convert(DATA / "mesh.unv", tmp_path / "mesh.fly")
    assert filecmp.cmp(tmp_path / "mesh.fly", DATA / "unvtofly" / "mesh.fly")
