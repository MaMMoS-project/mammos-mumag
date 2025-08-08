"""Check loop script."""

import numpy as np
import pyvista as pv
from mammos_entity.io import entities_from_file

from mammos_mumag.simulation import Simulation


def test_loop(DATA, tmp_path):
    """Test loop."""
    sim = Simulation(
        mesh=DATA / "cube.fly",
        materials_filepath=DATA / "cube.krn",
        parameters_filepath=DATA / "cube.p2",
    )

    # run loop
    sim.run_loop(outdir=tmp_path, name="cube")

    # check hysteresis loop
    content_1 = entities_from_file(DATA / "loop" / "cube.csv")
    content_2 = entities_from_file(tmp_path / "cube.csv")
    assert np.all(content_1.configuration_type == content_2.configuration_type)
    assert content_1.B_ext == content_2.B_ext
    assert content_1.J == content_2.J
    assert content_1.Jx == content_2.Jx
    assert content_1.Jy == content_2.Jy
    assert content_1.Jz == content_2.Jz
    assert content_1.energy_density == content_2.energy_density

    # check generated vtus
    vtu_list = [i.name for i in tmp_path.iterdir() if i.suffix == ".vtu"]
    for vtu_name in vtu_list:
        mesh_data = pv.read(DATA / "loop" / vtu_name)
        mesh_sim = pv.read(tmp_path / vtu_name)
        assert np.allclose(mesh_data.point_data["m"], mesh_sim.point_data["m"])
