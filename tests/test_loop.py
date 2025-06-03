"""Check loop script."""

import numpy as np
import pyvista as pv

from mammos_mumag.simulation import Simulation


def test_loop(DATA, tmp_path):
    """Test loop."""
    sim = Simulation(
        mesh_filepath=DATA / "cube.fly",
        materials_filepath=DATA / "cube.krn",
        parameters_filepath=DATA / "cube.p2",
    )

    # run loop
    sim.run_loop(outdir=tmp_path, name="cube")

    # check hysteresis loop
    data_loop = np.loadtxt(DATA / "loop" / "cube.dat")
    sim_loop = np.loadtxt(tmp_path / "cube.dat")
    assert np.allclose(data_loop, sim_loop)

    # check generated vtus
    vtu_list = [i.name for i in tmp_path.iterdir() if i.suffix == ".vtu"]
    for vtu_name in vtu_list:
        mesh_data = pv.read(DATA / "loop" / vtu_name)
        mesh_sim = pv.read(tmp_path / vtu_name)
        assert np.allclose(mesh_data.point_data["m"], mesh_sim.point_data["m"])
