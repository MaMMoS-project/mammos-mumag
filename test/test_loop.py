"""Check loop script."""

import meshio
import numpy as np
from mammos_mumag.simulation import Simulation


def test_loop(DATA, tmp_path):
    """Test loop."""
    sim = Simulation(
        mesh_filepath=DATA / "cube.fly",
        materials_filepath=DATA / "cube.krn",
        parameters_filepath=DATA / "cube.p2",
    )

    # run loop
    sim.run_loop(outdir=tmp_path)

    # check hysteresis loop
    loop_data = np.loadtxt(DATA / "loop" / "cube.dat")
    loop_out = np.loadtxt(tmp_path / "out.dat")
    assert np.allclose(loop_data, loop_out)

    # check generated vtus
    for i, m_out_i in enumerate(sim.loop_vtu_list):
        m_data_i = meshio.read(DATA / "loop" / f"cube_{i:04}.vtu")
        assert np.allclose(m_out_i.point_data["m"], m_data_i.point_data["m"])
