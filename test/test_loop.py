"""Check loop script."""

import meshio
import numpy as np
import pathlib
from mammos_mmag.simulation import Simulation

DATA = pathlib.Path(__file__).resolve().parent / "data"


def test_loop(tmp_path):
    """Test loop."""
    sim = Simulation()
    sim.mesh_path = DATA / "cube.fly"
    sim.materials.read(DATA / "cube.krn")
    sim.parameters.read(DATA / "cube.p2")
    sim.run_loop(outdir=tmp_path)

    loop_data = np.loadtxt(DATA / "loop" / "cube.dat")
    loop_out = np.loadtxt(tmp_path / "out.dat")
    assert np.linalg.norm(loop_data - loop_out) < 1.0e-07

    for i, m_out_i in enumerate(sim.loop_vtu_list):
        m_data_i = meshio.read(DATA / "loop" / f"cube_{i:04}.vtu")
        assert (
            np.linalg.norm(m_out_i.point_data["m"] - m_data_i.point_data["m"]) < 1.0e-06
        )
