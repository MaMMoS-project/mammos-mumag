"""Check loop script."""

import meshio
import numpy as np
import pathlib
import shutil
from mammos_mmag.simulation import Simulation

HERE = pathlib.Path(__file__).resolve().parent


def test_loop():
    """Test loop."""
    sim = Simulation()
    sim.mesh_path = HERE / "data" / "cube.fly"
    sim.materials.read(HERE / "data" / "cube.krn")
    sim.parameters.read(HERE / "data" / "cube.p2")
    sim.run_loop(outdir=HERE / "loop")

    loop_1 = np.loadtxt(HERE / "loop" / "out.dat")
    loop_2 = np.loadtxt(HERE / "data" / "loop" / "cube.dat")
    assert np.linalg.norm(loop_1 - loop_2) < 1.0e-07

    for i in range(1):
        m_1_i = meshio.read(HERE / "loop" / f"out_{i:04}.vtu")
        m_2_i = meshio.read(HERE / "data" / "loop" / f"cube_{i:04}.vtu")
        assert np.linalg.norm(m_1_i.point_data["m"] - m_2_i.point_data["m"]) < 1.0e-06

    shutil.rmtree(HERE / "loop")
