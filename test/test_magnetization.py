"""Check magnetization script."""

import numpy as np
import meshio
import pathlib
import shutil
from mammos_mmag import Simulation

HERE = pathlib.Path(__file__).resolve().parent


def test_magnetization():
    """Test magnetization."""
    sim = Simulation()
    sim.mesh_path = HERE / "data" / "cube.fly"
    sim.materials.read_krn(HERE / "data" / "cube.krn")
    sim.parameters.read_p2(HERE / "data" / "cube.p2")
    sim.run_magnetization(outdir=HERE / "magnetization")

    m_1 = meshio.read(HERE / "magnetization" / "out.0000.vtu")
    m_2 = meshio.read(HERE / "data" / "magnetization" / "cube.0000.vtu")
    assert np.linalg.norm(m_1.point_data["m"] - m_2.point_data["m"]) < 1.0e-09

    shutil.rmtree(HERE / "magnetization")
