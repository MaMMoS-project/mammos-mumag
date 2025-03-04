"""Check materials script."""

import meshio
import numpy as np
import pathlib
import shutil
from mammos_mmag import Simulation

HERE = pathlib.Path(__file__).resolve().parent


def test_materials():
    """Test materials."""
    sim = Simulation()
    sim.mesh_path = HERE / "data" / "cube.fly"
    sim.materials.read_krn(HERE / "data" / "cube.krn")
    sim.run_materials(outdir=HERE / "materials")

    m_1 = meshio.read(HERE / "materials" / "out.mat.vtu")
    m_2 = meshio.read(HERE / "data" / "materials" / "cube.mat.vtu")
    assert np.linalg.norm(m_1.cell_data["A"][0] - m_2.cell_data["A"][0]) < 1.0e-09
    assert np.linalg.norm(m_1.cell_data["Js"][0] - m_2.cell_data["Js"][0]) < 1.0e-09
    assert np.linalg.norm(m_1.cell_data["K"][0] - m_2.cell_data["K"][0]) < 1.0e-09
    assert np.linalg.norm(m_1.cell_data["u"][0] - m_2.cell_data["u"][0]) < 1.0e-09

    shutil.rmtree(HERE / "materials")
