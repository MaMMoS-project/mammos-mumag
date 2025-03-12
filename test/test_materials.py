"""Check materials script."""

import meshio
import numpy as np
import pathlib
from mammos_mmag.simulation import Simulation

DATA = pathlib.Path(__file__).resolve().parent / "data"


def test_materials(tmp_path):
    """Test materials."""
    # initialize + load parameters
    sim = Simulation()
    sim.mesh_path = DATA / "cube.fly"
    sim.materials.read(DATA / "cube.krn")

    # run hmag
    sim.run_materials(outdir=tmp_path / "materials")

    # check materials vtu
    data = meshio.read(DATA / "materials" / "cube_mat.vtu")
    assert (
        np.linalg.norm(data.cell_data["A"][0] - sim.materials_fields.cell_data["A"][0])
        < 1.0e-09
    )
    assert (
        np.linalg.norm(
            data.cell_data["Js"][0] - sim.materials_fields.cell_data["Js"][0]
        )
        < 1.0e-09
    )
    assert (
        np.linalg.norm(data.cell_data["K"][0] - sim.materials_fields.cell_data["K"][0])
        < 1.0e-09
    )
    assert (
        np.linalg.norm(data.cell_data["u"][0] - sim.materials_fields.cell_data["u"][0])
        < 1.0e-09
    )
