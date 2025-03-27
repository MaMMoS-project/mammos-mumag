"""Check materials script."""

import meshio
import numpy as np
from mammos_mmag.simulation import Simulation


def test_materials(DATA, tmp_path):
    """Test materials."""
    # initialize + load parameters
    sim = Simulation(
        mesh_filepath=DATA / "cube.fly",
        materials_filepath=DATA / "cube.krn",
    )

    # run hmag
    sim.run_materials(outdir=tmp_path / "materials")

    # check materials vtu
    data = meshio.read(DATA / "materials" / "cube_mat.vtu")
    assert np.allclose(data.cell_data["A"][0], sim.materials_fields.cell_data["A"][0])
    assert np.allclose(data.cell_data["Js"][0], sim.materials_fields.cell_data["Js"][0])
    assert np.allclose(data.cell_data["K"][0], sim.materials_fields.cell_data["K"][0])
    assert np.allclose(data.cell_data["u"][0], sim.materials_fields.cell_data["u"][0])
