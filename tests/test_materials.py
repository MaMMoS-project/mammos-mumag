"""Check materials script."""

import numpy as np
import pyvista as pv

from mammos_mumag.simulation import Simulation


def test_materials(DATA, tmp_path):
    """Test materials."""
    # initialize + load parameters
    sim = Simulation(
        mesh=DATA / "cube.fly",
        materials_filepath=DATA / "cube.krn",
    )

    # run hmag
    sim.run_materials(outdir=tmp_path, name="cube")

    # check materials vtu
    sim_materials = pv.read(tmp_path / "cube_mat.vtu")
    data = pv.read(DATA / "materials" / "cube_mat.vtu")
    assert np.allclose(data.cell_data["A"][0], sim_materials.cell_data["A"][0])
    assert np.allclose(data.cell_data["Js"][0], sim_materials.cell_data["Js"][0])
    assert np.allclose(data.cell_data["K"][0], sim_materials.cell_data["K"][0])
    assert np.allclose(data.cell_data["u"][0], sim_materials.cell_data["u"][0])
