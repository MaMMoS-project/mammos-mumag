"""Check materials script."""

import numpy as np
import pytest
import pyvista as pv

from mammos_mumag.materials import Materials
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


def test_wrong_numgrains(tmp_path):
    """Test equivalency of number of grains of mesh and materials.

    One singlegrain mesh and one multigrain mesh are tested.
    """
    mat = Materials(
        domains=[
            {
                "theta": 0,
                "phi": 0,
                "K1": 0,
                "K2": 0,
                "Ms": 0,
                "A": 0,
            }
        ]
    )

    sim = Simulation(
        mesh="cube40_singlegrain_msize1",
        materials=mat,
    )
    with pytest.raises(ValueError):
        sim.run_materials(outdir=tmp_path)

    sim = Simulation(
        mesh="cube40_colu_grains8_gsize20",
        materials=mat,
    )
    with pytest.raises(ValueError):
        sim.run_materials(outdir=tmp_path)
