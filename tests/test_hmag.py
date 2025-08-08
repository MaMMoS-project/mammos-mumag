"""Check hmag script."""

import numpy as np
import pandas as pd
import pyvista as pv

from mammos_mumag.simulation import Simulation


def test_hmag(DATA, tmp_path):
    """Test hmag."""
    # initialize + load parameters
    sim = Simulation(
        mesh=DATA / "cube.fly",
        materials_filepath=DATA / "cube.krn",
    )

    # run hmag
    sim.run_hmag(outdir=tmp_path, name="cube")

    # check vtk files
    sim_hmag = pv.read(tmp_path / "cube_hmag.vtu")
    data_hmag = pv.read(DATA / "hmag" / "cube_hmag.vtu")
    assert np.allclose(sim_hmag.point_data["U"], data_hmag.point_data["U"])
    assert np.allclose(sim_hmag.point_data["h_nodes"], data_hmag.point_data["h_nodes"])
    assert np.allclose(sim_hmag.point_data["m"], data_hmag.point_data["m"])
    assert np.allclose(sim_hmag.cell_data["h"][0], data_hmag.cell_data["h"][0])

    # check energies
    sim_energy = pd.read_csv(tmp_path / "cube.csv", skiprows=1)
    data_energy = pd.read_csv(DATA / "hmag" / "cube.csv", skiprows=1)
    assert np.allclose(sim_energy["value"], data_energy["value"])
