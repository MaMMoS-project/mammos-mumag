"""Check exani script."""

import numpy as np
import pandas as pd

from mammos_mumag.simulation import Simulation


def test_exani(DATA, tmp_path):
    """Test exani."""
    # initialize + load parameters
    sim = Simulation(mesh=DATA / "cube.fly", materials_filepath=DATA / "cube.krn")

    # run exani
    sim.run_exani(outdir=tmp_path)

    # check vortex
    data_vortex = pd.read_csv(DATA / "exani" / "cube_vortex.csv", skiprows=1)
    out_vortex = pd.read_csv(tmp_path / "out_vortex.csv", skiprows=1)
    assert np.allclose(data_vortex["value"], out_vortex["value"])

    # check uniform
    data_unif = pd.read_csv(DATA / "exani" / "cube_uniform.csv", skiprows=1)
    out_unif = pd.read_csv(tmp_path / "out_uniform.csv", skiprows=1)
    assert np.allclose(data_unif["value"], out_unif["value"])
