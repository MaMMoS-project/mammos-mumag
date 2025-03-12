"""Check exani script."""

import numpy as np
import pathlib
import polars as pl
from mammos_mmag.simulation import Simulation

DATA = pathlib.Path(__file__).resolve().parent / "data"


def test_exani(tmp_path):
    """Test exani."""
    # initialize + load parameters
    sim = Simulation()
    sim.mesh_path = DATA / "cube.fly"
    sim.materials.read(DATA / "cube.krn")

    # run exani
    sim.run_exani(outdir=tmp_path)

    # check vortex
    data_vortex = pl.read_csv(DATA / "exani" / "cube_vortex.csv", skip_rows=1)
    out_vortex = pl.read_csv(tmp_path / "out_vortex.csv", skip_rows=1)
    diff_vortex = (data_vortex["value"] - out_vortex["value"]).to_numpy()
    assert np.linalg.norm(diff_vortex) < 1.0e-09

    # check uniform
    data_unif = pl.read_csv(DATA / "exani" / "cube_uniform.csv", skip_rows=1)
    out_unif = pl.read_csv(tmp_path / "out_uniform.csv", skip_rows=1)
    diff_unif = (data_unif["value"] - out_unif["value"]).to_numpy()
    assert np.linalg.norm(diff_unif) < 1.0e-09
