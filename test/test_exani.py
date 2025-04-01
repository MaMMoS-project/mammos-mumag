"""Check exani script."""

import numpy as np
import polars as pl
from mammos_mumag.simulation import Simulation


def test_exani(DATA, tmp_path):
    """Test exani."""
    # initialize + load parameters
    sim = Simulation(
        mesh_filepath=DATA / "cube.fly", materials_filepath=DATA / "cube.krn"
    )

    # run exani
    sim.run_exani(outdir=tmp_path)

    # check vortex
    data_vortex = pl.read_csv(DATA / "exani" / "cube_vortex.csv", skip_rows=1)
    out_vortex = pl.read_csv(tmp_path / "out_vortex.csv", skip_rows=1)
    assert np.allclose(data_vortex["value"], out_vortex["value"])

    # check uniform
    data_unif = pl.read_csv(DATA / "exani" / "cube_uniform.csv", skip_rows=1)
    out_unif = pl.read_csv(tmp_path / "out_uniform.csv", skip_rows=1)
    assert np.allclose(data_unif["value"], out_unif["value"])
