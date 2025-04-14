"""Check external script."""

import numpy as np
import polars as pl
from mammos_mumag.simulation import Simulation


def test_external(DATA, tmp_path):
    """Test external."""
    # initialize + load parameters
    sim = Simulation(
        mesh_filepath=DATA / "cube.fly",
        materials_filepath=DATA / "cube.krn",
        parameters_filepath=DATA / "cube.p2",
    )

    # run external
    sim.run_external(outdir=tmp_path)

    # check Zeeman energy
    data = pl.read_csv(DATA / "external" / "cube.csv", skip_rows=1)
    out = pl.read_csv(tmp_path / "out.csv", skip_rows=1)
    assert np.allclose(data["value"], out["value"])
