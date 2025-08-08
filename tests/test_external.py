"""Check external script."""

import numpy as np
import pandas as pd

from mammos_mumag.simulation import Simulation


def test_external(DATA, tmp_path):
    """Test external."""
    # initialize + load parameters
    sim = Simulation(
        mesh=DATA / "cube.fly",
        materials_filepath=DATA / "cube.krn",
        parameters_filepath=DATA / "cube.p2",
    )

    # run external
    sim.run_external(outdir=tmp_path)

    # check Zeeman energy
    data = pd.read_csv(DATA / "external" / "cube.csv", skiprows=1)
    out = pd.read_csv(tmp_path / "out.csv", skiprows=1)
    assert np.allclose(data["value"], out["value"])
