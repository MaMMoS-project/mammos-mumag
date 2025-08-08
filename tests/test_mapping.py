"""Check mapping script."""

import numpy as np
import pandas as pd

from mammos_mumag.simulation import Simulation


def test_mapping(DATA, tmp_path):
    """Test mapping."""
    # initialize + load parameters
    sim = Simulation(
        mesh=DATA / "cube.fly",
        materials_filepath=DATA / "cube.krn",
        parameters_filepath=DATA / "cube.p2",
    )

    # run mapping
    sim.run_mapping(outdir=tmp_path)

    # check anisotropy energy
    data = pd.read_csv(DATA / "mapping" / "cube_anisotropy.csv", skiprows=1)
    out = pd.read_csv(tmp_path / "out_anisotropy.csv", skiprows=1)
    assert np.allclose(data["value"], out["value"])

    # check exchange energy
    data = pd.read_csv(DATA / "mapping" / "cube_exchange.csv", skiprows=1)
    out = pd.read_csv(tmp_path / "out_exchange.csv", skiprows=1)
    assert np.allclose(data["value"], out["value"])

    # check hmag energy
    data = pd.read_csv(DATA / "mapping" / "cube_hmag.csv", skiprows=1)
    out = pd.read_csv(tmp_path / "out_hmag.csv", skiprows=1)
    assert np.allclose(data["value"], out["value"])

    # check zeeman energy
    data = pd.read_csv(DATA / "mapping" / "cube_zeeman.csv", skiprows=1)
    out = pd.read_csv(tmp_path / "out_zeeman.csv", skiprows=1)
    assert np.allclose(data["value"], out["value"])

    # check total energy
    data = pd.read_csv(DATA / "mapping" / "cube_energy.csv", skiprows=1)
    out = pd.read_csv(tmp_path / "out_energy.csv", skiprows=1)
    assert np.allclose(data["value"], out["value"])
