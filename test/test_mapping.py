"""Check mapping script."""

import numpy as np
import pathlib
import polars as pl
from mammos_mmag.simulation import Simulation

DATA = pathlib.Path(__file__).resolve().parent / "data"


def test_mapping(tmp_path):
    """Test mapping."""
    # initialize + load parameters
    sim = Simulation()
    sim.mesh_path = DATA / "cube.fly"
    sim.materials.read(DATA / "cube.krn")
    sim.parameters.read(DATA / "cube.p2")

    # run mapping
    sim.run_mapping(outdir=tmp_path)

    # check anisotropy energy
    data = pl.read_csv(DATA / "mapping" / "cube_anisotropy.csv", skip_rows=1)
    out = pl.read_csv(tmp_path / "out_anisotropy.csv", skip_rows=1)
    diff = (data["value"] - out["value"]).to_numpy()
    assert np.linalg.norm(diff) < 1.0e-09

    # check exchange energy
    data = pl.read_csv(DATA / "mapping" / "cube_exchange.csv", skip_rows=1)
    out = pl.read_csv(tmp_path / "out_exchange.csv", skip_rows=1)
    diff = (data["value"] - out["value"]).to_numpy()
    assert np.linalg.norm(diff) < 1.0e-09

    # check hmag energy
    data = pl.read_csv(DATA / "mapping" / "cube_hmag.csv", skip_rows=1)
    out = pl.read_csv(tmp_path / "out_hmag.csv", skip_rows=1)
    diff = (data["value"] - out["value"]).to_numpy()
    assert np.linalg.norm(diff) < 1.0e-09

    # check zeeman energy
    data = pl.read_csv(DATA / "mapping" / "cube_zeeman.csv", skip_rows=1)
    out = pl.read_csv(tmp_path / "out_zeeman.csv", skip_rows=1)
    diff = (data["value"] - out["value"]).to_numpy()
    assert np.linalg.norm(diff) < 1.0e-09

    # check total energy
    data = pl.read_csv(DATA / "mapping" / "cube_energy.csv", skip_rows=1)
    out = pl.read_csv(tmp_path / "out_energy.csv", skip_rows=1)
    diff = (data["value"] - out["value"]).to_numpy()
    assert np.linalg.norm(diff) < 1.0e-09
