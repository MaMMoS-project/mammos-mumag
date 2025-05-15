"""Check mapping script."""

import numpy as np
import polars as pl
from mammos_mumag.simulation import Simulation


def test_mapping(DATA, tmp_path):
    """Test mapping."""
    # initialize + load parameters
    sim = Simulation(
        mesh_filepath=DATA / "cube.fly",
        materials_filepath=DATA / "cube.krn",
        parameters_filepath=DATA / "cube.p2",
    )

    # run mapping
    sim.run_mapping(outdir=tmp_path)

    # check anisotropy energy
    data = pl.read_csv(DATA / "mapping" / "cube_anisotropy.csv", skip_rows=1)
    out = pl.read_csv(tmp_path / "out_anisotropy.csv", skip_rows=1)
    assert np.allclose(data["value"], out["value"])

    # check exchange energy
    data = pl.read_csv(DATA / "mapping" / "cube_exchange.csv", skip_rows=1)
    out = pl.read_csv(tmp_path / "out_exchange.csv", skip_rows=1)
    assert np.allclose(data["value"], out["value"])

    # check hmag energy
    data = pl.read_csv(DATA / "mapping" / "cube_hmag.csv", skip_rows=1)
    out = pl.read_csv(tmp_path / "out_hmag.csv", skip_rows=1)
    assert np.allclose(data["value"], out["value"])

    # check zeeman energy
    data = pl.read_csv(DATA / "mapping" / "cube_zeeman.csv", skip_rows=1)
    out = pl.read_csv(tmp_path / "out_zeeman.csv", skip_rows=1)
    assert np.allclose(data["value"], out["value"])

    # check total energy
    data = pl.read_csv(DATA / "mapping" / "cube_energy.csv", skip_rows=1)
    out = pl.read_csv(tmp_path / "out_energy.csv", skip_rows=1)
    assert np.allclose(data["value"], out["value"])
