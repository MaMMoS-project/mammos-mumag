"""Check hmag script."""

import meshio
import numpy as np
import pathlib
import polars as pl
from mammos_mmag.simulation import Simulation

DATA = pathlib.Path(__file__).resolve().parent / "data"


def test_hmag(tmp_path):
    """Test hmag."""
    # initialize + load parameters
    sim = Simulation(
        mesh_filepath=DATA / "cube.fly",
        materials_filepath=DATA / "cube.krn",
    )

    # run hmag
    sim.run_hmag(outdir=tmp_path)

    # check vtk files
    data_hmag = meshio.read(DATA / "hmag" / "cube_hmag.vtu")
    assert (
        np.linalg.norm(sim.hmag.point_data["U"] - data_hmag.point_data["U"]) < 1.0e-09
    )
    assert (
        np.linalg.norm(sim.hmag.point_data["h_nodes"] - data_hmag.point_data["h_nodes"])
        < 1.0e-09
    )
    assert (
        np.linalg.norm(sim.hmag.point_data["m"] - data_hmag.point_data["m"]) < 1.0e-09
    )
    assert (
        np.linalg.norm(sim.hmag.cell_data["h"][0] - data_hmag.cell_data["h"][0])
        < 1.0e-09
    )

    # check energies
    data_energy = pl.read_csv(DATA / "hmag" / "cube.csv", skip_rows=1)
    out_energy = pl.read_csv(tmp_path / "out.csv", skip_rows=1)
    diff = (data_energy["value"] - out_energy["value"]).to_numpy()
    assert np.linalg.norm(diff) < 1.0e-09
