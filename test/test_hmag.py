"""Check hmag script."""

import meshio
import numpy as np
import pathlib
import shutil
from mammos_mmag.simulation import Simulation

HERE = pathlib.Path(__file__).resolve().parent


def test_hmag():
    """Test hmag."""
    sim = Simulation()
    sim.mesh_path = HERE / "data" / "cube.fly"
    sim.materials.read(HERE / "data" / "cube.krn")
    sim.run_hmag(outdir=HERE / "hmag")

    out_mesh = meshio.read(HERE / "hmag" / "out_hmag.vtu")
    cube_mesh = meshio.read(HERE / "data" / "hmag" / "cube_hmag.vtu")
    assert (
        np.linalg.norm(out_mesh.point_data["U"] - cube_mesh.point_data["U"]) < 1.0e-09
    )
    assert (
        np.linalg.norm(out_mesh.point_data["h_nodes"] - cube_mesh.point_data["h_nodes"])
        < 1.0e-09
    )
    assert (
        np.linalg.norm(out_mesh.point_data["m"] - cube_mesh.point_data["m"]) < 1.0e-09
    )
    assert (
        np.linalg.norm(out_mesh.cell_data["h"][0] - cube_mesh.cell_data["h"][0])
        < 1.0e-09
    )

    shutil.rmtree(HERE / "hmag")
