"""Check materials script."""

import pathlib
import filecmp
import shutil
from mammos_mmag import Simulation

HERE = pathlib.Path(__file__).resolve().parent


def test_materials():
    """Test materials."""
    sim = Simulation()
    sim.mesh_path = HERE / "data" / "cube.fly"
    sim.materials.read_krn(HERE / "data" / "cube.krn")
    sim.run_materials(outdir=HERE / "materials")
    assert filecmp.cmp(
        HERE / "materials" / "out.mat.vtu", HERE / "data" / "materials" / "cube.mat.vtu"
    )
    shutil.rmtree(HERE / "materials")
