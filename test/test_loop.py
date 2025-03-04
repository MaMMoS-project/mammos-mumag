"""Check loop script."""

import pathlib
import filecmp
import shutil
from mammos_mmag import Simulation

HERE = pathlib.Path(__file__).resolve().parent


def test_loop():
    """Test loop."""
    sim = Simulation()
    sim.mesh_path = HERE / "data" / "cube.fly"
    sim.materials.read_krn(HERE / "data" / "cube.krn")
    sim.parameters.read_p2(HERE / "data" / "cube.p2")
    sim.run_loop(outdir=HERE / "loop")
    assert filecmp.cmp(HERE / "loop" / "out.dat", HERE / "data" / "loop" / "cube.dat")
    shutil.rmtree(HERE / "loop")
