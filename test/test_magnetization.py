"""Check magnetization script."""

import pathlib
import filecmp
import shutil
from mammos_mmag import Simulation

HERE = pathlib.Path(__file__).resolve().parent


def test_magnetization():
    """Test magnetization."""
    sim = Simulation()
    sim.mesh_path = HERE / "data" / "cube.fly"
    sim.materials.read_krn(HERE / "data" / "cube.krn")
    sim.parameters.read_p2(HERE / "data" / "cube.p2")
    sim.run_magnetization(outdir=HERE / "magnetization")
    assert filecmp.cmp(
        HERE / "magnetization" / f"out.{0:04}.vtu",
        HERE / "data" / "magnetization" / f"cube.{0:04}.vtu",
    )
    shutil.rmtree(HERE / "magnetization")
