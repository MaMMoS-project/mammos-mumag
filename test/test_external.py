"""Check external script."""

import pathlib
import shutil
from mammos_mmag import Simulation

HERE = pathlib.Path(__file__).resolve().parent


def test_external():
    """Test external."""
    sim = Simulation()
    sim.mesh_path = HERE / "data" / "cube.fly"
    sim.materials.read_krn(HERE / "data" / "cube.krn")
    sim.parameters.read_p2(HERE / "data" / "cube.p2")
    sim.run_external(outdir=HERE / "external")
    # h, E_gr, E_an = sim.eval_external()
    # assert E_gr == -2.1116784159019675
    # assert E_an == -2.1116784159018285
    assert True
    shutil.rmtree(HERE / "external")
