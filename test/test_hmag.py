"""Check hmag script."""

import pathlib
import shutil
import filecmp
from mmag import Simulation

HERE = pathlib.Path(__file__).resolve().parent


def test_hmag():
    """Test hmag."""
    sim = Simulation()
    sim.mesh_path = HERE / "data" / "cube.fly"
    sim.materials.read_krn(HERE / "data" / "cube.krn")
    sim.run_hmag(outdir=HERE / "hmag")
    # sim.init_hmag(tol=1e-12)
    # m = [0.0, 0.0, 1.0]
    # fname = HERE / "tmp" / ("cube.hmag")
    # fname.mkdir(exist_ok=True)
    # E, E_gr, E_an = sim.eval_MagnetostaticEnergyDensity(m, fname)
    # assert E == 0.49684867825357726
    # assert E_gr == 0.49684867825356854
    # assert E_an == 0.5162666666666667
    assert filecmp.cmp(
        HERE / "hmag" / "out.hmag.vtu", HERE / "data" / "hmag" / "cube.hmag.vtu"
    )
    shutil.rmtree(HERE / "hmag")
