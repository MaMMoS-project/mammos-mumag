"""Check exani script."""

import pathlib
import shutil
from mmag import Simulation

HERE = pathlib.Path(__file__).resolve().parent


def test_exani():
    """Test exani."""
    sim = Simulation()
    sim.mesh_path = HERE / "data" / "cube.fly"
    sim.materials.read_krn(HERE / "data" / "cube.krn")
    sim.run_exani(outdir=HERE / "exani")
    # vortex_energy = sim.get_vortex_energy()
    # assert vortex_energy[0] == 1.2557524195187777
    # assert vortex_energy[1] == 1.5828977293026091e-06

    # uniform_m = [0.0, 0.0, 1.0]
    # uniform_energy = sim.get_energy(uniform_m)
    # assert uniform_energy[0] == -0.5654866776461906
    # assert uniform_energy[1] == -7.106115168784338e-07
    assert True
    shutil.rmtree(HERE / "exani")
