from pathlib import Path
from mmag.simulation import Simulation

here = Path(__file__).absolute().parent
sim = Simulation()
sim.read_mesh(here / "cube.fly")
sim.read_materials_file(here / "cube.krn")
sim.read_parameters_file(here / "cube.p2")
# sim.run_hysteresis_loop("cube")
sim.hystloop_init("cube", outdir=None)
print(sim._hystloop_out / f".{0:04}")
