import pathlib
from mmag.simulation import Simulation

here = pathlib.Path(__file__).absolute().parent
sim = Simulation()
sim.mesh_path = here / "cube.fly"
sim.materials.read_krn(here / "cube.krn")
sim.parameters.read_p2(here / "cube.p2")

sim.run_hystloop(name="cube")
