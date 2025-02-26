from mmag.simulation import Simulation

import esys.escript as e

sim = Simulation()
sim.read_mesh("cube.fly")
sim.read_materials_file("cube.krn")
tags = e.Function(sim.mesh).getListOfTags()
print(type(tags))
print(f"{tags=}")
print(f"{sim.A.__dir__()}")
print(f"{sim.A.tag()=}")
