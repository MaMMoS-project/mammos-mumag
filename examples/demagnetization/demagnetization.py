from pathlib import Path
from mmag.loop import Loop

here = Path(__file__).absolute().parent
loop = Loop("cube", outdir=here / "out")
loop.read_mesh(here / "cube.fly")
loop.read_params(here / "cube.p2")
loop.read_materials(here / "cube.krn")
loop.run()
