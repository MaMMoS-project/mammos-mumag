"""Generate test data."""

import pathlib
from mammos_mmag.simulation import Simulation

DATA = pathlib.Path(__file__).resolve().parent / "data"


def main():
    """Create test data."""
    sim = Simulation()
    sim.mesh_path = DATA / "cube.fly"
    sim.materials.read_krn(DATA / "cube.krn")
    sim.parameters.read(DATA / "cube.p2")

    sim.run_hmag(
        outdir=DATA / "hmag",
        name="cube",
    )
    sim.run_loop(
        outdir=DATA / "loop",
        name="cube",
    )
    sim.run_materials(
        outdir=DATA / "materials",
        name="cube",
    )


if __name__ == "__main__":
    main()
