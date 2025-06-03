"""Generate test data."""

import pathlib

from mammos_mumag.simulation import Simulation

DATA = pathlib.Path(__file__).resolve().parent / "data"


def main():
    """Create test data."""
    sim = Simulation(
        mesh_filepath=DATA / "cube.fly",
        materials_filepath=DATA / "cube.krn",
        parameters_filepath=DATA / "cube.p2",
    )

    sim.run_materials(
        outdir=DATA / "materials",
        name="cube",
    )
    sim.run_hmag(
        outdir=DATA / "hmag",
        name="cube",
    )
    sim.run_exani(
        outdir=DATA / "exani",
        name="cube",
    )
    sim.run_external(
        outdir=DATA / "external",
        name="cube",
    )
    sim.run_mapping(
        outdir=DATA / "mapping",
        name="cube",
    )
    sim.run_loop(
        outdir=DATA / "loop",
        name="cube",
    )


if __name__ == "__main__":
    main()
