"""Functions for evaluating and processin the hysteresis loop."""

import pandas as pd
import pathlib

from mammos_mumag.materials import Materials
from mammos_mumag.parameters import Parameters
from mammos_mumag.results import LoopResults
from mammos_mumag.simulation import Simulation
import mammos_entity as me
import mammos_units as u


def run(
    Ms: float | u.Quantity | me.Entity,
    A: float | u.Quantity | me.Entity,
    K1: float | u.Quantity | me.Entity,
    mesh_filepath: pathlib.Path,
    hstart: float | u.Quantity = (2 * u.T).to(
        u.A / u.m, equivalencies=u.magnetic_flux_field()
    ),
    hfinal: float | u.Quantity = (-2 * u.T).to(
        u.A / u.m, equivalencies=u.magnetic_flux_field()
    ),
    hstep: float | u.Quantity | None = None,
    hnsteps: int = 20,
    outdir: str | pathlib.Path = "hystloop",
) -> LoopResults:
    """Run hysteresis loop."""
    if hstep is None:
        hstep = (hfinal - hstart) / hnsteps

    if not isinstance(A, u.Quantity) or A.unit != u.J / u.m:
        A = me.A(A, unit=u.J / u.m)
    if not isinstance(K1, u.Quantity) or K1.unit != u.J / u.m**3:
        K1 = me.Ku(K1, unit=u.J / u.m**3)
    if not isinstance(Ms, u.Quantity) or Ms.unit != u.A / u.m:
        Ms = me.Ms(Ms, unit=u.A / u.m)

    sim = Simulation(
        mesh_filepath=mesh_filepath,
        materials=Materials(
            domains=[
                {
                    "theta": 0,
                    "phi": 0.0,
                    "K1": K1,
                    "K2": me.Ku(0),
                    "Ms": Ms,
                    "A": A,
                },
                {
                    "theta": 0.0,
                    "phi": 0.0,
                    "K1": me.Ku(0),
                    "K2": me.Ku(0),
                    "Ms": me.Ms(0),
                    "A": me.A(0),
                },
                {
                    "theta": 0.0,
                    "phi": 0.0,
                    "K1": me.Ku(0),
                    "K2": me.Ku(0),
                    "Ms": me.Ms(0),
                    "A": me.A(0),
                },
            ],
        ),
        parameters=Parameters(
            size=1.0e-9,
            scale=0,
            m_vect=[0, 0, 1],
            hstart=hstart.to(u.T, equivalencies=u.magnetic_flux_field()).value,
            hfinal=hfinal.to(u.T, equivalencies=u.magnetic_flux_field()).value,
            hstep=hstep.to(u.T, equivalencies=u.magnetic_flux_field()).value,
            h_vect=[0.01745, 0, 0.99984],
            mstep=0.4,
            mfinal=-1.2,
            tol_fun=1e-10,
            tol_hmag_factor=1,
            precond_iter=10,
        ),
    )
    sim.run_loop(outdir=outdir, name="hystloop")
    res = LoopResults(
        pd.read_csv(
            f"{outdir}/hystloop.dat",
            delimiter=" ",
            names=["idx", "mu0_Hext", "polarisation", "E"],
        ),
        [
            fname
            for fname in pathlib.Path(outdir).resolve().iterdir()
            if fname.suffix == ".vtu"
        ],
    )
    return res
