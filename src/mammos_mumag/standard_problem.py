"""Functions for the standard problem."""

import math
import pandas as pd

from mammos_mumag.materials import Materials
from mammos_mumag.parameters import Parameters
from mammos_mumag.simulation import Simulation
import mammos_units as u


def hystloop(
    mesh_filepath,
    Ms,
    A,
    K1,
    hstart=2 * u.T,
    hfinal=-2 * u.T,
    hstep=None,
    hnsteps=20,
    outdir="hystloop",
):
    """Run hysteresis loop."""
    if hstep is None:
        hstep = (hfinal - hstart) / hnsteps
    if K1.unit != u.J / u.m**3:
        K1 = K1.to(u.J / u.m**3)
    if Ms.unit != u.T:
        Ms = Ms.to(u.T, equivalencies=u.magnetic_flux_field())
    if A.unit != u.J / u.m:
        A = A.to(u.J / u.m)

    sim = Simulation(
        mesh_filepath=mesh_filepath,
        materials=Materials(
            domains=[
                {
                    "theta": 0,
                    "phi": 0.0,
                    "K1": 4e-7 * math.pi * K1.value,
                    "K2": 0.0,
                    "Js": Ms.to(u.T, equivalencies=u.magnetic_flux_field()).value,
                    "A": 4e-7 * math.pi * A.value,
                },
                {
                    "theta": 0.0,
                    "phi": 0.0,
                    "K1": 0.0,
                    "K2": 0.0,
                    "Js": 0.0,
                    "A": 0.0,
                },
                {
                    "theta": 0.0,
                    "phi": 0.0,
                    "K1": 0.0,
                    "K2": 0.0,
                    "Js": 0.0,
                    "A": 0.0,
                },
            ],
        ),
        parameters=Parameters(
            size=1.0e-9,
            scale=0,
            m_vect=[0, 0, 1],
            hstart=hstart.value,
            hfinal=hfinal.value,
            hstep=hstep.value,
            h_vect=[0.01745, 0, 0.99984],
            mstep=0.4,
            mfinal=-1.2,
            tol_fun=1e-10,
            tol_hmag_factor=1,
            precond_iter=10,
        ),
    )
    sim.run_loop(outdir=outdir, name="stdpb")
    hl = pd.read_csv(
        f"{outdir}/stdpb.dat", delimiter=" ", names=["idx", "mu0_Hext", "pol", "E"]
    )
    return hl, sim.loop_vtu_list
