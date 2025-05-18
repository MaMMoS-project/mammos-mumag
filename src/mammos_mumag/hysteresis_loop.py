"""Functions for evaluating and processin the hysteresis loop."""

import matplotlib.pyplot as plt
import pandas as pd
import pathlib
from textwrap import dedent
from typing import Optional

from mammos_mumag.materials import Materials
from mammos_mumag.mesh import Mesh
from mammos_mumag.parameters import Parameters
from mammos_mumag.simulation import Simulation
import mammos_entity as me
import mammos_units as u


def run(
    Ms,
    A,
    K1,
    hstart=2 * u.T,
    hfinal=-2 * u.T,
    mesh: Optional[Mesh] = None,
    mesh_filepath: Optional[pathlib.Path] = None,
    hstep=None,
    hnsteps=20,
    outdir="hystloop",
):
    """Run hysteresis loop."""
    if mesh is None and mesh_filepath is None:
        raise ValueError(
            dedent(
                """
                mesh and mesh_filepath are both None.
                Either define mesh as a `mammos_mumag.mesh.Mesh` instance,
                or define the location of the mesh file.
                """
            )
        )
    elif mesh is not None:
        mesh_filepath = mesh.value
    if hstep is None:
        hstep = (hfinal - hstart) / hnsteps
    if isinstance(K1, u.Quantity):
        if K1.unit != u.J / u.m**3:
            K1 = K1.to(u.J / u.m**3)
    else:
        K1 = me.Ku(K1, unit=u.J / u.m**3)

    if isinstance(Ms, u.Quantity):
        if Ms.unit != u.A / u.m:
            Ms = Ms.to(u.A / u.m)
    else:
        Ms = me.Ms(Ms, unit=u.A / u.m)

    if isinstance(A, u.Quantity):
        if A.unit != u.J / u.m:
            A = A.to(u.J / u.m)
    else:
        A = me.A(A, unit=u.J / u.m)

    sim = Simulation(
        mesh_filepath=mesh_filepath,
        materials=Materials(
            domains=[
                {
                    "theta": 0,
                    "phi": 0.0,
                    "K1": K1,
                    "K2": me.Ku(0),
                    "Js": Ms,
                    "A": A,
                },
                {
                    "theta": 0.0,
                    "phi": 0.0,
                    "K1": me.Ku(0),
                    "K2": me.Ku(0),
                    "Js": me.Ms(0),
                    "A": me.A(0),
                },
                {
                    "theta": 0.0,
                    "phi": 0.0,
                    "K1": me.Ku(0),
                    "K2": me.Ku(0),
                    "Js": me.Ms(0),
                    "A": me.A(0),
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
    sim.run_loop(outdir=outdir, name="hystloop")
    hl = pd.read_csv(
        f"{outdir}/hystloop.dat", delimiter=" ", names=["idx", "mu0_Hext", "pol", "E"]
    )
    return hl, sim.loop_vtu_list


def plot(hl, duplicate=True):
    """Plot hysteresis loop."""
    plt.plot(hl["mu0_Hext"], hl["pol"])
    j = 0
    for i, r in hl.iterrows():
        if r["idx"] != j:
            plt.plot(r["mu0_Hext"], r["pol"], "rx")
            j = r["idx"]
    if duplicate:
        plt.plot(-hl["mu0_Hext"], -hl["pol"])
