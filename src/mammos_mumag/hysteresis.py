"""Functions for evaluating and processin the hysteresis loop."""

from __future__ import annotations

import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
import pyvista as pv


import mammos_entity as me
import mammos_units as u
from mammos_mumag.materials import Materials
from mammos_mumag.parameters import Parameters
from mammos_mumag.simulation import Simulation


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
) -> Result:
    r"""Run hysteresis loop.

    Args:
        Ms: Spontaneous magnetisation in :math:`\mathrm{A}/\mathrm{m}`.
        A: Exchange stiffness constant in :math:`\mathrm{J}/\mathrm{m}`.
        K1: First magnetocrystalline anisotropy constant in
            :math:`\mathrm{J}/\mathrm{m}^3`.
        mesh_filepath: TODO
        hstart: TODO
        hfinal: TODO
        hstep: TODO
        hnsteps: TODO
        outdir: TODO

    Returns:
       TODO

    """
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
    df = pd.read_csv(
        f"{outdir}/hystloop.dat",
        delimiter=" ",
        names=["configuration_type", "mu0_Hext", "polarisation", "energy_density"],
    )
    return Result(
        H=me.Entity(
            "ExternalMagneticField",
            value=(df["mu0_Hext"].to_numpy() * u.T).to(u.A / u.m, equivalencies=u.magnetic_flux_field()),
            unit=u.A / u.m,
        ),
        M=me.Ms(
            (df["polarisation"].to_numpy() * u.T).to(u.A / u.m, equivalencies=u.magnetic_flux_field()),
            unit=u.A / u.m,
        ),
        energy_density=me.Entity("EnergyDensity", value=df["energy_density"], unit=u.J / u.m**3),
        configurations=[
            fname
            for fname in pathlib.Path(outdir).resolve().iterdir()
            if fname.suffix == ".vtu"
        ],
        configuration_type=df["configuration_type"].to_numpy(),
    )


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, frozen=True))
class Result:
    """Hysteresis loop Result."""

    H: me.Entity
    M: me.Entity
    energy_density: me.Entity | None = None
    configuration_type: np.ndarray | None = None
    configurations: list[pathlib.Path] | None = None

    @property
    def dataframe(self) -> pandas.DataFrame:
        """Dataframe containing the result data of the hysteresis loop."""
        return pd.DataFrame(
            {
                "configuration_type": self.configuration_type,
                "H": self.H,
                "M": self.M,
                "energy_density": self.energy_density,
            }
        )

    def plot(self, duplicate: bool = True, configuration_marks: bool = False) -> None:
        """Plot hysteresis loop.

        Args:
            duplicate: Also plot loop with -M and -H to simulate full hysteresis.
            configuration_marks: Show markers where a configuration has been saved.

        """
        fig, ax = plt.subplots()
        plt.plot(self.dataframe["mu0_Hext"], self.dataframe["polarisation"])
        j = 0
        if configuration_marks:
            for _, r in self.dataframe.iterrows():
                idx = int(r["idx"])
                if idx != j:
                    plt.plot(r["mu0_Hext"], r["polarisation"], "rx")
                    j = idx
                    ax.annotate(
                        j,
                        xy=(r["mu0_Hext"], r["polarisation"]),
                        xytext=(-2, -10),
                        textcoords="offset points",
                    )
        if duplicate:
            plt.plot(-self.dataframe["mu0_Hext"], -self.dataframe["polarisation"])

    def plot_configuration(self, idx: int, jupyter_backend: str = "trame") -> None:
        """Plot configuration with index `idx`.

        Args:
            idx: Index of the configuration.
            jupyter_backend: Plotting backend.

        """
        config = pv.read(self.configurations[idx])
        config["m_norm"] = np.linalg.norm(config["m"], axis=1)
        glyphs = config.glyph(
            orient="m",
            scale="m_norm",
        )
        pl = pv.Plotter()
        pl.add_mesh(
            glyphs,
            scalars=glyphs["GlyphVector"][:, 2],
            lighting=False,
            cmap="coolwarm",
            scalar_bar_args={"title": "m_z"},
        )
        pl.show_axes()
        pl.show(jupyter_backend=jupyter_backend)
