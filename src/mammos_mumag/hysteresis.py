"""Functions for evaluating and processin the hysteresis loop."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import mammos_entity as me
import mammos_units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import pyvista as pv
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from mammos_mumag.materials import MaterialDomain, Materials
from mammos_mumag.parameters import Parameters
from mammos_mumag.simulation import Simulation

if TYPE_CHECKING:
    import matplotlib
    import pyvista


def run(
    Ms: float | u.Quantity | me.Entity | list[float | u.Quantity | me.Entity],
    A: float | u.Quantity | me.Entity | list[float | u.Quantity | me.Entity],
    K1: float | u.Quantity | me.Entity | list[float | u.Quantity | me.Entity],
    theta: float | u.Quantity | me.Entity | list[float | u.Quantity | me.Entity],
    phi: float | u.Quantity | me.Entity | list[float | u.Quantity | me.Entity],
    mesh_filepath: pathlib.Path,
    hstart: float | u.Quantity,
    hfinal: float | u.Quantity,
    hstep: float | u.Quantity | None = None,
    hnsteps: int = 20,
    mfinal: float = -2.0,
    outdir: str | pathlib.Path = "hystloop",
) -> Result:
    r"""Run hysteresis loop.

    Args:
        Ms: Spontaneous magnetisation in :math:`\mathrm{A}/\mathrm{m}`.
        A: Exchange stiffness constant in :math:`\mathrm{J}/\mathrm{m}`.
        K1: First magnetocrystalline anisotropy constant in
            :math:`\mathrm{J}/\mathrm{m}^3`.
        mesh_filepath: TODO
        hstart: Initial strength of the external field.
        hfinal: Final strength of the external field.
        hstep: Step size.
        hnsteps: Number of steps in the field sweep.
        outdir: Directory where simulation results are written to.

    Returns:
       Result object.

    """
    Ms = _listify_material_parameter("Ms", Ms, "SpontaneousMagnetization")
    A = _listify_material_parameter("A", A, "ExchangeStiffnessConstant")
    K1 = _listify_material_parameter("K1", K1, "UniaxialAnisotropyConstant")
    theta = _listify_material_parameter("theta", theta, "Angle")
    phi = _listify_material_parameter("theta", theta, "Angle")

    if len({len(e) for e in [Ms, A, K1, theta, phi]}) != 1:
        print(theta)
        raise ValueError("All material parameters must have the same length.")

    if hstep is None:
        hstep = (hfinal - hstart) / hnsteps

    materials = Materials(
        domains=[
            MaterialDomain(
                theta=theta_i,
                phi=phi_i,
                K1=me.Ku(K1_i),
                Ms=me.Ms(Ms_i),
                A=me.A(A_i),
            )
            for Ms_i, A_i, K1_i, theta_i, phi_i in zip(Ms, A, K1, theta, phi, strict=True)
        ] + [MaterialDomain()] * 2 # empty loops for air and shell regions
    )

    sim = Simulation(
        mesh_filepath=mesh_filepath,
        materials=materials,
        parameters=Parameters(
            size=1.0e-9,
            scale=0,
            m_vect=[0, 0, 1],
            hstart=hstart.to(u.T, equivalencies=u.magnetic_flux_field()).value,
            hfinal=hfinal.to(u.T, equivalencies=u.magnetic_flux_field()).value,
            hstep=hstep.to(u.T, equivalencies=u.magnetic_flux_field()).value,
            h_vect=[0.01745, 0, 0.99984],
            mstep=0.4,
            mfinal=mfinal,
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
            value=(df["mu0_Hext"].to_numpy() * u.T).to(
                u.A / u.m, equivalencies=u.magnetic_flux_field()
            ),
            unit=u.A / u.m,
        ),
        M=me.Ms(
            (df["polarisation"].to_numpy() * u.T).to(
                u.A / u.m, equivalencies=u.magnetic_flux_field()
            ),
            unit=u.A / u.m,
        ),
        energy_density=me.Entity(
            "EnergyDensity", value=df["energy_density"], unit=u.J / u.m**3
        ),
        configurations={
            i + 1: fname
            for i, fname in enumerate(
                sorted(pathlib.Path(outdir).resolve().glob("*.vtu"))
            )
        },
        configuration_type=df["configuration_type"].to_numpy(),
    )


def _listify_material_parameter(name, parameter, label):
    if isinstance(parameter, list):
        if isinstance(parameter, me.Entity) and not all([p.ontology_label == label for p in parameter]):
                raise ValueError(f"All items of {name} must be {label} entities")
    else:
        if isinstance(parameter, me.Entity) and parameter.q.size > 1:
            parameter = [me.Entity(label, p) for p in parameter.q]
        if isinstance(parameter, np.ndarray) and parameter.size > 1:
            parameter = [me.Entity(label, p) for p in parameter]
    return parameter


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, frozen=True))
class Result:
    """Hysteresis loop Result."""

    H: me.Entity
    """Array of external field strengths."""
    M: me.Entity
    """Array of spontaneous magnetization values for the field strengths."""
    energy_density: me.Entity | None = None
    """Array of energy densities for the field strengths."""
    configuration_type: np.ndarray | None = None
    """Array of indices of representative configurations for the field strengths."""
    configurations: dict[int, pathlib.Path] | None = None
    """Mapping of configuration indices to file paths."""

    @property
    def dataframe(self) -> pandas.DataFrame:
        """Dataframe containing the result data of the hysteresis loop."""
        return pd.DataFrame(
            {
                "configuration_type": self.configuration_type,
                "H": self.H.q,
                "M": self.M.q,
                "energy_density": self.energy_density.q,
            }
        )

    def plot(
        self,
        duplicate: bool = True,
        duplicate_change_color: bool = True,
        configuration_marks: bool = False,
        ax: matplotlib.axes.Axes | None = None,
        label: str | None = None,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plot hysteresis loop.

        Args:
            duplicate: Also plot loop with -M and -H to simulate full hysteresis.
            configuration_marks: Show markers where a configuration has been saved.
            ax: Matplotlib axes object to which the plot is added. A new one is create
                if not passed.
            kwargs: Additional keyword arguments passed to `ax.plot` when plotting the
                hysteresis lines.

        Returns:
            The `matplotlib.axes.Axes` object which was used to plot the hysteresis loop

        """
        if ax:
            ax = ax
        else:
            _, ax = plt.subplots()
        if label:
            (line,) = ax.plot(self.dataframe.H, self.dataframe.M, label=label, **kwargs)
        else:
            (line,) = ax.plot(self.dataframe.H, self.dataframe.M, **kwargs)
        j = 0
        if configuration_marks:
            for _, row in self.dataframe.iterrows():
                idx = int(row.configuration_type)
                if idx != j:
                    plt.plot(row.H, row.M, "rx")
                    j = idx
                    ax.annotate(
                        j,
                        xy=(row.H, row.M),
                        xytext=(-2, -10),
                        textcoords="offset points",
                    )
        ax.set_title("Hysteresis Loop")
        ax.set_xlabel(self.H.axis_label)
        ax.set_ylabel(self.M.axis_label)
        if label:
            ax.legend()
        if duplicate:
            if not duplicate_change_color:
                kwargs.setdefault("color", line.get_color())
            ax.plot(-self.dataframe.H, -self.dataframe.M, **kwargs)

        return ax

    def plot_configuration(
        self,
        idx: int,
        jupyter_backend: str = "trame",
        plotter: pyvista.Plotter | None = None,
    ) -> None:
        """Plot configuration with index `idx`.

        This method does only directly show the plot if no plotter is passed in.
        Otherwise, the caller must call ``plotter.show()`` separately. This behavior
        is based on the assumption that the user will want to further modify the plot
        before displaying/saving it when passing a plotter.

        Args:
            idx: Index of the configuration.
            jupyter_backend: Plotting backend.
            plotter: Pyvista plotter to which glyphs will be added. A new plotter is
                created if no plotter is passed.

        """
        config = pv.read(self.configurations[idx])
        config["m_norm"] = np.linalg.norm(config["m"], axis=1)
        glyphs = config.glyph(
            orient="m",
            scale="m_norm",
        )
        pl = plotter or pv.Plotter()
        pl.add_mesh(
            glyphs,
            scalars=glyphs["GlyphVector"][:, 2],
            lighting=False,
            cmap="coolwarm",
            clim=[-1, 1],
            scalar_bar_args={"title": "m_z"},
        )
        pl.show_axes()
        if plotter is None:
            pl.show(jupyter_backend=jupyter_backend)
