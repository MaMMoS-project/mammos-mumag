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

from mammos_mumag.materials import Materials
from mammos_mumag.parameters import Parameters
from mammos_mumag.simulation import Simulation

if TYPE_CHECKING:
    import matplotlib
    import pyvista

    import mammos_mumag


def run(
    Ms: float | u.Quantity | me.Entity,
    A: float | u.Quantity | me.Entity,
    K1: float | u.Quantity | me.Entity,
    mesh: mammos_mumag.mesh.Mesh | pathlib.Path | str,
    hstart: float | u.Quantity,
    hfinal: float | u.Quantity,
    hstep: float | u.Quantity | None = None,
    hnsteps: int = 20,
    outdir: str | pathlib.Path = "hystloop",
) -> mammos_mumag.hysteresis.Result:
    r"""Run hysteresis loop.

    Args:
        Ms: Spontaneous magnetisation in :math:`\mathrm{A}/\mathrm{m}`.
        A: Exchange stiffness constant in :math:`\mathrm{J}/\mathrm{m}`.
        K1: First magnetocrystalline anisotropy constant in
            :math:`\mathrm{J}/\mathrm{m}^3`.
        mesh: The mesh can either be given as a :py:class:`~mammos_mumag.mesh.Mesh`
            instance (for meshes available through `mammos_mumag`) or its path can be
            specified. The only possible mesh format is `.fly`.
        hstart: Initial strength of the external field.
        hfinal: Final strength of the external field.
        hstep: Step size.
        hnsteps: Number of steps in the field sweep.
        outdir: Directory where simulation results are written to.

    Returns:
       Result object.

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
        mesh=mesh,
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
    return read_result(outdir=outdir, name="hystloop")


def read_result(
    outdir: str | pathlib.Path,
    name: str = "out",
) -> mammos_mumag.hysteresis.Result:
    r"""Read hysteresis loop output from directory.

    Args:
        outdir: Path of output directory where the results of the hysteresis loop are
            stored.
        name: System name with which the loop output files are stored.

    Returns:
       Result object.

    Raises:
        FileNotFoundError: hysteresis loop .dat file not found.

    """
    try:
        res = me.io.entities_from_file(pathlib.Path(outdir) / f"{name}.csv")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Hysteresis file {name}.csv not found in outdir='{outdir}'."
        ) from None
    return Result(
        H=me.Entity(
            "ExternalMagneticField",
            value=res.B_ext.q.to(u.A / u.m, equivalencies=u.magnetic_flux_field()),
            unit=u.A / u.m,
        ),
        M=me.Ms(
            res.J.q.to(u.A / u.m, equivalencies=u.magnetic_flux_field()),
            unit=u.A / u.m,
        ),
        Mx=me.Ms(
            res.Jx.q.to(u.A / u.m, equivalencies=u.magnetic_flux_field()),
            unit=u.A / u.m,
        ),
        My=me.Ms(
            res.Jy.q.to(u.A / u.m, equivalencies=u.magnetic_flux_field()),
            unit=u.A / u.m,
        ),
        Mz=me.Ms(
            res.Jz.q.to(u.A / u.m, equivalencies=u.magnetic_flux_field()),
            unit=u.A / u.m,
        ),
        energy_density=res.energy_density,
        configurations={
            i + 1: fname
            for i, fname in enumerate(
                sorted(pathlib.Path(outdir).resolve().glob("*.vtu"))
            )
        },
        configuration_type=np.asarray(res.configuration_type),
    )


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, frozen=True))
class Result:
    """Hysteresis loop Result."""

    H: me.Entity
    """Array of external field strengths."""
    M: me.Entity
    """Array of spontaneous magnetization values for the field strengths in the
    direction of H."""
    Mx: me.Entity
    """Component x of the spontaneous magnetization."""
    My: me.Entity
    """Component y of the spontaneous magnetization."""
    Mz: me.Entity
    """Component z of the spontaneous magnetization."""
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
                "Mx": self.Mx.q,
                "My": self.My.q,
                "Mz": self.Mz.q,
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
            duplicate_change_color: If set to false use the same color for both branches
                of the hysteresis plot.
            configuration_marks: Show markers where a configuration has been saved.
            ax: Matplotlib axes object to which the plot is added. A new one is create
                if not passed.
            label: Label shown in the legend. A legend is automatically added to the
                plot if this argument is not None.
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
