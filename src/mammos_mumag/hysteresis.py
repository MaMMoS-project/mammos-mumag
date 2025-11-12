"""Functions for evaluating and processin the hysteresis loop."""

from __future__ import annotations

import pathlib
from numbers import Number
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

    import mammos_mumag


def run(
    Ms: Number | u.Quantity | me.Entity,
    A: Number | u.Quantity | me.Entity,
    K1: Number | u.Quantity | me.Entity,
    theta: Number | u.Quantity | me.Entity,
    phi: Number | u.Quantity | me.Entity,
    mesh: mammos_mumag.mesh.Mesh | pathlib.Path | str,
    h_start: Number | u.Quantity | me.Entity | None = None,
    h_final: Number | u.Quantity | me.Entity | None = None,
    h_step: Number | u.Quantity | me.Entity | None = None,
    h_n_steps: int = 20,
    m_final: Number | u.Quantity | me.Entity | None = None,
    outdir: str | pathlib.Path = "hystloop",
) -> mammos_mumag.hysteresis.Result:
    r"""Run hysteresis loop.

    Args:
        Ms: Spontaneous magnetisation in :math:`\mathrm{A}/\mathrm{m}`.
        A: Exchange stiffness constant in :math:`\mathrm{J}/\mathrm{m}`.
        K1: First magnetocrystalline anisotropy constant in
            :math:`\mathrm{J}/\mathrm{m}^3`.
        theta: Angle of the magnetocrystalline anisotropy axis from the
            :math:`z`-direction in radians.
        phi: Angle of the magnetocrystalline anisotropy axis from the
            :math:`x`-direction in radians.
        mesh: The mesh can either be given as a :py:class:`~mammos_mumag.mesh.Mesh`
            instance (for meshes available through `mammos_mumag`) or its path can be
            specified. The only possible mesh format is `.fly`.
        h_start: Initial strength of the external field in
            :math:`\mathrm{A}/\mathrm{m}`.
        h_final: Final strength of the external field in :math:`\mathrm{A}/\mathrm{m}`.
        h_step: Step size in :math:`\mathrm{A}/\mathrm{m}`.
        h_n_steps: Number of steps in the field sweep.
        m_final: Value of magnetization (along the external field direction) at which
            the hysteresis calculation will stop in :math:`\mathrm{A}/\mathrm{m}`.
        outdir: Directory where simulation results are written to.

    Returns:
       Hysteresis result object.

    """
    Ms = me.Ms(Ms, unit=u.A / u.m)
    A = me.A(A, unit=u.J / u.m)
    K1 = me.Ku(K1, unit=u.J / u.m**3)
    theta = me.Entity("Angle", theta)
    phi = me.Entity("Angle", phi)

    if (
        len(set(e.q.size for e in [Ms, A, K1, theta, phi])) != 1
        or len(set(e.q.shape for e in [Ms, A, K1, theta, phi])) != 1
    ):
        raise ValueError("All material parameters must have the same length.")

    materials = Materials()
    if Ms.q.shape:  # More than zero dimensions
        for Ms_i, A_i, K1_i, theta_i, phi_i in zip(
            Ms.q.flatten(),
            A.q.flatten(),
            K1.q.flatten(),
            theta.q.flatten(),
            phi.q.flatten(),
            strict=True,
        ):
            materials.domains.append(
                MaterialDomain(
                    Ms=Ms_i,
                    A=A_i,
                    K1=K1_i,
                    theta=theta_i,
                    phi=phi_i,
                )
            )
    else:  # entities are zero dimensions
        materials.domains.append(
            MaterialDomain(
                Ms=Ms,
                A=A,
                K1=K1,
                theta=theta,
                phi=phi,
            )
        )
    materials.domains.append(MaterialDomain())  # empty domain for air
    materials.domains.append(MaterialDomain())  # empty domain for shell

    parameters = Parameters()  # initialize default simulation parameters

    if h_start is not None:
        parameters.h_start = me.Entity("ExternalMagneticField", h_start, unit=u.A / u.m)
    if h_final is not None:
        parameters.h_final = me.Entity("ExternalMagneticField", h_final, unit=u.A / u.m)
    if h_step is not None:
        parameters.h_step = me.Entity("ExternalMagneticField", h_step, unit=u.A / u.m)
    else:
        parameters.h_step = me.Entity(
            "ExternalMagneticField",
            (parameters.h_final.q - parameters.h_start.q) / h_n_steps,
            unit=u.A / u.m,
        )
    if m_final is not None:
        parameters.m_final = me.Entity("Magnetization", m_final, unit=u.A / u.m)

    sim = Simulation(
        mesh=mesh,
        materials=materials,
        parameters=parameters,
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
    r"""Array of external field strengths in :math:`\mathrm{A}/\mathrm{m}`."""
    M: me.Entity
    r"""Array of spontaneous magnetization values for the field strengths in the
    direction of H in :math:`\mathrm{A}/\mathrm{m}`."""
    Mx: me.Entity
    r"""Component x of the spontaneous magnetization in
    :math:`\mathrm{A}/\mathrm{m}`."""
    My: me.Entity
    r"""Component y of the spontaneous magnetization in
    :math:`\mathrm{A}/\mathrm{m}`."""
    Mz: me.Entity
    r"""Component z of the spontaneous magnetization in
    :math:`\mathrm{A}/\mathrm{m}`."""
    energy_density: me.Entity | None = None
    r"""Array of energy densities for the field strengths in
    :math:`\mathrm{J}/\mathrm{m^3}`."""
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
        tesla: bool = False,
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
            tesla: If true, plots External Magnetic Flux Density B instead of External
                Magnetic Field H and Spontaneous Polarisation Js instead of Spontaneous
                Magnetization Ms.
            kwargs: Additional keyword arguments passed to `ax.plot` when plotting the
                hysteresis lines.

        Returns:
            The `matplotlib.axes.Axes` object which was used to plot the hysteresis loop

        """
        if ax:
            ax = ax
        else:
            _, ax = plt.subplots()
        df = self.dataframe
        if tesla:
            B = me.B(self.H.q.to("T", equivalencies=u.magnetic_flux_field()))
            J = me.J(self.M.q.to("T", equivalencies=u.magnetic_flux_field()))
            df["x"] = B.q
            df["y"] = J.q
            x_label = B.axis_label
            y_label = J.axis_label
        else:
            df = df.rename(columns={"H": "x", "M": "y"})
            x_label = self.H.axis_label
            y_label = self.M.axis_label
        if label:
            (line,) = ax.plot(df.x, df.y, label=label, **kwargs)
        else:
            (line,) = ax.plot(df.x, df.y, **kwargs)
        j = 0
        if configuration_marks:
            for _, row in df.iterrows():
                idx = int(row.configuration_type)
                if idx != j:
                    plt.plot(row.x, row.y, "rx")
                    j = idx
                    ax.annotate(
                        j,
                        xy=(row.x, row.y),
                        xytext=(-2, -10),
                        textcoords="offset points",
                    )
        ax.set_title("Hysteresis Loop")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if label:
            ax.legend()
        if duplicate:
            if not duplicate_change_color:
                kwargs.setdefault("color", line.get_color())
            ax.plot(-df.x, -df.y, **kwargs)

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
