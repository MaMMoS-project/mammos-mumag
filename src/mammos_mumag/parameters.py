"""Parameters class."""

import configparser
import pathlib
from typing import Any

import mammos_entity as me
import mammos_units as u
import numpy as np
from jinja2 import Environment, PackageLoader, select_autoescape
from pydantic import ConfigDict, Field, field_validator
from pydantic.dataclasses import dataclass

from mammos_mumag.tools import check_path


@dataclass(
    config=ConfigDict(
        arbitrary_types_allowed=True, extra="forbid", validate_assignment=True
    )
)
class Parameters:
    r"""Class storing simulation parameters.

    Args:
        size: Size of the mesh. This factor usually indicates the magnitude of the
            geometry, i.e., 1e-9 for nanometer meshes, 1e-9 for micrometer, etc.
        scale: Scale of the mesh. This factor can include other scaling, so that
            the total scale of the mesh is `size` * `scale`.
        state: Name of the initial magnetization state. Scripts recognize the strings
            `flower`, `vortex`, `twisted`, and `random`. Other strings are interpreted
            as the default case. The default case is a uniformly magnetized state.
        h_mag_on: Whether the external field is on (True) or off (False).
        h_start: Strength of the external field that the hysteresis loop starts from.
        h_final: Strength of the external field that the hysteresis loop finishes at.
        h_step: Difference in field strength between two hysteresis loop measurements.
        h_vect: External field vector :math:`\mathbf{h}` as a list of floats or a
            `Vector` entity. This vector is not necessarily normal. The property `h`
            will be the normalized field. If not defined, the external field is zero.
        m_step: Threshold at which the magnetization is saved. If in the hysteresis
            calculation the difference between two consecutive values of magnetization
            along the external field vector is bigger than this value, a new
            configuration index will appear on the output csv table. Different
            configurations mean that the magnetization is behaving differently,
            possibly changing states.
        m_final: Value of magnetization (along the external field direction) at which
            the hysteresis calculation will stop.
        m_vect: Magnetization field :math:`\mathbf{m}` as a list of floats or a `Vector`
            entity.
        precond_iter: Conjugate gradient iterations for inverse Hessian approximation.
        tol_fun: Total energy tolerance to obtain the equilibrium configuration.
        tol_h_mag_factor: Factor defining the tolerance for the magnetostatic scalar
            potential according to the formula `tol_u = tol_fun * tol_h_mag_factor`.
        filepath: Path of parameter file to read at initialization. In this case all
            other parameters will be overwritten (if specified in the parameter file).
    """

    size: float = 1.0e-09
    scale: float = 0.0
    state: str = Field(default_factory=lambda: "")
    h_mag_on: bool = True
    h_start: me.Entity = Field(
        default_factory=lambda: me.Entity(
            "ExternalMagneticField",
            (10 * u.T).to(u.A / u.m, equivalencies=u.magnetic_flux_field()),
        )
    )
    h_final: me.Entity = Field(
        default_factory=lambda: me.Entity(
            "ExternalMagneticField",
            (-10 * u.T).to(u.A / u.m, equivalencies=u.magnetic_flux_field()),
        )
    )
    h_step: me.Entity = Field(
        default_factory=lambda: me.Entity(
            "ExternalMagneticField",
            (-1 * u.T).to(u.A / u.m, equivalencies=u.magnetic_flux_field()),
        )
    )
    h_vect: me.Entity = Field(
        default_factory=lambda: me.Entity(
            "Vector",
            [0, 0, 1],
        )
    )
    m_step: me.Entity = Field(
        default_factory=lambda: me.Entity(
            "Magnetization",
            (1 * u.T).to(u.A / u.m, equivalencies=u.magnetic_flux_field()),
        )
    )
    m_final: me.Entity = Field(
        default_factory=lambda: me.Entity(
            "Magnetization",
            (-2 * u.T).to(u.A / u.m, equivalencies=u.magnetic_flux_field()),
        ),
    )
    m_vect: me.Entity = Field(
        default_factory=lambda: me.Entity(
            "Vector",
            [0, 0, 1],
        )
    )
    precond_iter: int = 10
    tol_fun: float = 1e-10
    tol_h_mag_factor: float = 1.0
    filepath: pathlib.Path | None = Field(default=None, repr=False)

    @field_validator("h_start", mode="before")
    @classmethod
    def _convert_h_start(cls, h_start: Any) -> me.Entity:
        """Convert h_start to the rigth Entity."""
        h_start = me.Entity("ExternalMagneticField", h_start, unit=u.A / u.m)
        return h_start

    @field_validator("h_final", mode="before")
    @classmethod
    def _convert_h_final(cls, h_final: Any) -> me.Entity:
        """Convert h_final to the right Entity."""
        h_final = me.Entity("ExternalMagneticField", h_final, unit=u.A / u.m)
        return h_final

    @field_validator("h_step", mode="before")
    @classmethod
    def _convert_h_step(cls, h_step: Any) -> me.Entity:
        """Convert h_step to the right Entity."""
        h_step = me.Entity("ExternalMagneticField", h_step, unit=u.A / u.m)
        return h_step

    @field_validator("h_vect", mode="before")
    @classmethod
    def _convert_h_vect(cls, h_vect: Any) -> me.Entity:
        """Convert h_vect to the right Entity."""
        h_vect = me.Entity("Vector", h_vect)
        if h_vect.q.size != 3:
            raise ValueError(
                f"`h_vect` has the wrong size ({h_vect.q.size} instead of 3)."
            )
        return h_vect

    @field_validator("m_step", mode="before")
    def _convert_m_step(cls, m_step: Any) -> me.Entity:
        """Convert m_step to the right Entity."""
        m_step = me.Entity("Magnetization", m_step, unit=u.A / u.m)
        return m_step

    @field_validator("m_final", mode="before")
    def _convert_m_final(cls, m_final: Any) -> me.Entity:
        """Convert m_final to the right Entity."""
        m_final = me.Entity("Magnetization", m_final, unit=u.A / u.m)
        return m_final

    @field_validator("m_vect", mode="before")
    @classmethod
    def _convert_m_vect(cls, m_vect: Any) -> me.Entity:
        """Convert m_vect to the right Entity."""
        m_vect = me.Entity("Vector", m_vect)
        if m_vect.q.size != 3:
            raise ValueError(
                f"`m_vect` has the wrong size ({m_vect.q.size} instead of 3)."
            )
        return m_vect

    def __post_init__(self) -> None:
        """Initialize parameters with a file.

        If the parameters is initialized with a not-`None` `filepath`
        attribute, the materials files will be read automatically.
        """
        if self.filepath is not None:
            self.read(self.filepath)

    @property
    def m(self) -> list[float]:
        """Normalized magnetization."""
        return _normalize(self.m_vect)

    @property
    def h(self) -> list[float]:
        """Direction of the external field."""
        return _normalize(self.h_vect)

    def read(self, fname: str | pathlib.Path) -> None:
        """Read parameter file.

        Args:
            fname: File path

        Raises:
            NotImplementedError: Wrong file format.

        """
        fpath = check_path(fname)

        if fpath.suffix == ".yaml":
            self.read_yaml(fpath)

        elif fpath.suffix == ".p2":
            self.read_p2(fpath)

        else:
            raise NotImplementedError(
                f"{fpath.suffix} parameter file is not supported."
            )

    def read_p2(self, fpath: str | pathlib.Path) -> None:
        """Read parameter file in `p2` format.

        The speciality of this file format is that magnetization values are stored
        in Tesla for readability. Hence, they need to be converted to A/m first.
        Furthermore, in this format some of the names have a special formatting.

        Args:
            fpath: Parameter file path.
        """
        u.set_enabled_equivalencies(u.magnetic_flux_field())
        pars = configparser.ConfigParser()
        pars.read(fpath)

        mesh = pars["mesh"]
        if "size" in mesh:
            self.size = float(mesh["size"])
        if "scale" in mesh:
            self.scale = float(mesh["scale"])

        initial_state = pars["initial state"]
        if "state" in initial_state:
            self.state = str(initial_state["state"])
        self.m_vect = [
            float(initial_state["mx"]),
            float(initial_state["my"]),
            float(initial_state["mz"]),
        ]

        field = pars["field"]
        if "hmag_on" in field:
            self.h_mag_on = bool(field["hmag_on"])
        self.h_start = me.Entity(
            "ExternalMagneticField", (float(field["hstart"]) * u.T).to(u.A / u.m)
        )
        self.h_final = me.Entity(
            "ExternalMagneticField", (float(field["hfinal"]) * u.T).to(u.A / u.m)
        )
        self.h_step = me.Entity(
            "ExternalMagneticField", (float(field["hstep"]) * u.T).to(u.A / u.m)
        )
        self.h_vect = [
            float(field["hx"]),
            float(field["hy"]),
            float(field["hz"]),
        ]
        if "mstep" in field:
            self.m_step = me.Entity(
                "Magnetization", (float(field["mstep"]) * u.T).to(u.A / u.m)
            )
        if "mfinal" in field:
            self.m_final = me.Entity(
                "Magnetization", (float(field["mfinal"]) * u.T).to(u.A / u.m)
            )

        minimizer = pars["minimizer"]
        if "precond_iter" in minimizer:
            self.precond_iter = int(minimizer["precond_iter"])
        if "tol_fun" in minimizer:
            self.tol_fun = float(minimizer["tol_fun"])
        if "tol_hmag_factor" in minimizer:
            self.tol_h_mag_factor = float(minimizer["tol_hmag_factor"])
        if "truncation" in minimizer:
            self.truncation = int(minimizer["truncation"])

    def read_yaml(self, fpath: str | pathlib.Path) -> None:
        """Read parameter file in `yaml` format.

        We expect the parameters to be saved using the module :py:mod:`mammos_entity.io`
        and use the function :py:func:`mammos_entity.io.entities_from_file`.

        Args:
            fpath: Parameter file path.
        """
        content = me.io.entities_from_file(fpath)
        self.size = content.mesh_size
        self.scale = content.mesh_scale
        self.state = content.initial_state
        self.m_vect = [content.initial_mx, content.initial_my, content.initial_mz]
        self.h_mag_on = content.h_mag_on
        self.h_start = content.h_start
        self.h_final = content.h_final
        self.h_step = content.h_step
        self.h_vect = [content.hx, content.hy, content.hz]
        self.m_step = content.m_step
        self.m_final = content.m_final
        self.tol_fun = content.minimizer_tol_fun
        self.tol_h_mag_factor = content.minimizer_tol_h_mag_factor
        self.precond_iter = content.minimizer_precond_iter

    def write_p2(self, fname: str | pathlib.Path) -> None:
        """Write parameter `p2` file.

        Args:
            fname: File path

        Examples:
            >>> from mammos_mumag.parameters import Parameters
            >>> par = Parameters()
            >>> par.write_p2("parameters.p2")

        """
        u.set_enabled_equivalencies(u.magnetic_flux_field())
        env = Environment(
            loader=PackageLoader("mammos_mumag"),
            autoescape=select_autoescape(),
        )
        template = env.get_template("p2.jinja")
        parameters_dict = {
            **self.__dict__,
            "mx": self.m[0],
            "my": self.m[1],
            "mz": self.m[2],
            "hx": self.h[0],
            "hy": self.h[1],
            "hz": self.h[2],
            "hmag_on": int(self.h_mag_on),
            "hstart": self.h_start.q.to(u.T).value,
            "hfinal": self.h_final.q.to(u.T).value,
            "hstep": self.h_step.q.to(u.T).value,
            "mstep": self.m_step.q.to(u.T).value,
            "mfinal": self.m_final.q.to(u.T).value,
            "tol_hmag_factor": self.tol_h_mag_factor,
        }
        with open(fname, "w") as file:
            file.write(template.render(parameters_dict))

    def write_yaml(self, fname: str | pathlib.Path) -> None:
        """Write parameter `yaml` file.

        Args:
            fname: File path

        Examples:
            >>> from mammos_mumag.parameters import Parameters
            >>> par = Parameters()
            >>> par.write_yaml("parameters.yaml")

        """
        me.io.entities_to_file(
            fname,
            "File containing simulation parameters.",
            mesh_size=self.size,
            mesh_scale=self.scale,
            initial_state=self.state,
            initial_mx=self.m[0],
            initial_my=self.m[1],
            initial_mz=self.m[2],
            h_mag_on=self.h_mag_on,
            h_start=self.h_start,
            h_final=self.h_final,
            h_step=self.h_step,
            hx=self.h[0],
            hy=self.h[1],
            hz=self.h[2],
            m_step=self.m_step,
            m_final=self.m_final,
            minimizer_tol_fun=self.tol_fun,
            minimizer_tol_h_mag_factor=self.tol_h_mag_factor,
            minimizer_precond_iter=self.precond_iter,
        )


def _normalize(vector: me.Entity) -> list[float]:
    """Normalize Vector Entity and transform it into a list of float.

    Args:
        vector: 3D Vector Entity to normalize.

    """
    v = vector.value
    s = np.linalg.norm(v)
    if s <= 1.0e-13:
        return list(v)
    else:
        return list(v / s)
