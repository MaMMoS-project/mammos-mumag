"""Parameters class.

The simulation parameters define discretization parameters such as the scale of the
system, initial values such as initial magnetization state, and solver options
such as the convergence tolerance.
"""

import configparser
import os
from pathlib import Path
from typing import Any

import mammos_entity as me
import mammos_units as u
import numpy as np
from jinja2 import Environment, PackageLoader, select_autoescape
from pydantic import ConfigDict, Field, field_validator
from pydantic.dataclasses import dataclass


@dataclass(
    config=ConfigDict(
        arbitrary_types_allowed=True, extra="forbid", validate_assignment=True
    )
)
class Parameters:
    """Class storing simulation parameters.

    They include discretization parameters, initial values, and solver options.
    """

    size: float = 1.0e-09
    """Size of the mesh. This factor usually indicates the magnitude of the
    geometry, i.e., ``1e-9`` for nanometer meshes, ``1e-6`` for micrometer, etc."""
    scale: float = 0.0
    """This factor can be used to define the volume of the magnetic material.
    If ``scale`` is different from 0, the volume of the magnetic material is evaluated
    as ``scale ** 3``. This only makes sense if the system is a cube with side
    length equal to ``scale``.

    If equal to 0, this parameter is ignored, and the volume
    is evaluated by integrating the constant 1 over the domain where spontaneous
    magnetization of the material is positive. In those case where the system is a
    multigrain cube with possible non-magnetic integranular phases, evaluating the
    volume using the ``scale`` parameter can be more accurate."""
    state: str = ""
    r"""Name of the initial magnetization state. The following strings are recognized:

    * ``"flower"`` for a flower state. The magnetization is defined as:

      .. math::
        \mathbf{m} = \left[\begin{array}{c}
          \frac{xz}{10s} \\
          \frac{yz}{10s} \\
          1
        \end{array}\right],

      where :math:`s = \max \{x,y,z\}` is the maximum value over all components and
      all mesh points.

    * ``"vortex"`` for a vortex state on the :math:`xy` plane. If
      :math:`r = r(x,y) = \sqrt{x^2 + y^2}` is the radial variable on such plane, the
      magnetization is defined as:

      .. math::
        \mathbf{m} = \left[\begin{array}{c}
          e^{-2 r / R} \\
          - \frac{z}{r} \sqrt{1 - e^{-4 r^2 / R^2}} \\
          \frac{y}{r} \sqrt{1 - e^{-4 r^2 / R^2}}
        \end{array}\right],

      where :math:`R = 0.14 * \max r(x,y)`.

    * ``"twisted"`` for a twisted state. This is a mixture of vortex and flower, where
      the vortex appears on the :math:`xy` plane and the flower in the :math:`z`
      direction connects the two vortices with opposing chirality. If also in this case
      :math:`r = r(x,y) = \sqrt{x^2 + y^2}` is the radial variable on the :math:`xy`
      plane, the magnetization is defined as:

      .. math::
        \mathbf{m} = \left[\begin{array}{c}
          \frac{xz}{10s} - 4 \frac{yz}{rs} \\
          \frac{yz}{10s} + 4 \frac{xz}{rs} \\
          1
        \end{array}\right],

      where :math:`s = \max \{x,y,z\}` is the maximum value over all components and
      all mesh points.

    * ``"random"`` for a randomly magnetized state. In particular, each magnetization
      component is generated uniformly in :math:`[-1, 1]`.


    Other strings are interpreted as a uniformly magnetized state.
    The default value is ``""`` and defines a uniformly magnetized state.
    """
    h_mag_on: bool = True
    """Whether the external field is on (True) or off (False)."""
    h_start: me.Entity = Field(
        default_factory=lambda: me.Entity(
            "ExternalMagneticField",
            (10 * u.T).to(u.A / u.m, equivalencies=u.magnetic_flux_field()),
        )
    )
    """Initial strength of the external field as an :entity:`ExternalMagneticField`.
    Interpreted in A/m if passed without unit.
    The default value is the equivalent of 10 Tesla."""
    h_final: me.Entity = Field(
        default_factory=lambda: me.Entity(
            "ExternalMagneticField",
            (-10 * u.T).to(u.A / u.m, equivalencies=u.magnetic_flux_field()),
        )
    )
    """Final strength of the external field as an :entity:`ExternalMagneticField`.
    Interpreted in A/m if passed without unit.
    The default value is the equivalent of -10 Tesla."""
    h_step: me.Entity = Field(
        default_factory=lambda: me.Entity(
            "ExternalMagneticField",
            (-1 * u.T).to(u.A / u.m, equivalencies=u.magnetic_flux_field()),
        )
    )
    """Step size of external magnetic field in the hysteresis loop as an
    :entity:`ExternalMagneticField`. Interpreted in A/m if passed without unit.
    The default value is the equivalent of -1 Tesla."""
    h_vect: me.Entity = Field(
        default_factory=lambda: me.Entity(
            "Vector",
            [0, 0, 1],
        )
    )
    r"""External field direction vector :math:`\mathbf{h}` as a :entity:`Vector`.
    If any iterable of floats is used (such as a list or a tuple of length 3), it will
    be casted internally. This vector must not be normal. A private property is
    internally used to normalize it. The default value is the unit vector [0, 0, 1]."""
    m_step: me.Entity = Field(
        default_factory=lambda: me.Entity(
            "Magnetization",
            (1 * u.T).to(u.A / u.m, equivalencies=u.magnetic_flux_field()),
        )
    )
    """Threshold determining when magnetization profiles are saved as a
    :entity:`Magnetization`. If in the hysteresis calculation the difference
    between two consecutive values of magnetization (intended along the direction of the
    external field :py:attr:`~mammos_mumag.parameters.Parameters.h_vect`) is bigger than
    this value, the current magnetization field is saved as a ``vtu`` file and the
    configuration index (appearing in :py:attr:`~mammos_mumag.hysteresis.Result`) will
    increase. In practice this value determines how often we save a magnetization file.
    A very low value will impact performance as every step will produce such a file.
    Interpreted in A/m if passed without unit.
    The default value is the equivalent of -2 Tesla."""
    m_final: me.Entity = Field(
        default_factory=lambda: me.Entity(
            "Magnetization",
            (-2 * u.T).to(u.A / u.m, equivalencies=u.magnetic_flux_field()),
        ),
    )
    """Value of :entity:`Magnetization` (along the external field direction) at which
    the hysteresis calculation will stop. Interpreted in A/m if passed without unit."""
    m_vect: me.Entity = Field(
        default_factory=lambda: me.Entity(
            "Vector",
            [0, 0, 1],
        )
    )
    """Initial magnetization direction as a :entity:`Vector`. This value will be
    modified by the choice :py:attr:`~mammos_mumag.parameters.Parameters.state` unless
    a uniform state is selected. If any iterable of floats is used (such as a list or a
    tuple of length 3), it will be casted into the ``Vector`` entity internally. This
    vector must not be normal. A private property is internally used to normalize it.
    The default value is the unit vector [0, 0, 1]."""
    precond_iter: int = 10
    """Number of iteration for the approximation of the inverse Hessian in the conjugate
    gradient optimization."""
    tol_fun: float = 1e-10
    """Total energy tolerance to obtain the equilibrium configuration."""
    tol_h_mag_factor: float = 1.0
    """Factor defining the tolerance for the magnetostatic scalar
    potential according to the formula ``tol_u``
    = :py:attr:`~mammos_mumag.parameters.Parameters.tol_fun` *
    :py:attr:`~mammos_mumag.parameters.Parameters.tol_h_mag_factor`."""
    filepath: os.PathLike | str | None = Field(default=None, repr=False)
    """Path of parameter file (in format ``p2`` or ``yaml``) to read at initialization.
    In this case, all other parameters will be overwritten if specified in the parameter
    file."""

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

    def read(self, fname: str | os.PathLike) -> None:
        """Read parameter file.

        This function only overwrites the parameters defined in the file.

        Supported format are ``p2`` (with extension ``.p2``) and ``yaml``
        (with extensions ``.yaml`` or ``.yml``).

        Args:
            fname: File to read.

        Raises:
            FileNotFoundError: Parameter file not found.
            NotImplementedError: Wrong file format.

        """
        fpath = Path(fname)
        if not fpath.is_file():
            raise FileNotFoundError(f"File {fpath} not found.")

        if fpath.suffix == ".yaml":
            self._read_yaml(fpath)

        elif fpath.suffix == ".p2":
            self._read_p2(fpath)

        else:
            raise NotImplementedError(
                f"{fpath.suffix} parameter file is not supported."
            )

    def _read_p2(self, fpath: str | os.PathLike) -> None:
        """Read parameter file in ``p2`` format.

        The speciality of this file format is that magnetization values are stored
        in Tesla for readability. Hence, they need to be converted to A/m first.
        Furthermore, in this format some of the names have a specific formatting
        different than the one used in the attributes of this class.

        Args:
            fpath: File to read.
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

    def _read_yaml(self, fpath: str | os.PathLike) -> None:
        """Read parameter file in ``yaml`` format.

        We expect the parameters to be saved in the mammos yaml format. See
        :py:func:`mammos_entity.EntityCollection.to_yaml` for more information
        on this format.

        Args:
            fpath: File to read.
        """
        content = me.from_yaml(fpath)
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

    def write_p2(self, fname: str | os.PathLike) -> None:
        """Write parameter file in the ``p2`` format.

        Args:
            fname: File to write.

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

    def write_yaml(self, fname: str | os.PathLike) -> None:
        """Write parameter file in the  ``yaml`` format.

        Args:
            fname: File to write.

        Examples:
            >>> from mammos_mumag.parameters import Parameters
            >>> par = Parameters()
            >>> par.write_yaml("parameters.yaml")

        """
        collection = me.EntityCollection(
            description="File containing simulation parameters.",
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
        collection.to_yaml(fname)


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
