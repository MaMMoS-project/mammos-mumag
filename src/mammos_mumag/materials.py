"""Materials class.

This submodule contains classes to describe (possibly multigrain) magnetic materials.

* :py:class:`~mammos_mumag.materials.MaterialDomain` contains magnetic properties for
  each domain.
* :py:class:`~mammos_mumag.materials.Materials` contains information about the whole
  magnetic material. Together with a list of
  :py:class:`~mammos_mumag.materials.MaterialDomain` objects, this class has also
  methods to read and write material information.
"""

from __future__ import annotations

import numbers
import os
from pathlib import Path
from typing import Any

import mammos_entity as me
import mammos_units as u
import yaml
from jinja2 import Environment, PackageLoader, select_autoescape
from pydantic import ConfigDict, Field, field_validator
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class MaterialDomain:
    """Uniform material domain.

    A domain is a volume with constant magnetic parameters.
    This class contains the values of these magnetic parameters.

    Geometric information are not included.
    """

    theta: me.Entity = Field(default=me.Entity("Angle"))
    """Angle of the magnetocrystalline anisotropy axis from the :math:`z`-direction
    as an ``Angle``. Interpreted in radians if passed without unit.
    Default value is zero."""
    phi: me.Entity = Field(default=me.Entity("Angle"))
    """Angle of the magnetocrystalline anisotropy axis from the :math:`x`-direction
    as an ``Angle``. Interpreted in radians if passed without unit.
    Default value is zero."""
    K1: me.Entity = Field(default=me.Entity("MagnetocrystallineAnisotropyConstantK1"))
    r"""First magnetocrystalline anisotropy constant as
    `MagnetocrystallineAnisotropyConstantK1`, defined by the uniaxial anisotropy energy
    density :math:`K_1 \sin^2(\theta)`, where :math:`\theta` is the angle between the
    anisotropy axis and the magnetization. Interpreted in J/m³ if passed without unit.
    """
    Ms: me.Entity = Field(default=me.Entity("SpontaneousMagnetization"))
    """Spontaneous magnetization as ``SpontaneousMagnetization``.
    Interpreted in A/m if passed without unit."""
    A: me.Entity = Field(default=me.Entity("ExchangeStiffnessConstant"))
    """Exchange stiffness constant as ``ExchangeStiffnessConstant``.
    Interpreted in J/m if passed without unit."""

    @field_validator("theta", mode="before")
    @classmethod
    def _convert_theta(cls, theta: Any) -> Any:
        """Convert number or Quantity to Entity."""
        if isinstance(theta, u.Quantity):
            with u.set_enabled_equivalencies(u.dimensionless_angles()):
                theta = theta.to("")
        return me.Entity("Angle", theta)

    @field_validator("phi", mode="before")
    @classmethod
    def _convert_phi(cls, phi: Any) -> Any:
        """Convert number or Quantity to Entity."""
        if isinstance(phi, u.Quantity):
            with u.set_enabled_equivalencies(u.dimensionless_angles()):
                phi = phi.to("")
        return me.Entity("Angle", phi)

    @field_validator("K1", mode="before")
    @classmethod
    def _convert_K1(cls, K1: Any) -> Any:
        """Convert number or Quantity to Entity."""
        if isinstance(K1, numbers.Real | u.Quantity):
            K1 = me.Ku(K1, unit=u.J / u.m**3)
        return K1

    @field_validator("A", mode="before")
    @classmethod
    def _convert_A(cls, A: Any) -> Any:
        """Convert number or Quantity to Entity."""
        if isinstance(A, numbers.Real | u.Quantity):
            A = me.A(A, unit=u.J / u.m)
        return A

    @field_validator("Ms", mode="before")
    @classmethod
    def _convert_Ms(cls, Ms: Any) -> Any:
        """Convert number or Quantity to Entity."""
        if isinstance(Ms, numbers.Real | u.Quantity):
            Ms = me.Ms(Ms, unit=u.A / u.m)
        return Ms


@dataclass
class Materials:
    """This class stores, reads, and writes material parameters."""

    domains: list[MaterialDomain] = Field(default_factory=list)
    """Each domain is a MaterialDomain class of material parameters, constant in each
    region."""
    filepath: os.PathLike | None = Field(default=None, repr=False)
    """Material file path."""

    def __post_init__(self) -> None:
        """Initialize materials with a file.

        If the materials is initialized with an empty `domains` attribute
        and with a not-`None` `filepath` attribute, the materials files
        will be read automatically.
        """
        if (len(self.domains) == 0) and (self.filepath is not None):
            self.read(self.filepath)

    def add_domain(
        self, A: float, Ms: float, K1: float, phi: float, theta: float
    ) -> None:
        """Append domain with specified parameters.

        All the inputs should be float numbers and be without unit.
        They should be however be expressed in specific units (specified
        in each argument docstring).

        Args:
            A: Exchange stiffness constant in J/m.
            Ms: Spontaneous magnetization in A/m.
            K1: First magnetocrystalline anisotropy constant in J/m³.
            phi: Angle of the magnetocrystalline anisotropy axis
                from the :math:`x`-direction in radians.
            theta: Angle of the magnetocrystalline anisotropy axis
                from the :math:`z`-direction in radians.

        Examples:
            >>> from mammos_mumag.materials import Materials
            >>> mat = Materials()
            >>> mat.add_domain(A=1, Ms=2, K1=3, phi=0, theta=0)
            >>> mat
            Materials(domains=[MaterialDomain(theta=..., phi=..., K1=..., Ms=..., A=...)])

        """  # noqa: E501
        dom = MaterialDomain(
            theta=theta,
            phi=phi,
            K1=K1,
            Ms=Ms,
            A=A,
        )
        self.domains.append(dom)

    def read(self, fname: str | os.PathLike) -> None:
        """Read material information from file.

        This function overwrites the current
        :py:attr:`~mammos_mumag.materials.Materials.domains` attribute, hence
        overwriting all material parameters previously defined.

        Supported formats are ``krn`` (with extension ``.krn``) and ``yaml`` (with
        extension ``.yaml`` or ``.yml``).

        Args:
            fname: File to read.

        Raises:
            NotImplementedError: Wrong file format.

        """
        fpath = Path(fname)
        if not fpath.is_file():
            raise FileNotFoundError(f"File {fpath} not found.")

        if fpath.suffix == ".yaml":
            self._read_yaml(fpath)

        elif fpath.suffix == ".krn":
            self._read_krn(fpath)

        else:
            raise NotImplementedError(
                f"{fpath.suffix} materials file is not supported."
            )

    def _read_krn(self, fname: str | os.PathLike) -> None:
        """Read material `krn` file.

        This function overwrites the current ``domains`` attribute.

        Args:
            fname: File path

        """
        with open(fname) as file:
            lines = file.readlines()
        self.domains = []
        for line in lines:
            line = line.split()
            self.add_domain(
                theta=float(line[0]),
                phi=float(line[1]),
                K1=float(line[2]),
                Ms=(float(line[4]) * u.T).to(
                    u.A / u.m, equivalencies=u.magnetic_flux_field()
                ),
                A=float(line[5]),
            )

    def _read_yaml(self, fname: str | os.PathLike) -> None:
        """Read material `yaml` file.

        This function overwrites the current ``domains`` attribute.

        Args:
            fname: File path

        """
        with open(fname) as file:
            domains = yaml.safe_load(file)
        self.domains = []
        for dom in domains:
            self.add_domain(
                theta=dom["theta"],
                phi=dom["phi"],
                K1=dom["K1"],
                Ms=(dom["Ms"] * u.T).to(
                    u.A / u.m, equivalencies=u.magnetic_flux_field()
                ),
                A=dom["A"],
            )

    def write_krn(self, fname: str | os.PathLike) -> None:
        """Write material parameters in the  ``krn`` file.

        Each domain in :py:attr:`~domains` is written on a single line
        with spaces as separators.

        Args:
            fname: File to write.


        Examples:
            >>> from mammos_mumag.materials import Materials
            >>> mat = Materials()
            >>> mat.write_krn("materials.krn")

        """
        env = Environment(
            loader=PackageLoader("mammos_mumag"),
            autoescape=select_autoescape(),
        )
        template = env.get_template("krn.jinja")
        with open(fname, "w") as file:
            file.write(
                template.render(
                    {
                        "domains": self.domains,
                        "u": u,
                        "eq": u.magnetic_flux_field(),
                    }
                )
            )

    def write_yaml(self, fname: str | os.PathLike) -> None:
        """Write material parameters in the ``yaml`` format.

        The output file will contain a list of all domains, where each domain
        is a ``parameter: value`` map.

        Args:
            fname: Output file path

        """
        domains = [
            {
                "theta": dom.theta.value.tolist(),
                "phi": dom.phi.value.tolist(),
                "K1": dom.K1.value.tolist(),
                "Ms": dom.Ms.q.to(
                    u.T, equivalencies=u.magnetic_flux_field()
                ).value.tolist(),
                "A": dom.A.value.tolist(),
            }
            for dom in self.domains
        ]
        with open(fname, "w") as file:
            yaml.dump(domains, file)
