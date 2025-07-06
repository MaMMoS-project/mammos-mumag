"""Materials class."""

import numbers
import pathlib
from typing import Any

import mammos_entity as me
import mammos_units as u
import yaml
from jinja2 import Environment, PackageLoader, select_autoescape
from pydantic import ConfigDict, Field, field_validator
from pydantic.dataclasses import dataclass

from mammos_mumag.tools import check_path


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class MaterialDomain:
    """Uniform material domain.

    It collects material parameters, constant in a certain domain.
    """

    theta: float = 0.0
    """Angle of the magnetocrystalline anisotropy axis from the :math:`z`-direction in
    radians."""
    phi: float = 0.0
    """Angle of the magnetocrystalline anisotropy axis from the :math:`x`-direction in
    radians."""
    K1: me.Entity = Field(default_factory=me.Ku)
    r"""First magnetocrystalline anisotropy constant in
    :math:`\mathrm{J}/\mathrm{m}^3`."""
    K2: me.Entity = Field(default_factory=me.Ku)
    r"""Second magnetocrystalline anisotropy constant in
    :math:`\mathrm{J}/\mathrm{m}^3`."""
    Ms: me.Entity = Field(default_factory=me.Ms)
    r"""Spontaneous magnetisation in :math:`\mathrm{A}/\mathrm{m}`."""
    A: me.Entity = Field(default_factory=me.A)
    r"""Exchange stiffness constant in :math:`\mathrm{J}/\mathrm{m}`."""

    @field_validator("K1", mode="before")
    @classmethod
    def _convert_K1(cls, K1: Any) -> Any:
        """Convert number or Quantity to Entity."""
        if isinstance(K1, numbers.Real | u.Quantity):
            K1 = me.Ku(K1, unit=u.J / u.m**3)
        return K1

    @field_validator("K2", mode="before")
    @classmethod
    def _convert_K2(cls, K2: Any) -> Any:
        """Convert number or Quantity to Entity."""
        if isinstance(K2, float | int | u.Quantity):
            K2 = me.Ku(K2, unit=u.J / u.m**3)
        return K2

    @field_validator("A", mode="before")
    @classmethod
    def _convert_A(cls, A: Any) -> Any:
        """Convert number or Quantity to Entity."""
        if isinstance(A, float | int | u.Quantity):
            A = me.A(A, unit=u.J / u.m)
        return A

    @field_validator("Ms", mode="before")
    @classmethod
    def _convert_Ms(cls, Ms: Any) -> Any:
        """Convert number or Quantity to Entity."""
        if isinstance(Ms, float | int | u.Quantity):
            Ms = me.Ms(Ms, unit=u.A / u.m)
        return Ms


@dataclass
class Materials:
    """This class stores, reads, and writes material parameters."""

    domains: list[MaterialDomain] = Field(default_factory=list)
    """Each domain is a MaterialDomain class of material parameters, constant in each
    region."""
    filepath: pathlib.Path | None = Field(default=None, repr=False)
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
        self, A: float, Ms: float, K1: float, K2: float, phi: float, theta: float
    ) -> None:
        r"""Append domain with specified parameters.

        Args:
            A: Exchange stiffness constant in :math:`\mathrm{J}/\mathrm{m}`.
            Ms: Spontaneous magnetisation in :math:`\mathrm{T}`.
            K1: First magnetocrystalline anisotropy constant in
                :math:`\mathrm{J}/\mathrm{m}^3`.
            K2: Second magnetocrystalline anisotropy constant in
                :math:`\mathrm{J}/\mathrm{m}^3`.
            phi: Angle of the magnetocrystalline anisotropy axis
                from the :math:`x`-direction in radians.
            theta: Angle of the magnetocrystalline anisotropy axis
                from the :math:`z`-direction in radians.

        Examples:
            >>> from mammos_mumag.materials import Materials
            >>> mat = Materials()
            >>> mat.add_domain(A=1, Ms=2, K1=3, K2=0, phi=0, theta=0)
            >>> mat
            Materials(domains=[MaterialDomain(theta=..., phi=..., K1=..., K2=..., Ms=..., A=...)])

        """  # noqa: E501
        dom = MaterialDomain(
            theta=theta,
            phi=phi,
            K1=K1,
            K2=K2,
            Ms=Ms,
            A=A,
        )
        self.domains.append(dom)

    def read(self, fname: str | pathlib.Path) -> None:
        """Read materials file.

        This function overwrites the current
        :py:attr:`~mammos_mumag.materials.Materials.domains` attribute.

        Currently accepted formats: ``krn`` and ``yaml``.

        Args:
            fname: File name.

        Raises:
            NotImplementedError: Wrong file format.

        """
        fpath = check_path(fname)

        if fpath.suffix == ".yaml":
            self.domains = read_yaml(fpath)

        elif fpath.suffix == ".krn":
            self.domains = read_krn(fpath)

        else:
            raise NotImplementedError(
                f"{fpath.suffix} materials file is not supported."
            )

    def write_krn(self, fname: str | pathlib.Path) -> None:
        """Write material `krn` file.

        Each domain in :py:attr:`~domains` is written on a single line
        with spaces as separators.

        Args:
            fname: File path

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

    def write_yaml(self, fname: str | pathlib.Path) -> None:
        """Write material `yaml` file.

        Args:
            fname: File path

        """
        domains = [
            {
                "theta": dom.theta,
                "phi": dom.phi,
                "K1": dom.K1.value.tolist(),
                "K2": dom.K2.value.tolist(),
                "Ms": dom.Ms.q.to(
                    u.T, equivalencies=u.magnetic_flux_field()
                ).value.tolist(),
                "A": dom.A.value.tolist(),
            }
            for dom in self.domains
        ]
        with open(fname, "w") as file:
            yaml.dump(domains, file)


def read_krn(fname: str | pathlib.Path) -> list[MaterialDomain]:
    """Read material `krn` file and return as list of dictionaries.

    Args:
        fname: File path

    Returns:
        Domains as list of dictionaries, with each dictionary defining
        the material constant in a specific region.

    """
    with open(fname) as file:
        lines = file.readlines()
    lines = [line.split() for line in lines]
    return [
        MaterialDomain(
            theta=float(line[0]),
            phi=float(line[1]),
            K1=me.Ku(float(line[2]), unit="J/m3"),
            K2=me.Ku(float(line[3]), unit="J/m3"),
            Ms=me.Ms(
                (float(line[4]) * u.T).to(
                    u.A / u.m, equivalencies=u.magnetic_flux_field()
                ),
                unit="A/m",
            ),
            A=me.A(float(line[5]), unit="J/m"),
        )
        for line in lines
    ]


def read_yaml(fname: str | pathlib.Path) -> list[MaterialDomain]:
    """Read material `yaml` file.

    Args:
        fname: File path

    Returns:
        Domains as list of dictionaries, with each dictionary defining
        the material constant in a specific region.

    """
    with open(fname) as file:
        domains = yaml.safe_load(file)
    return [
        MaterialDomain(
            theta=float(dom["theta"]),
            phi=float(dom["phi"]),
            K1=me.Ku(float(dom["K1"]), unit=u.J / u.m**3),
            K2=me.Ku(float(dom["K2"]), unit=u.J / u.m**3),
            Ms=me.Ms(
                (float(dom["Ms"]) * u.T).to(
                    u.A / u.m, equivalencies=u.magnetic_flux_field()
                )
            ),
            A=me.A(float(dom["A"]), unit=u.J / u.m),
        )
        for dom in domains
    ]
