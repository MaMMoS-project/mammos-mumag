"""Materials class."""

from jinja2 import Environment, PackageLoader, select_autoescape
import numbers
import pathlib
from pydantic import ConfigDict, Field, field_validator
from pydantic.dataclasses import dataclass
import yaml

import mammos_entity as me
import mammos_units as u

from mammos_mumag.tools import check_path


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class MaterialDomain:
    r"""Uniform material domain.

    It collects material parameters, constant in a certain domain.

    :param theta: Angle of the magnetocrystalline anisotropy axis
        from the :math:`z`-direction in radians. Defaults to 0.
    :type theta: float
    :param phi: Angle of the magnetocrystalline anisotropy axis
        from the :math:`x`-direction in radians. Defaults to 0.
    :type phi: float
    :param K1: First magnetocrystalline anisotropy constant in
        :math:`\mathrm{J}/\mathrm{m}^3`. Defaults to 0.
    :type K1: float
    :param K2: Second magnetocrystalline anisotropy constant in
        :math:`\mathrm{J}/\mathrm{m}^3`. Defaults to 0.
    :type K2: float
    :param Ms: Spontaneous magnetisation in :math:`\mathrm{A}/\mathrm{m}`.
        Defaults to 0.
    :type Ms: float
    :param A: Exchange stiffness constant in :math:`\mathrm{J}/\mathrm{m}`.
        Defaults to 0.
    :type A: float
    """

    theta: float = 0.0
    phi: float = 0.0
    K1: me.Entity = me.Ku(0.0, unit=u.J / u.m**3)
    K2: me.Entity = me.Ku(0.0, unit=u.J / u.m**3)
    Ms: me.Entity = me.Ms(0.0, unit=u.A / u.m)
    A: me.Entity = me.A(0.0, unit=u.J / u.m)

    @field_validator("K1", mode="before")
    @classmethod
    def convert_K1(cls, K1):
        """Convert K1."""
        if isinstance(K1, (numbers.Real, u.Quantity)):
            K1 = me.Ku(K1, unit=u.J / u.m**3)
        return K1

    @field_validator("K2", mode="before")
    @classmethod
    def convert_K2(cls, K2):
        """Convert K2."""
        if isinstance(K2, float) or isinstance(K2, int) or isinstance(K2, u.Quantity):
            K2 = me.Ku(K2, unit=u.J / u.m**3)
        return K2

    @field_validator("A", mode="before")
    @classmethod
    def convert_A(cls, A):
        """Convert A."""
        if isinstance(A, float) or isinstance(A, int) or isinstance(A, u.Quantity):
            A = me.A(A, unit=u.J / u.m)
        return A

    @field_validator("Ms", mode="before")
    @classmethod
    def convert_Ms(cls, Ms):
        """Convert Ms."""
        if isinstance(Ms, float) or isinstance(Ms, int) or isinstance(Ms, u.Quantity):
            Ms = me.Ms(Ms, unit=u.A / u.m)
        return Ms


@dataclass
class Materials:
    """This class stores, reads, and writes material parameters.

    :param domains: list of domains. Each domain is a MaterialDomain
        class of material parameters, constant in each region.
        It defaults to an empty list.
    :type domains: list[MaterialDomain]
    :param filepath: material file path
    :type filepath: pathlib.Path
    """

    domains: list[MaterialDomain] = Field(default_factory=list)
    filepath: pathlib.Path | None = Field(default=None, repr=False)

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

        :param A: Exchange stiffness constant in :math:`\mathrm{J}/\mathrm{m}`.
        :type A: float
        :param Ms: Spontaneous magnetisation in :math:`\mathrm{T}`.
        :type Ms: float
        :param K1: First magnetocrystalline anisotropy constant in
            :math:`\mathrm{J}/\mathrm{m}^3`.
        :type K1: float
        :param K2: Second magnetocrystalline anisotropy constant in
            :math:`\mathrm{J}/\mathrm{m}^3`.
        :type K2: float
        :param phi: Angle of the magnetocrystalline anisotropy axis
            from the :math:`x`-direction in radians.
        :type phi: float
        :param theta: Angle of the magnetocrystalline anisotropy axis
            from the :math:`z`-direction in radians.
        :type theta: float
        """
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
        :py:attr:`~materials.Materials.domains` attribute.

        Currently accepted formats: ``krn`` and ``yaml``.

        :param fname: File name.
        :type fname: str or pathlib.Path
        :raises NotImplementedError: Wrong file format.
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

        :param fname: File path
        :type fname: str or pathlib.Path
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

        :param fname: File path
        :type fname: str or pathlib.Path
        """
        domains = [
            {
                "theta": dom.theta,
                "phi": dom.phi,
                "K1": dom.K1.value.tolist(),
                "K2": dom.K2.value.tolist(),
                "Ms": dom.Ms.to(
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

    :param fname: File path
    :type fname: str or pathlib.Path
    :return: Domains as list of dictionaries, with each dictionary defining
        the material constant in a specific region.
    :rtype: list[dict]
    """
    with open(fname, "r") as file:
        lines = file.readlines()
    lines = [line.split() for line in lines]
    return [
        MaterialDomain(
            theta=float(line[0]),
            phi=float(line[1]),
            K1=me.Ku(float(line[2]), unit=u.J / u.m**3),
            K2=me.Ku(float(line[3]), unit=u.J / u.m**3),
            Ms=me.Ms(
                (float(line[4]) * u.T).to(
                    u.A / u.m, equivalencies=u.magnetic_flux_field()
                )
            ),
            A=me.A(float(line[5]), unit=u.J / u.m),
        )
        for line in lines
    ]


def read_yaml(fname: str | pathlib.Path) -> list[MaterialDomain]:
    """Read material `yaml` file.

    :param fname: File path
    :type fname: str or pathlib.Path
    :return: Domains as list of dictionaries, with each dictionary defining
        the material constant in a specific region.
    :rtype: list[dict]
    """
    with open(fname, "r") as file:
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
