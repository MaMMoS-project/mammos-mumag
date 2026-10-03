"""Materials class."""

from __future__ import annotations

import os
import pathlib
from textwrap import indent
from typing import TYPE_CHECKING

import mammos_entity as me
import mammos_units as u
import yaml
from jinja2 import Environment, PackageLoader, select_autoescape

if TYPE_CHECKING:
    import numbers

    import mammos_entity
    import mammos_units


class MaterialDomain(me.EntityCollection):
    """Uniform material domain.

    It collects material parameters, constant in a certain domain.
    """

    def __init__(
        self,
        theta: mammos_entity.Entity | mammos_units.Quantity | numbers.Real = 0,
        phi: mammos_entity.Entity | mammos_units.Quantity | numbers.Real = 0,
        K1: mammos_entity.Entity | mammos_units.Quantity | numbers.Real = 0,
        Ms: mammos_entity.Entity | mammos_units.Quantity | numbers.Real = 0,
        A: mammos_entity.Entity | mammos_units.Quantity | numbers.Real = 0,
        description: str = "",
    ):
        r"""Create a new MaterialDomain instance.

        Args:
            theta: :entity:`Angle` of the magnetocrystalline anisotropy axis from the
                :math:`z`-direction. Interpreted in radians if passed without unit.
            phi: :entity:`Angle` of the magnetocrystalline anisotropy axis from the
                :math:`x`-direction. Interpreted in radians if passed without unit.
            K1: First uniaxial magnetocrystalline anisotropy constant as
                :entity:`UniaxialAnisotropyConstant`. Interpreted in J/m³ if
                passed without unit.
            Ms: :entity:`SpontaneousMagnetization`. Interpreted in A/m if passed
                without unit.
            A: :entity:`ExchangeStiffnessConstant`. Interpreted in J/m if passed
                without unit.
            description: Description of the domain.
        """
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            if isinstance(theta, u.Quantity):
                theta = theta.to("rad").value
            if isinstance(phi, u.Quantity):
                phi = phi.to("rad").value
        theta = me._entity.from_compatible("Angle", "", theta=theta, enforce_unit=True)
        phi = me._entity.from_compatible("Angle", "", phi=phi, enforce_unit=True)
        Ms = me._entity.from_compatible(
            "SpontaneousMagnetization", "A/m", Ms=Ms, enforce_unit=True
        )
        K1 = me._entity.from_compatible(
            "UniaxialAnisotropyConstant", "J/m3", K1=K1, enforce_unit=True
        )
        A = me._entity.from_compatible(
            "ExchangeStiffnessConstant", "J/m", A=A, enforce_unit=True
        )
        super().__init__(
            description=description, theta=theta, phi=phi, K1=K1, Ms=Ms, A=A
        )


class Materials:
    """This class stores, reads, and writes material parameters."""

    def __init__(
        self,
        domains: list[MaterialDomain] | None = None,
        filepath: os.PathLike | str = "",
    ):
        """Create a new Materials instance.

        Args:
            domains: Each domain is a MaterialDomain class of material parameters,
                constant in each region.
            filepath: Material file path. If the materials is initialized with a
                non-empty ``filepath`` attribute, the materials file will be read
                automatically and the ``domains`` attribute will be overwritten.

        Raises:
            ValueError: Input `domains` is not a list.

        """
        if domains is None:
            self.domains = []
        elif not (isinstance(domains, list)):
            raise ValueError(
                f"Input `domains` should be a list. Given object: {domains}."
            )
        else:
            self.domains = []
            for dom in domains:
                if isinstance(dom, MaterialDomain):
                    self.domains.append(dom)
                elif isinstance(dom, dict):
                    self.domains.append(MaterialDomain(**dom))
        if filepath != "":
            self.read(filepath)

    def __repr__(self):
        """Repr string."""
        out = "Materials(\n"
        if len(self.domains) > 0:
            out += indent("domains=[\n", " " * 4)
            for dd in self.domains:
                out += indent(f"{dd},\n", " " * 8)
            out += indent("]\n", " " * 4)
        out += ")"
        return out

    def add_domain(
        self, A: float, Ms: float, K1: float, phi: float, theta: float
    ) -> None:
        r"""Append domain with specified parameters.

        Args:
            A: Exchange stiffness constant in :math:`\mathrm{J}/\mathrm{m}`.
            Ms: Spontaneous magnetisation in :math:`\mathrm{A}/\mathrm{m}`.
            K1: First magnetocrystalline anisotropy constant in
                :math:`\mathrm{J}/\mathrm{m}^3`.
            phi: Angle of the magnetocrystalline anisotropy axis
                from the :math:`x`-direction in radians.
            theta: Angle of the magnetocrystalline anisotropy axis
                from the :math:`z`-direction in radians.

        Examples:
            >>> from mammos_mumag.materials import Materials
            >>> mat = Materials()
            >>> mat.add_domain(A=1, Ms=2, K1=3, phi=0, theta=0)
            >>> mat
            Materials(...)

        """
        dom = MaterialDomain(
            theta=theta,
            phi=phi,
            K1=K1,
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
        fpath = pathlib.Path(fname)
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

    def _read_krn(self, fname: str | pathlib.Path) -> None:
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

    def _read_yaml(self, fname: str | pathlib.Path) -> None:
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
