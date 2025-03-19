"""Materials class."""

import yaml

from math import pi

from .tools import check_path
from jinja2 import Environment, PackageLoader, select_autoescape


class Materials:
    """This class stores, reads, and writes material parameters.

    :param domains: list of domains. Each domain is a dictionary of material
        parameters, constant in each region.
    :type domains: list[dict]
    """

    def __init__(self):
        """Initialize empty materials class."""
        self.domains = []

    def new_domain(self, A, Js, K1, K2, phi, theta):
        r"""Append domain with specified parameters.

        :param A: Exchange stiffness constant in :math:`\mathrm{J}/\mathrm{m}`.
        :type A: float
        :param Js: Spontaneous magnetic polarisation in :math:`\mathrm{T}`.
        :type Js: float
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
        dom = {
            "theta": theta,
            "phi": phi,
            "K1": K1,
            "K2": K2,
            "Js": Js,
            "A": A,
        }
        self.domains.append(dom)

    def read(self, fname):
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
            raise NotImplementedError(f"{fpath.suffix} materials file is not supported.")

    def write_krn(self, fname):
        """Write material `krn` file.

        Each domain in :py:attr:`~domains` is written on a single line
        with spaces as separators.

        :param fname: File path
        :type fname: str or pathlib.Path
        """
        env = Environment(
            loader=PackageLoader("mammos_mmag"),
            autoescape=select_autoescape(),
        )
        template = env.get_template("krn.jinja")
        with open(fname, "w") as file:
            file.write(template.render({"domains": self.domains, "const": 4e-7 * pi}))

    def write_yaml(self, fname):
        """Write material `yaml` file.

        :param fname: File path
        :type fname: str or pathlib.Path
        """
        domains = [
            {
                "theta": dom["theta"],
                "phi": dom["phi"],
                "K1": dom["K1"] / (4e-7 * pi),
                "K2": dom["K2"] / (4e-7 * pi),
                "Js": dom["Js"],
                "A": dom["A"] / (4e-7 * pi),
            }
            for dom in self.domains
        ]
        with open(fname, "w") as file:
            yaml.dump(domains, file)


def read_krn(fname):
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
        {
            "theta": float(line[0]),
            "phi": float(line[1]),
            "K1": 4e-7 * pi * float(line[2]),
            "K2": 4e-7 * pi * float(line[3]),
            "Js": float(line[4]),
            "A": 4e-7 * pi * float(line[5]),
        }
        for line in lines
    ]


def read_yaml(fname):
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
        {
            "theta": float(dom["theta"]),
            "phi": float(dom["phi"]),
            "K1": 4e-7 * pi * float(dom["K1"]),
            "K2": 4e-7 * pi * float(dom["K2"]),
            "Js": float(dom["Js"]),
            "A": 4e-7 * pi * float(dom["A"]),
        }
        for dom in domains
    ]
