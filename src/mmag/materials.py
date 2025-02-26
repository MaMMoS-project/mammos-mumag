"""Materials class."""

import yaml

from math import pi

from .tools import check_path
from jinja2 import Environment, PackageLoader, select_autoescape


class Materials:
    """Materials class."""

    def __init__(self):
        """Initialize Materials class."""
        self.domains = []

    def new_domain(self, A, Js, K1, phi, theta, K2=0.0):
        """Append domain with specified parameters.

        :param A: Exchange stiffness constant in J/m.
        :type A: float
        :param Js: Spontaneous magnetic polarisation in T.
        :type Js: float
        :param K1: Magnetocrystalline anisotropy constant in J/m^3.
        :type K1: float
        :param phi: angle of the magnetocrystalline anisotropy axis
            from the x-direction in radians.
        :type phi: float
        :param theta: Angle of the magnetocrystalline anisotropy axis
            from the z-direction in radians.
        :type theta: float
        :param K2: Not sure, defaults to 0.0.
        :type K2: float, optional
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

    def read_krn(self, fname):
        """Read material `krn` file.

        Material parameters are read and stored.

        :param fname: File path
        :type fname: str or pathlib.Path
        """
        check_path(fname)
        with open(fname, "r") as file:
            lines = file.readlines()
        lines = [line.split() for line in lines]
        self.domains = [
            {
                "theta": float(line[0]),
                "phi": float(line[1]),
                "K1": 4e-7 * pi * float(line[2]),
                "K2": float(line[3]),  # this might be wrong
                "Js": float(line[4]),
                "A": 4e-7 * pi * float(line[5]),
            }
            for line in lines
        ]

    def read_yaml(self, fname):
        """Read material `yaml` file.

        Material parameters are read and stored.

        :param fname: File path
        :type fname: str or pathlib.Path
        """
        check_path(fname)
        with open(fname, "r") as file:
            domains = yaml.safe_load(file)
        self.domains = [
            {
                "theta": float(dom["theta"]),
                "phi": float(dom["phi"]),
                "K1": 4e-7 * pi * float(dom["K1"]),
                "K2": float(dom["K2"]),
                "Js": float(dom["Js"]),
                "A": 4e-7 * pi * float(dom["A"]),
            }
            for dom in domains
        ]

    def write_krn(self, fname):
        """Write material `krn` file.

        :param fname: File path
        :type fname: str or pathlib.Path
        """
        env = Environment(
            loader=PackageLoader("mmag"),
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
                "K2": dom["K2"],
                "Js": dom["Js"],
                "A": dom["A"] / (4e-7 * pi),
            }
            for dom in self.domains
        ]
        with open(fname, "w") as file:
            yaml.dump(domains, file)
