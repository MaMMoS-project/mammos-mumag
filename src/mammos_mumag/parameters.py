"""Parameters class."""

import configparser
import math
import pathlib

import yaml
from jinja2 import Environment, PackageLoader, select_autoescape
from pydantic import Field
from pydantic.dataclasses import dataclass

from mammos_mumag.tools import check_path


@dataclass
class Parameters:
    r"""Class storing simulation parameters.

    Args:
        size: Size of the mesh.
        scale: Scale of the mesh.
        state: Name of the state. Scripts recognize the strings `flower`, `vortex`,
            `twisted`, and `random`. Other strings are interpreted as the default case.
        m_vect: Magnetization field :math:`\mathbf{m}`.
        hmag_on: 1 or 0 indicating whether the external field is on (1) or off (0).
        hstart: Initial external field.
        hfinal: Final external field.
        hstep: External field step.
        h_vect: External field vector :math:`\mathbf{h}`.
        mstep: TODO
        mfinal: TODO
        iter_max: Max number of iterations of optimizer. TODO NOT USED AT THE MOMENT.
        precond_iter: conjugate gradient iterations for inverse Hessian approximation.
        tol_fun: Tolerance of the total energy.
        tol_hmag_factor: Factor defining the tolerance for the magnetostatic scalar
            potential.
        tol_u: TODO
        verbose: verbosity
        filepath: TODO
    """

    size: float = 1.0e-09
    scale: float = 0.0
    state: str = Field(default_factory=lambda: "")
    m_vect: list[float] = Field(
        default_factory=lambda: [0, 0, 0],
        min_length=3,
        max_length=3,
    )
    hmag_on: int = 1
    hstart: float = 0.0
    hfinal: float = 0.0
    hstep: float = 0.0
    h_vect: list[float] = Field(
        default_factory=lambda: [0, 0, 0],
        min_length=3,
        max_length=3,
    )
    mstep: float = 1.0
    mfinal: float = -0.8
    iter_max: int = 1000
    precond_iter: int = 10
    tol_fun: float = 1e-10
    tol_hmag_factor: float = 1.0
    tol_u: float = 1e-10
    verbose: int = 0
    filepath: pathlib.Path | None = Field(default=None, repr=False)

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

    @m.setter
    def m(self, value: list[float]) -> None:
        self.m_vect = _normalize(value)

    @property
    def h(self) -> list[float]:
        """Direction of the external field."""
        return _normalize(self.h_vect)

    @h.setter
    def h(self, value: list[float]) -> None:
        self.h_vect = _normalize(value)

    def read(self, fname: str | pathlib.Path) -> None:
        """Read parameter file in `yaml` or `p2` format.

        Simulation parameters are read and stored.

        Args:
            fname: File path

        Raises:
            NotImplementedError: Wrong file format.

        """
        fpath = check_path(fname)

        if fpath.suffix == ".yaml":
            with open(fpath) as file:
                pars = yaml.safe_load(file)

        elif fpath.suffix == ".p2":
            pars = configparser.ConfigParser()
            pars.read(fpath)

        else:
            raise NotImplementedError(
                f"{fpath.suffix} materials file is not supported."
            )

        mesh = pars["mesh"]
        if "size" in mesh:
            self.size = float(mesh["size"])
        if "scale" in mesh:
            self.scale = float(mesh["scale"])

        initial_state = pars["initial state"]
        if "state" in initial_state:
            self.state = str(initial_state["state"])
        self.m = [
            float(initial_state["mx"]),
            float(initial_state["my"]),
            float(initial_state["mz"]),
        ]

        field = pars["field"]
        if "hmag_on" in field:
            self.hmag_on = int(field["hmag_on"])
        self.hstart = float(field["hstart"])
        self.hfinal = float(field["hfinal"])
        self.hstep = float(field["hstep"])
        self.h = [
            float(field["hx"]),
            float(field["hy"]),
            float(field["hz"]),
        ]
        if "mstep" in field:
            self.mstep = float(field["mstep"])
        if "mfinal" in field:
            self.mfinal = float(field["mfinal"])

        minimizer = pars["minimizer"]
        if "iter_max" in minimizer:
            self.iter_max = int(minimizer["iter_max"])
        if "precond_iter" in minimizer:
            self.precond_iter = int(minimizer["precond_iter"])
        if "tol_fun" in minimizer:
            self.tol_fun = float(minimizer["tol_fun"])
        if "tol_hmag_factor" in minimizer:
            self.tol_hmag_factor = float(minimizer["tol_hmag_factor"])
        self.tol_u = self.tol_fun * self.tol_hmag_factor
        if "truncation" in minimizer:
            self.truncation = int(minimizer["truncation"])
        if "verbose" in minimizer:
            self.verbose = int(minimizer["verbose"])

    def write_p2(self, fname: str | pathlib.Path) -> None:
        """Write parameter `p2` file.

        Args:
            fname: File path

        Examples:
            >>> from mammos_mumag.parameters import Parameters
            >>> par = Parameters()
            >>> par.write_p2("parameters.p2")

        """
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
        parameters_dict = {
            "mesh": {
                "size": self.size,
                "scale": self.scale,
            },
            "initial state": {
                "state": self.state,
                "mx": self.m[0],
                "my": self.m[1],
                "mz": self.m[2],
            },
            "field": {
                "hmag_on": self.hmag_on,
                "hstart": self.hstart,
                "hfinal": self.hfinal,
                "hstep": self.hstep,
                "hx": self.h[0],
                "hy": self.h[1],
                "hz": self.h[2],
            },
            "minimizer": {
                "iter_max": self.iter_max,
                "tol_fun": self.tol_fun,
                "tol_hmag_factor": self.tol_hmag_factor,
                "precond_iter": self.precond_iter,
                "verbose": self.verbose,
            },
        }
        with open(fname, "w") as file:
            yaml.dump(parameters_dict, file)


def _normalize(v: list[float]) -> list[float]:
    """Normalize list.

    Args:
        v: list to normalize

    """
    s = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if s <= 1.0e-13:
        return v
    else:
        return [v[0] / s, v[1] / s, v[2] / s]
