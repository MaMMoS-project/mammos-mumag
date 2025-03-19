"""Parameters class."""

import math
import configparser
from pydantic.dataclasses import dataclass
import yaml

from .tools import check_path

from jinja2 import Environment, PackageLoader, select_autoescape


@dataclass
class Parameters:
    r"""Class storing simulation parameters.

    :param size: Size of the mesh, defaults to 1e-9.
    :type size: float
    :param scale: Scale of the mesh, defaults to 0.
    :type scale: float
    :param state: Name of the state. Scripts recognize the strings `flower`, `vortex`,
        and `twisted`. Other strings are interpreted as the default case. Defaults to
        "mxyz".
    :type state: str
    :param mx: :math:`x` component of the magnetization field :math:`\mathbf{m}`.
        Defaults to 0.
    :type mx: float
    :param my: :math:`y` component of the magnetization field :math:`\mathbf{m}`.
        Defaults to 0.
    :type my: float
    :param mz: :math:`z` component of the magnetization field :math:`\mathbf{m}`.
        Defaults to 0.
    :type mz: float
    :param hmag_on: 1 or 0 indicating whether the external field is on (1) or off (0).
        Defaults to 1.
    :type hmag_on: int
    :param hstart: Initial external field. Defaults to 0.
    :type hstart: float
    :param hfinal: Final external field. Defaults to 0.
    :type hfinal: float
    :param hstep: External field step. Defaults to 0.
    :type hstep: float
    :param hx: :math:`x` component of the external field :math:`\mathbf{h}`.
        Defaults to 0.
    :type hx: float
    :param hy: :math:`y` component of the external field :math:`\mathbf{h}`.
        Defaults to 0.
    :type hy: float
    :param hz: :math:`z` component of the external field :math:`\mathbf{h}`.
        Defaults to 0.
    :type hz: float
    :param mstep: TODO. Defaults to 1.0.
    :type mstep: float
    :param mfinal: TODO. Defaults to -0.8.
    :type mfinal: float
    :param iter_max: Max number of iterations of optimizer.
        TODO NOT USED AT THE MOMENT. Defaults to 1000.
    :type iter_max: int
    :param precond_iter: conjugate gradient iterations for inverse
        Hessian approximation. Defaults to 10.
    :type precond_iter: int
    :param tol_fun: Tolerance of the total energy. Defaults to 1e-10.
    :type tol_fun: float
    :param tol_hmag_factor: Factor defining the tolerance for the
        magnetostatic scalar potential. Defaults to 1.
    :type tol_hmag_factor: float
    :param tol_u: TODO. Defaults to 1e-10.
    :type tol_u: float
    :param verbose: verbosity. Defaults to 0
    :type verbose: int
    """

    size: float = 1.0e-09
    scale: float = 0.0
    state: str = "mxyz"
    mx: float = 0.0
    my: float = 0.0
    mz: float = 0.0
    hmag_on: int = 1
    hstart: float = 0.0
    hfinal: float = 0.0
    hstep: float = 0.0
    hx: float = 0.0
    hy: float = 0.0
    hz: float = 0.0
    mstep: float = 1.0
    mfinal: float = -0.8
    iter_max: int = 1000
    precond_iter: int = 10
    tol_fun: float = 1e-10
    tol_hmag_factor: float = 1.0
    tol_u: float = 1e-10
    verbose: int = 0

    @property
    def m(self):
        """Return list m given the components mx, my, mz."""
        return normalize([self.mx, self.my, self.mz])

    @m.setter
    def m(self, value):
        """Assign mx, my, mz given m."""
        self.mx = value[0]
        self.my = value[1]
        self.mz = value[2]

    @property
    def h(self):
        """Return list h given the components hx, hy, hz."""
        return normalize([self.hx, self.hy, self.hz])

    @h.setter
    def h(self, value):
        """Assign hx, hy, hz given h."""
        self.hx = value[0]
        self.hy = value[1]
        self.hz = value[2]

    def read(self, fname):
        """Read parameter file in `yaml` or `p2` format.

        Simulation parameters are read and stored.

        :param fname: File path
        :type fname: str or pathlib.Path
        :raises NotImplementedError: Wrong file format.
        """
        fpath = check_path(fname)

        if fpath.suffix == ".yaml":
            with open(fpath, "r") as file:
                pars = yaml.safe_load(file)

        elif fpath.suffix == ".p2":
            pars = configparser.ConfigParser()
            pars.read(fpath)

        else:
            raise NotImplementedError("Wrong file format.")

        mesh = pars["mesh"]
        if "size" in mesh:
            self.size = float(mesh["size"])
        if "scale" in mesh:
            self.scale = float(mesh["scale"])

        initial_state = pars["initial state"]
        if "state" in initial_state:
            self.state = str(initial_state["state"])
        self.mx = float(initial_state["mx"])
        self.my = float(initial_state["my"])
        self.mz = float(initial_state["mz"])

        field = pars["field"]
        if "hmag_on" in field:
            self.hmag_on = int(field["hmag_on"])
        self.hstart = float(field["hstart"])
        self.hfinal = float(field["hfinal"])
        self.hstep = float(field["hstep"])
        self.hx = float(field["hx"])
        self.hy = float(field["hy"])
        self.hz = float(field["hz"])
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

    def write_p2(self, fname):
        """Write parameter `p2` file.

        :param fname: File path
        :type fname: str or pathlib.Path
        """
        env = Environment(
            loader=PackageLoader("mammos_mmag"),
            autoescape=select_autoescape(),
        )
        template = env.get_template("p2.jinja")
        with open(fname, "w") as file:
            file.write(template.render(self.__dict__))

    def write_yaml(self, fname):
        """Write parameter `yaml` file.

        :param fname: File path
        :type fname: str or pathlib.Path
        """
        parameters_dict = {
            "mesh": {
                "size": self.size,
                "scale": self.scale,
            },
            "initial state": {
                "state": self.state,
                "mx": self.mx,
                "my": self.my,
                "mz": self.mz,
            },
            "field": {
                "hmag_on": self.hmag_on,
                "hstart": self.hstart,
                "hfinal": self.hfinal,
                "hstep": self.hstep,
                "hx": self.hx,
                "hy": self.hy,
                "hz": self.hz,
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


def normalize(v):
    """Normalize list.

    :param v: list to normalize
    :type v: list
    """
    s = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if s <= 1.0e-13:
        return v
    else:
        return [v[0] / s, v[1] / s, v[2] / s]
