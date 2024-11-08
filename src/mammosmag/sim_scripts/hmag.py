import sys
from time import time

import esys.escript as e
from esys.escript.linearPDEs import LinearSinglePDE, SolverOptions
from esys.weipa import saveVTK
from esys.escript.pdetools import MaskFromTag

import numpy as np

from materials import Materials
from magnetization import getM
from matrix import gx, gy, gz
from tools import dot, read_Js


def findBoundary(x):
    xe = []
    for i in range(3):
        xe.append(e.sup(x[i]))
    boundaryMask = (
        e.whereZero(x[0] + xe[0])
        + e.whereZero(x[0] - xe[0])
        + e.whereZero(x[1] + xe[1])
        + e.whereZero(x[1] - xe[1])
        + e.whereZero(x[2] + xe[2])
        + e.whereZero(x[2] - xe[2])
    )
    return boundaryMask


def check_domain(x):
    Rinf = e.sup(e.length(x))
    airbox = abs(np.sum(e.convertToNumpy(e.whereZero(e.length(x) - Rinf))) - 8) < 1e-8
    return airbox, Rinf


# IEEE TRANSACTIONS ON MAGNETICS, VOL. 26, NO. 5, SEPTEMBER 1990
def get_T3D(domain, R, Rinf, tags):
    k = e.Tensor(0, e.Function(domain))
    xx = e.Function(domain).getX()
    x0 = xx[0]
    x1 = xx[1]
    x2 = xx[2]
    x = e.length(xx)
    a2 = R * (Rinf - R)
    ax = Rinf / x
    a2x2 = a2 / (x * x)
    k[0, 0] = a2x2 * (1 + (ax * (2 - ax) / (ax - 1) ** 2) * (1 - (x0 / x) ** 2))
    k[1, 1] = a2x2 * (1 + (ax * (2 - ax) / (ax - 1) ** 2) * (1 - (x1 / x) ** 2))
    k[2, 2] = a2x2 * (1 + (ax * (2 - ax) / (ax - 1) ** 2) * (1 - (x2 / x) ** 2))
    k[0, 1] = -a2x2 * ((ax * (2 - ax) / (ax - 1) ** 2) * (x0 / x) * (x1 / x))
    k[0, 2] = -a2x2 * ((ax * (2 - ax) / (ax - 1) ** 2) * (x0 / x) * (x2 / x))
    k[1, 2] = -a2x2 * ((ax * (2 - ax) / (ax - 1) ** 2) * (x1 / x) * (x2 / x))
    k[1, 0] = k[0, 1]
    k[2, 0] = k[0, 2]
    k[2, 1] = k[1, 2]
    for tag in tags[:-1]:
        k.setTaggedValue(tag, e.kronecker(domain))
    return k


class Hmag:
    def __init__(self, Js, volume, tol=1e-8, verbose=0):
        self.cum_time = 0.0
        self.rhs_time = 0.0
        self.u_time = 0.0
        self.grad_time = 0.0
        domain = Js.getDomain()

        x = domain.getX()
        Rinf = e.sup(e.length(x))
        boundaryMask = e.whereZero(e.length(x) - Rinf)
        airbox = abs(np.sum(e.convertToNumpy(boundaryMask)) - 8) < 1e-8
        if airbox:
            boundaryMask = findBoundary(x)
            k = e.kronecker(domain)
        else:
            tags = e.Function(domain).getListOfTags()
            R = e.sup(e.length(x) * MaskFromTag(domain, tags[-2]))
            k = get_T3D(domain, R, Rinf, tags)

        self._poisson = LinearSinglePDE(domain, isComplex=False)
        self._poisson.setSymmetryOn()
        self._poisson.setValue(A=k, q=boundaryMask, r=0.0)
        self._poisson.getSolverOptions().setPreconditioner(SolverOptions.AMG)
        self._poisson.getSolverOptions().setPackage(SolverOptions.TRILINOS)
        self._poisson.getSolverOptions().setTolerance(tol)
        self._gx = gx(Js, volume)
        self._gy = gy(Js, volume)
        self._gz = gz(Js, volume)
        self._Js = Js
        if verbose >= 4:
            self.set_verbose(True)

    def solve_u(self, m):
        T0 = time()
        self._poisson.setValue(X=self._Js * m)
        T1 = time()
        self.rhs_time += T1 - T0
        u = self._poisson.getSolution()
        self.u_time += time() - T1
        return u

    def solve_uh(self, m):
        u = self.solve_u(m)
        return u, -e.grad(u)

    def u2g(self, u):
        T0 = time()
        g = e.Vector(0.0, e.Solution(u.getDomain()))
        g[0] = self._gx.of(u)
        g[1] = self._gy.of(u)
        g[2] = self._gz.of(u)
        self.grad_time += time() - T0
        return g

    def solve_g(self, m):
        T0 = time()
        u = self.solve_u(m)
        g = self.u2g(u)
        self.cum_time += time() - T0
        return g

    def solve_e(self, m):
        return 0.5 * dot(m, self.solve_g(m))

    def set_options(self, package="trilinos", precond="amg"):
        if precond.lower() == "amg":
            self._poisson.getSolverOptions().setPreconditioner(SolverOptions.AMG)
            self._poisson.getSolverOptions().setPackage(SolverOptions.TRILINOS)
        if precond.lower() == "jacobi":
            self._poisson.getSolverOptions().setPreconditioner(SolverOptions.JACOBI)
        if precond.lower() == "gauss_seidel":
            self._poisson.getSolverOptions().setPreconditioner(
                SolverOptions.GAUSS_SEIDEL
            )
        if precond.lower() == "ilu0":
            self._poisson.getSolverOptions().setPreconditioner(SolverOptions.ILU0)
        if precond.lower() == "rilu":
            self._poisson.getSolverOptions().setPreconditioner(SolverOptions.RILU)
        if package.lower() == "trilinos":
            self._poisson.getSolverOptions().setPackage(SolverOptions.TRILINOS)
        if package.lower() == "paso":
            self._poisson.getSolverOptions().setPackage(SolverOptions.PASO)

    def set_verbose(self, verbose=False):
        if verbose:
            self._poisson.getSolverOptions().setVerbosityOn()
        else:
            self._poisson.getSolverOptions().setVerbosityOff()

    def write_summary(self):
        print("Hmag solver options :")
        print(self._poisson.getSolverOptions().getSummary())

    def getDiagnostics(self):
        return int(
            self._poisson.getSolverOptions().getDiagnostics("cum_num_iter")
        ), self._poisson.getSolverOptions().getDiagnostics("cum_time")


if __name__ == "__main__":
    try:
        name = sys.argv[1]
    except IndexError:
        sys.exit("usage run-escript hmag.py modelname")

    # Hmag solver
    materials = Materials(name)
    hmag = Hmag(materials.Js, materials.volume, 1e-12, 1)

    # scalar potential, field, and energy
    m = getM(e.wherePositive(materials.meas), [0.0, 0.0, 1.0])
    u, h = hmag.solve_uh(m)
    emag = e.integrate(-0.5 * e.inner(h, materials.Js * m)) / materials.volume

    # energy
    Js = read_Js(name)

    print("energy from field   ", emag)
    print("       from gradient", hmag.solve_e(m))
    print("       analytic     ", Js * Js / 6)

    # field at nodes
    g = hmag.solve_g(m)

    meas = e.whereZero(materials.meas) + materials.meas
    h_at_nodes = e.Vector(0.0, e.Solution(materials.getDomain()))
    h_at_nodes[0] = -materials.volume * (g[0] / meas)
    h_at_nodes[1] = -materials.volume * (g[1] / meas)
    h_at_nodes[2] = -materials.volume * (g[2] / meas)

    saveVTK(
        name + ".hmag", tags=materials.get_tags(), m=m, U=u, h=h, h_nodes=h_at_nodes
    )
