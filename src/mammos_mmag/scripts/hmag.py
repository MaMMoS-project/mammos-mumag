import inspect
import sys
from time import time

import esys.escript as e
from esys.escript.linearPDEs import LinearSinglePDE, SolverOptions
from esys.weipa import saveVTK

from materials import Materials
from magnetization import getM
from matrix import gx, gy, gz, dx, dy, dz, poisson
from tools import read_Js, get_mu0
from escript_tools import dot
from converters import operator2matrix

class Hmag:
    def __init__(self, Js, volume, tol=1e-8, verbose=0):
        self.cum_time = 0.0
        self.rhs_time = 0.0
        self.u_time = 0.0
        self.grad_time = 0.0
        self._poisson = poisson(Js, volume)
        self._gx = gx(Js, volume)
        self._gy = gy(Js, volume)
        self._gz = gz(Js, volume)
        self._Js = Js
        if verbose >= 4:
            self.set_verbose(True)

    def solve_u(self, m):
        T0 = time()
        self._poisson.setValue(X=self._Js * m)
        b = self._poisson.getRightHandSide()
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

    def getStiffnessMatrix(self):
        return operator2matrix( self._poisson.getOperator() )
        
    def getGradientMatrices(self):
        return ( operator2matrix(self._gx,diag=False)[0], 
                 operator2matrix(self._gy,diag=False)[0], 
                 operator2matrix(self._gz,diag=False)[0]) 

    def getDivergenzMatrices(self):
        return ( operator2matrix(dx(self._Js),diag=False)[0], 
                 operator2matrix(dy(self._Js),diag=False)[0], 
                 operator2matrix(dz(self._Js),diag=False)[0])  


if __name__ == "__main__":
    try:
        name = sys.argv[1]
    except IndexError:
        sys.exit("Argument `name` missing.")



    # Hmag solver
    materials = Materials(name)
    hmag = Hmag(materials.Js, materials.volume, 1e-12, 1)

    # Get matrices, do this before solving with escript
    mat_stiff, dia_stiff   = hmag.getStiffnessMatrix()
    mat_dx, mat_dy, mat_dz = hmag.getDivergenzMatrices()
    mat_gx, mat_gy, mat_gz = hmag.getGradientMatrices()

    # scalar potential, field, and energy
    m = getM(e.wherePositive(materials.meas), [0.0, 0.0, 1.0])
    u, h = hmag.solve_uh(m)
    emag = e.integrate(-0.5 * e.inner(h, materials.Js * m)) / materials.volume

    # energy
    Js = read_Js(name)

    params = mat_dx, mat_dy, mat_dz, mat_stiff, dia_stiff, mat_gx, mat_gy, mat_gz
    tol = 1e-10

    mu0 = get_mu0()

    with open(name + ".csv", "w") as file:
        file.write(
            inspect.cleandoc(
                f"""
                Magnetostatic energy density of uniformly magnetized cube.
                name,value,explanation
                E_field,{emag/mu0},Energy density evaluated from field (J/m^3).
                E_gradient,{hmag.solve_e(m)/mu0},Energy density evaluated from gradient (J/m^3).
                E_analytic,{(Js * Js / 6)/mu0},Energy density evaluated analytically (J/m^3).
                """
            )
        )

    # field at nodes
    g = hmag.solve_g(m)

    meas = e.whereZero(materials.meas) + materials.meas
    h_at_nodes = e.Vector(0.0, e.Solution(materials.getDomain()))
    h_at_nodes[0] = -materials.volume * (g[0] / meas)
    h_at_nodes[1] = -materials.volume * (g[1] / meas)
    h_at_nodes[2] = -materials.volume * (g[2] / meas)

    saveVTK(
        name + "_hmag", tags=materials.get_tags(), m=m, U=u, h=h, h_nodes=h_at_nodes
    )
