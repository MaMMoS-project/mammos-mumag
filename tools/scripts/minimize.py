import esys.escript as e
from esys.escript.minimizer import CostFunction

# from lbfgsm import LBFGSM
from lbfgs import LBFGS

from tools import dot, Linf_norm, L2_norm, get_logger

from time import time

import numpy as np


def inner3(a, b):
    c = e.Vector(0, e.Solution(a.getDomain()))
    ab = e.inner(a, b)
    c[0] = ab
    c[1] = ab
    c[2] = ab
    return c


def projection(m, h):
    mh = inner3(m, h)
    return h - (mh * m)


"""
def cg(self,m,gNoProj, x,b,k, itmax):
    normB = self._fun.norm(b)
    tau   = self.computeTau(normB,k)
    r     = b-self.hessianVector(m,gNoProj,x)
    p     = r.copy()
    rr0   = self._fun.dot(r,r)
    for j in range(itmax):
      Ap = self.hessianVector(m,gNoProj,p)
      rho = self._fun.dot(Ap,p)
      if rho < 0:
        return x
      alpha = rr0/rho
      x = x + alpha*p
      r = r - alpha*Ap
      rr = self._fun.dot(r,r)
      if isinstance(r,e.Data):
        d = math.sqrt(rr)/normB
      else:
        d = math.sqrt(rr.value)/normB
      if d < tau:
        break
      beta = rr/rr0
      p = execute(r + beta*p)
      rr0 = rr
    return x
"""


class MumagFunc(CostFunction):
    def __init__(self, external, exani, hmag, cgiter=5):
        super().__init__()
        self._external = external
        self._exani = exani
        self._hmag = hmag
        self.cgiter = cgiter
        self.dot_time = 0.0
        self.norm_time = 0.0
        self.energies_time = 0.0

    def hessianVector(self, m, gloc, p):
        hp = self._exani.solve_g(p)  # self._fun.gLocalNoProj(p)
        php = projection(m, hp)
        # ph = inner3(p, gloc)
        hm = inner3(m, gloc)
        Hp = php - (hm * p)  # - ph*m  # omitting this term makes the hessian symmetric
        return Hp

    def simplecg(self, m, gloc, x, b):
        eps2 = np.sqrt(np.finfo(float).eps)
        normB = self.getL2NormAndCount(b)
        tau = min(1.0, normB)

        r = b - self.hessianVector(m, gloc, x)
        p = r.copy()
        rr0 = self.getDualProductAndCount(r, r)

        d = np.sqrt(rr0) / normB
        if d < tau:
            return x

        for j in range(self.cgiter):
            Ap = self.hessianVector(m, gloc, p)
            rho = self.getDualProductAndCount(Ap, p)
            if rho < eps2 * self.getDualProductAndCount(
                p, p
            ):  # Luksan, Vlcek, technical report 1177
                # print(j,'exit Luksan')
                return x
            alpha = rr0 / rho
            x += alpha * p
            r -= alpha * Ap
            rr = self.getDualProductAndCount(r, r)
            d = np.sqrt(rr) / normB
            if d < tau:
                # print(j,'exit tau')
                return x
            beta = rr / rr0
            p = r + beta * p
            rr0 = rr

        return x

    def getDualProduct(self, m, g):
        T0 = time()
        d = dot(g, m)
        self.dot_time += time() - T0
        return d

    def getArguments(self, m):
        T0 = time()
        mn = e.normalize(m)
        if self._hmag:
            a = (
                mn,
                self._external.solve_g(mn),
                self._exani.solve_g(mn),
                self._hmag.solve_g(mn),
            )
        else:
            a = (mn, self._external.solve_g(mn), self._exani.solve_g(mn), None)
        self.energies_time += time() - T0
        return a

    def getValue(self, m, *args):
        T0 = time()
        mn, g_ext, g_exani, g_hmag = args
        if g_hmag:
            value = dot(mn, g_ext) + 0.5 * dot(mn, g_exani) + 0.5 * dot(mn, g_hmag)
        else:
            value = dot(mn, g_ext) + 0.5 * dot(mn, g_exani)
        self.energies_time += time() - T0
        return value

    def getGradient(self, m, *args):
        T0 = time()
        mn, g_ext, g_exani, g_hmag = args
        if g_hmag:
            g = e.cross(mn, e.cross(g_ext + g_exani + g_hmag, mn))
        else:
            g = e.cross(mn, e.cross(g_ext + g_exani, mn))
        self.energies_time += time() - T0
        return g

    def getNorm(self, m):
        T0 = time()
        n = Linf_norm(m)
        self.norm_time += time() - T0
        return n

    def getL2Norm(self, m):
        T0 = time()
        n = L2_norm(m)
        self.norm_time += time() - T0
        return n

    def getL2NormAndCount(self, m):
        self.L2Norm_calls += 1
        return self.getL2Norm(m)

    def grad2mxh(self, g):
        T0 = time()
        n = Linf_norm(g / (e.whereZero(self._external._meas) + self._external._meas))
        self.norm_time += time() - T0
        return n

    def getInverseHessianApproximation(self, r, m, *args, initializeHessian=False):
        """Approximate inverse of Hessian.

        This function returns an approximation of the
        Hessian inverse at m for vector r.
        Hx = r
        """
        if self.cgiter > 0:
            m = args[0]
            gloc = args[2]
            x = r.copy()
            return self.simplecg(m, gloc, x, r)
        else:
            return r

    def resetStatistics(self):
        """Reset all counters."""
        self.DualProduct_calls = 0
        self.Value_calls = 0
        self.Gradient_calls = 0
        self.Arguments_calls = 0
        self.InverseHessianApproximation_calls = 0
        self.Norm_calls = 0
        self.L2Norm_calls = 0

    def getStatistics(self, minimizer_iter, minimizer_time):
        """Return the call statistics as a string."""
        out = "\n"
        out += f"lbfgs iterations                    : {minimizer_iter}\n"
        out += "number of inner product evaluations : %d\n" % self.DualProduct_calls
        out += "number of norm evaluations          : %d\n" % self.Norm_calls
        out += "number of L2 norm evaluations       : %d\n" % self.L2Norm_calls
        out += "number of argument evaluations      : %d\n" % self.Arguments_calls
        out += "number of cost function evaluations : %d\n" % self.Value_calls
        out += "number of gradient evaluations      : %d\n" % self.Gradient_calls
        out += f"minimizer time                      : {minimizer_time}\n"
        out += f"  inner products                    : {self.dot_time}\n"
        out += f"  norms                             : {self.norm_time}\n"
        out += f"  function and gradient             : {self.energies_time}\n"
        solver_iter, solver_time = self._hmag.getDiagnostics()
        out += f"    zeeman                          : {self._external.cum_time}\n"
        out += f"    exchange+anistorpy              : {self._exani.cum_time}\n"
        out += f"    magnetostatic                   : {self._hmag.cum_time}\n"
        out += f"      right hand side               : {self._hmag.rhs_time}\n"
        out += f"      potential                     : {self._hmag.u_time}\n"
        out += f"      gradient                      : {self._hmag.grad_time}\n"
        out += f"    linear system iterations        : {solver_iter}"
        return out


class Minimize:
    def __init__(self, external, exani, hmag, min_params):
        self.cum_time = 0.0
        truncation, m_tol, grad_tol, precond_iter, iter_max, verbose = min_params
        self.F = MumagFunc(external, exani, hmag, cgiter=precond_iter)

        logger = get_logger("min", verbose)
        self._solver = LBFGS(self.F, logger=logger)
        self._solver.setOptions(
            truncation=truncation,
            m_tol=m_tol,
            grad_tol=grad_tol,
            iterMax=iter_max,
            scaleSearchDirection=False,
            restart=50,
        )

    def solve(self, m):
        T0 = time()
        self._solver.run(m)
        self.cum_time += time() - T0
        return self._solver.getResult()

    def getStatistics(self):
        return self.F.getStatistics(self._solver.total_iter, self.cum_time)
