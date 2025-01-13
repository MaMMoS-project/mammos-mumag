import sys
import math
from time import time

import esys.escript as e
from esys.escript.linearPDEs import LinearPDE

from .tools import dot


def exani_matrix(A, K, u, volume):
    if A:
        domain = A.getDomain()
    else:
        domain = K.getDomain()
    pde = LinearPDE(domain)
    if A:
        pde.setValue(A=(2.0 * A / volume) * e.identityTensor4(domain))
    if K:
        pde.setValue(D=(-2.0 * K / volume) * e.outer(u, u))
    return pde.getOperator()


class ExAni:
    def __init__(self, A, K, u, volume):
        self._matrix = exani_matrix(A, K, u, volume)
        self.cum_time = 0.0

    def solve_g(self, m):
        T0 = time()
        g = self._matrix.of(m)
        self.cum_time += time() - T0
        return g

    def solve_e(self, m):
        return 0.5 * dot(m, self.solve_g(m))

    def saveMM(self, fn):
        self._matrix.saveMM(fn)


if __name__ == "__main__":
    try:
        name = sys.argv[1]
    except IndexError:
        sys.exit("usage run-escript hmag.py modelname")

    from .scripts.exani import main as exani_script
    exani_script(name)
