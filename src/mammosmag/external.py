import sys
from time import time

import esys.escript as e

from .tools import normalize, dot

import numpy as np


class External:
    def __init__(self, start, final, step, direction, meas, volume):
        self.cum_time = 0.0
        self._meas = meas
        self._volume = volume
        self.direction = normalize(direction)
        self.value = start - step
        self._start = start
        self._final = final
        self._step = step

    def solve_g(self, m):
        T0 = time()
        g = e.Vector(0.0, e.Solution(self._meas.getDomain()))
        g[0] = (-self.value * self.direction[0] / self._volume) * self._meas
        g[1] = (-self.value * self.direction[1] / self._volume) * self._meas
        g[2] = (-self.value * self.direction[2] / self._volume) * self._meas
        self.cum_time += time() - T0
        return g

    def solve_e(self, m):
        return dot(m, self.solve_g(m))

    def next(self):
        if np.abs(self.value - self._final) < 1e-12:
            return False
        if np.sign(self._final - self._start) * (self.value - self._final) <= 0.0:
            self.value += self._step
            return True
        else:
            return False

if __name__ == "__main__":
    try:
        name = sys.argv[1]
    except IndexError:
        sys.exit("usage run-escript external.py modelname")
    
    from .scripts.external import main as external_script
    external_script(name)
