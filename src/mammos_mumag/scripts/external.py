import inspect
import sys
from time import time

import esys.escript as e

from materials import Materials
from magnetization import getM
from tools import read_params, read_Js, normalize, get_mu0
from escript_tools import dot

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

    mag_pars, hext_pars, hmag_on, min_pars, verbose  = read_params(name)
    m, _, _, _ =  mag_pars
    h, start, final, step = hext_pars
    Js = read_Js(name)
    ezee = -Js * (start - step) * (m[0] * h[0] + m[1] * h[1] + m[2] * h[2])

    materials = Materials(name)
    m = getM(e.wherePositive(materials.meas), m)
    external = External(start, final, step, h, materials.meas, materials.volume)
        
    mu0 = get_mu0()
    E_gradient = external.solve_e(m) / mu0
    E_analytic = ezee / mu0

    with open(name + ".csv", "w") as file:
        file.write(
            inspect.cleandoc(
                f"""
                Zeeman energy density for an external field of mu_0 Hext {external.value} (T).
                name,value,explanation
                E_gradient,{E_gradient},Energy evaluated from gradient (J/m^3).
                E_analytic,{E_analytic},Energy evaluated analytically (J/m^3).
                """
            )
        )
