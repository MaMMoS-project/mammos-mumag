import inspect
import sys
import math
from time import time

import esys.escript as e

from materials import Materials
from magnetization import xM, getM
from matrix import exani_matrix
from tools import read_A, readAnisotropyEnergy, get_mu0
from escript_tools import dot
from converters import operator2matrix

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
        
    def getMatrix(self):
        return operator2matrix(self._matrix)


if __name__ == "__main__":
    try:
        name = sys.argv[1]
    except IndexError:
        sys.exit("usage run-escript hmag.py modelname")

    materials = Materials(name)

    exani = ExAni(materials.A, materials.K, materials.u, materials.volume)

    scale = materials.volume**0.3333333333333333
    size = materials.size
    m, k = xM(e.wherePositive(materials.meas), scale)

    A = read_A(name)
    mu0 = 4.0e-7 * math.pi

    mat_exani, dia_exani = exani.getMatrix()
    
    mu0 = get_mu0()
    E_gradient = exani.solve_e(m)/mu0
    E_analytic = (2 * k * k * mu0 * A / (size * size))/mu0

    with open(name + "_vortex.csv", "w") as file:
        file.write(
            inspect.cleandoc(
                f"""
                Exchange energy density of a vortex on a, b plane.
                name,value,explanation
                E_gradient,{E_gradient},Energy evaluated from gradient (J/m^3).
                E_analytic,{E_analytic},Energy evaluated analytically (J/m^3).
                """
            )
        )

    uniform_m = [0.0, 0.0, 1.0]
    m = getM(e.wherePositive(materials.meas), uniform_m)
    eani = readAnisotropyEnergy(name, uniform_m)
    E_gradient = exani.solve_e(m) / mu0
    E_analytic = eani / mu0
    
    with open(name + "_uniform.csv", "w") as file:
        file.write(
            inspect.cleandoc(
                f"""
                Magnetocrystalline anisotropy energy density of uniformly magnetized sample in direction {uniform_m}.
                name,value,explanation
                E_gradient,{E_gradient},Energy evaluated from gradient (J/m^3).
                E_analytic,{E_analytic},Energy evaluated analytically (J/m^3).
                """
            )
        )
