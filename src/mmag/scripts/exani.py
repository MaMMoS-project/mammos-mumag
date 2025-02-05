import sys

import esys.escript as e

from mammosmag.exani import ExAni
from mammosmag.materials import Materials, read_A, read_AnisotropyEnergy
from mammosmag.magnetization import getM, xM

def main(name):
    materials = Materials(name)

    exani = ExAni(materials.A, materials.K, materials.u, materials.volume)

    scale = materials.volume**0.3333333333333333
    size = materials.size
    m, k = xM(e.wherePositive(materials.meas), scale)

    A = read_A(name)
    mu0 = 4.0e-7 * math.pi

    print("vortex on a, b plane")
    print("energy from gradient", exani.solve_e(m))
    print("       analytic     ", 2 * k * k * mu0 * A / (size * size))

    uniform_m = [0.0, 0.0, 1.0]
    m = getM(e.wherePositive(materials.meas), uniform_m)
    eani = readAnisotropyEnergy(name, uniform_m)

    print("uniform m", uniform_m)
    print("energy from gradient", exani.solve_e(m))
    print("       analytic     ", eani)


if __name__ == "__main__":
    try:
        name = sys.argv[1]
    except IndexError:
        sys.exit("usage mammosmag run exani -n <name_system>")
    main(name)
