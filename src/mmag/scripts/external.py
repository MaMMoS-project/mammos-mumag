import sys

import esys.escript as e

from mmag.external import External
from mmag.materials import Materials, read_Js
from mmag.magnetisation import getM
from mmag.tools import read_params


def main(name):
    m, h, start, final, step, _, _, _ = read_params(name)
    Js = read_Js(name)
    ezee = -Js * (start - step) * (m[0] * h[0] + m[1] * h[1] + m[2] * h[2])
    materials = Materials(name)
    m = getM(e.wherePositive(materials.meas), m)
    external = External(start, final, step, h, materials.meas, materials.volume)

    print("       h  value     ", external.value)
    print("energy from gradient", external.solve_e(m))
    print("       analytic     ", ezee)


if __name__ == "__main__":
    try:
        name = sys.argv[1]
    except IndexError:
        sys.exit("usage run-escript external.py modelname")
    main(name)
