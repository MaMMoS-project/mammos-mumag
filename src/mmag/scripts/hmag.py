import sys

import esys.escript as e
from esys.weipa import saveVTK

from mmag.hmag import Hmag
from mmag.materials import Materials, read_Js
from mmag.magnetization import getM


def main(name):
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


if __name__ == "__main__":
    try:
        name = sys.argv[1]
    except IndexError:
        sys.exit("usage run-escript hmag.py modelname")
    main(name)
