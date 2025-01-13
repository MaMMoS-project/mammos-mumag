import esys.escript as e
from esys.weipa import saveVTK
import sys

from mammosmag.materials import Materials
from mammosmag.tools import read_params


def main(name):
    params = read_params(name)
    materials = Materials(name)
    m = getM(e.wherePositive(materials.meas), params[0], params[1])
    i = 0
    saveVTK(name + f".{i:04}", tags=materials.get_tags(), m=m)

if __name__ == "__main__":
    try:
        name = sys.argv[1]
    except IndexError:
        sys.exit("usage run-escript magnetization.py modelname")
    main(name)
