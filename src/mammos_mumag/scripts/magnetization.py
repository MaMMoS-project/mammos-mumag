import esys.escript as e
from esys.weipa import saveVTK
import math
import sys

from materials import Materials
from tools import read_params

def randomM(domain):
  fs = e.Solution(domain)
  return e.normalize(2.0*e.RandomData((3,),fs,1)-1.0)

def getVortex(domain):
    m = e.Vector(0, e.Solution(domain))
    x = domain.getX()
    r = e.sqrt(x[1] * x[1] + x[2] * x[2])
    rc = 0.14 * e.sup(r)
    m[0] = e.exp(-2 * (r / rc))
    m[1] = -(x[2] / r) * e.sqrt(1 - e.exp(-4 * (r * r) / (rc * rc)))
    m[2] = (x[1] / r) * e.sqrt(1 - e.exp(-4 * (r * r) / (rc * rc)))
    return m


def getFlower(domain):
    m = e.Vector(0, e.Solution(domain))
    x = domain.getX()
    s = e.sup(x)
    a = 10
    m[0] = (x[0] * x[2]) / (a * s)
    m[1] = (x[1] * x[2]) / (a * s)
    m[2] = 1.0
    return m


def getTwisted(domain):
    m = e.Vector(0, e.Solution(domain))
    x = domain.getX()
    r = e.sqrt(x[1] * x[1] + x[0] * x[0])
    s = e.sup(x)
    a = 10
    # b = 4
    m[0] = (x[0] * x[2]) / (a * s) + 4 * (
        e.wherePositive(x[2]) * (-x[1] / r) * (x[2] / s)
        + e.whereNegative(x[2]) * (x[1] / r) * (-x[2] / s)
    )
    m[1] = (x[1] * x[2]) / (a * s) + 4 * (
        e.wherePositive(x[2]) * (x[0] / r) * (x[2] / s)
        + e.whereNegative(x[2]) * (-x[0] / r) * (-x[2] / s)
    )
    m[2] = 1.0
    return m


def getM(mask, v, state=None):
    domain = mask.getDomain()
    if state == 2:
        m = mask * getVortex(domain)
        # print('create vortex state')
    elif state == 1:
        m = mask * getFlower(domain)
        # print('create flower state')
    elif state == 3:
        m = mask * getTwisted(domain)
    elif state == 4:
        m = mask * randomM(domain)
        print('create random magnetization')
    else:
        m = mask * e.Vector(v, e.Solution(domain))
        # print('create uniformly magnetised state')
    return e.normalize(m)


def xM(mask, scale):
    domain = mask.getDomain()
    m = e.Vector(0, e.Solution(domain))
    x = domain.getX()
    k = 0.5 * math.pi / scale
    kr = k * (x[0] + x[1])
    m[0] = e.sin(kr)
    m[1] = e.cos(kr)
    m[2] = 0
    return mask * m, k


"""
exchange energy:
  mx  =   sin(kr)
  my  =   cos(kr)
  dmx              dmy
  --- = k*cos(kr), --- = - k*sin(kr)
  dx               dx
  dmx              dmy
  --- = k*cos(kr), --- = - k*sin(kr)
  dy               dy
  mx,i mx,i = k*k*cos2 + k*k*cos2 = 2 k*k *cos2
  my,i my,i = k*k*sin2 + k*k*sin2 = 2 k*k *sin2
  eex = 2 A k^2
"""

if __name__ == "__main__":
    try:
        name = sys.argv[1]
    except IndexError:
        sys.exit("usage run-escript magnetization.py modelname")

    mag_pars, hext_pars, hmag_on, min_pars, verbose  = read_params(name)
    m, _, _, state_id = mag_pars

    materials = Materials(name)
    m = getM(e.wherePositive(materials.meas), m, state_id)
    i = 0
    saveVTK(name + f".{i:04}", tags=materials.get_tags(), m=m)
