import esys.escript as e
from esys.escript.pdetools import MaskFromTag

def findBoundary(x):
    xe = []
    for i in range(3):
        xe.append(e.sup(x[i]))
    boundaryMask = (
        e.whereZero(x[0] + xe[0])
        + e.whereZero(x[0] - xe[0])
        + e.whereZero(x[1] + xe[1])
        + e.whereZero(x[1] - xe[1])
        + e.whereZero(x[2] + xe[2])
        + e.whereZero(x[2] - xe[2])
    )
    return boundaryMask

def check_domain(x):
    Rinf = e.sup(e.length(x))
    airbox = abs(np.sum(e.convertToNumpy(e.whereZero(e.length(x) - Rinf))) - 8) < 1e-8
    return airbox, Rinf

# IEEE TRANSACTIONS ON MAGNETICS, VOL. 26, NO. 5, SEPTEMBER 1990
def get_T3D(domain, R, Rinf, tags):
    k = e.Tensor(0, e.Function(domain))
    xx = e.Function(domain).getX()
    x0 = xx[0]
    x1 = xx[1]
    x2 = xx[2]
    x = e.length(xx)
    a2 = R * (Rinf - R)
    ax = Rinf / x
    a2x2 = a2 / (x * x)
    k[0, 0] = a2x2 * (1 + (ax * (2 - ax) / (ax - 1) ** 2) * (1 - (x0 / x) ** 2))
    k[1, 1] = a2x2 * (1 + (ax * (2 - ax) / (ax - 1) ** 2) * (1 - (x1 / x) ** 2))
    k[2, 2] = a2x2 * (1 + (ax * (2 - ax) / (ax - 1) ** 2) * (1 - (x2 / x) ** 2))
    k[0, 1] = -a2x2 * ((ax * (2 - ax) / (ax - 1) ** 2) * (x0 / x) * (x1 / x))
    k[0, 2] = -a2x2 * ((ax * (2 - ax) / (ax - 1) ** 2) * (x0 / x) * (x2 / x))
    k[1, 2] = -a2x2 * ((ax * (2 - ax) / (ax - 1) ** 2) * (x1 / x) * (x2 / x))
    k[1, 0] = k[0, 1]
    k[2, 0] = k[0, 2]
    k[2, 1] = k[1, 2]
    for tag in tags[:-1]:
        k.setTaggedValue(tag, e.kronecker(domain))
    return k
