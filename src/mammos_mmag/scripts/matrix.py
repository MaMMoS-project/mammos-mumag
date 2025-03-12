import numpy as np

from esys.escript.linearPDEs import LinearSinglePDE, LinearPDE
import esys.escript as e

from trafo import findBoundary, get_T3D
from esys.escript.pdetools import MaskFromTag

def grad_matrix(Js, volume, v):
    pde = LinearSinglePDE(Js.getDomain())
    pde.setValue(C=(Js / volume) * v)
    return pde.getOperator()

def gx(Js, volume):
    return grad_matrix(Js, volume, [1, 0, 0])

def gy(Js, volume):
    return grad_matrix(Js, volume, [0, 1, 0])

def gz(Js, volume):
    return grad_matrix(Js, volume, [0, 0, 1])

def rhs_matrix(Js,v):
    pde = LinearPDE(Js.getDomain())
    pde.setValue(B=Js*v)
    return pde.getOperator()

def dx(Js):
    return rhs_matrix(Js,[1,0,0])

def dy(Js):
    return rhs_matrix(Js,[0,1,0])

def dz(Js):
    return rhs_matrix(Js,[0,0,1])

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
    
def poisson(Js, volume):
    domain = Js.getDomain()

    x = domain.getX()
    Rinf = e.sup(e.length(x))
    boundaryMask = e.whereZero(e.length(x) - Rinf)
    airbox = abs(np.sum(e.convertToNumpy(boundaryMask)) - 8) < 1e-8
    if airbox:
        print('apply square airbox')
        boundaryMask = findBoundary(x)
        k = e.kronecker(domain)
    else:
        # print('apply spherical shell transformation')
        tags = e.Function(domain).getListOfTags()
        R = e.sup(e.length(x) * MaskFromTag(domain, tags[-2]))
        k = get_T3D(domain, R, Rinf, tags)

    pde = LinearSinglePDE(domain, isComplex=False)
    pde.setSymmetryOn()
    pde.setValue(A=k, q=boundaryMask, r=0.0)
        
    return pde
    
def stiffness_matrix(Js, volume):
    pde = poisson(Js, volume)
    return pde.getOperator()
