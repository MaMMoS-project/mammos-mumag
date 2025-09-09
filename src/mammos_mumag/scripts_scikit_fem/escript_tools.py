import esys.escript as e
from esys.finley import ReadMesh
from esys.escript.linearPDEs import LinearSinglePDE
from esys.weipa import saveVTK
import numpy
from converters import toEscriptScalar, toEscriptVector

def dot(a, b):
    np_a = e.convertToNumpy(a)
    np_b = e.convertToNumpy(b)
    return numpy.dot(np_b.flatten(), np_a.flatten())

def get_meas(Js):
    domain = Js.getDomain()
    pde = LinearSinglePDE(domain)
    pde.setValue(Y=Js)
    return pde.getRightHandSide()
    
def readmesh_get_tags(name):
    domain = ReadMesh(name + ".fly")
    return e.makeTagMap(e.Function(domain))
    
def write_m(name,counter,m,tags):
    saveVTK(f"{name}_{counter:04d}",tags=tags,m=toEscriptVector(m,tags.getDomain()))

def write_magnetization_and_potential(name,counter,m,u,tags):
    saveVTK(f"{name}_{counter:04d}",
            tags=tags,
            m=toEscriptVector(m,tags.getDomain()),
            u=toEscriptScalar(u,tags.getDomain())
             )
