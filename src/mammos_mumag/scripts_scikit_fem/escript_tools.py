# import esys.escript as e
# from esys.finley import ReadMesh
# from esys.escript.linearPDEs import LinearSinglePDE
# from esys.weipa import saveVTK
import numpy
# from converters import toEscriptScalar, toEscriptVector
import skfem

# def dot(a, b):
#     np_a = e.convertToNumpy(a)
#     np_b = e.convertToNumpy(b)
#     return numpy.dot(np_b.flatten(), np_a.flatten())

def get_meas(Js, basis_0):
    """
    Returns b with entries b_K = ∫_K (v_K * Js_K) dx = Js_K * |K|
    where v_K is the P0 test function on cell K.
    """
    @skfem.LinearForm
    def rhs(v, w):
        return v * w['Js']           # ∫ v * Js

    return skfem.asm(rhs, basis_0, Js=Js)  # length = number of cells
    
# def readmesh_get_tags(name):
#     domain = ReadMesh(name + ".fly")
#     return e.makeTagMap(e.Function(domain))
    
# def write_m(name,counter,m,tags):
#     saveVTK(f"{name}_{counter:04d}",tags=tags,m=toEscriptVector(m,tags.getDomain()))

# def write_magnetization_and_potential(name,counter,m,u,tags):
#     saveVTK(f"{name}_{counter:04d}",
#             tags=tags,
#             m=toEscriptVector(m,tags.getDomain()),
#             u=toEscriptScalar(u,tags.getDomain())
#              )
