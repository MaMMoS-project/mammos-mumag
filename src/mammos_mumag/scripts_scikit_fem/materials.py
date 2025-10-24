import sys
from math import sin, cos, pi

# import esys.escript as e # to delete
import skfem

# from esys.weipa import saveVTK
from skfem.mesh import Mesh as SkfemMesh

import mesh
from escript_tools import get_meas

from skfem.models.poisson import mass, unit_load
import numpy as np

class Materials(mesh.Mesh):
    """Class to handle material properties for micromagnetic simulations using scikit-fem."""
    def __init__(self, name, size=1.0e-9, scale=0.0):
        self.name = name
        mesh.Mesh.__init__(self, name)
        domain = self.getDomain()
        if domain.t.shape[0] != 4:
            raise ValueError("Unsupported topology; add the appropriate P0 element.")
        basis_0 = skfem.Basis(domain, skfem.ElementTetP0())
        basis_V0 = skfem.Basis(domain, skfem.ElementVector(skfem.ElementTetP0()))
        self.K = basis_0.zeros()
        self.u = basis_V0.zeros() # direction of anisotropy constant
        self.Js = basis_0.zeros()
        self.A = basis_0.zeros()
        self.mu0 = 4e-7 * pi
        tags = list(domain.subdomains.keys())
        krn = open(name + ".krn", "r")
        for tag in tags:
            line = krn.readline().split()
            theta, phi = float(line[0]), float(line[1])
            Js = float(line[4])
            u_vector = [sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)]
            if Js > 0:
                for comp in range(3):
                    self.u[basis_V0.get_dofs(elements=tag).all([f'u^{comp+1}'])] = u_vector[comp]
                self.K[basis_0.get_dofs(elements=tag)] = self.mu0 * float(line[2])
                self.Js[basis_0.get_dofs(elements=tag)] = Js
                self.A[basis_0.get_dofs(elements=tag)] = self.mu0 * float(line[5]) / (size * size)
        krn.close()

        if scale == 0.0:
            # skfem.asm(unit_load, basis_0) assembles the linear 
            # form “load = 1” on the P0 basis, so each entry of cell_meas 
            # is the integral of 1 over one element—effectively that 
            # element’s length/area/volume.
            self.volume = skfem.asm(unit_load, basis_0)[self.Js > 0].sum()
        else:
            self.volume = scale * scale * scale
        self.size = size
        self.meas = get_meas(self.Js, basis_0)

    # 
    def computeMh(self, m, h):
        """Computes the average magnetization component in the direction of h."""
        return (np.sum(m * h) * self.Js).sum() / self.volume

    def get_tags(self):
        """Returns a tag map for the mesh elements.
        Analogue of escript: makeTagMap(Function(domain))
        -> returns {tag: chi_tag}, where chi_tag is P0 (cellwise) 1 on tag, 0 elsewhere,
        plus the P0 Basis used.

        Works with MeshTri1 / MeshTet1 that have mesh.subdomains populated.
        """
        domain = self.getDomain()

        basis_0 = skfem.Basis(domain, skfem.ElementTetP0())

        # pull tags from mesh.subdomains (dict: tag -> np.ndarray of element indices)
        if not hasattr(domain, "subdomains") or len(domain.subdomains) == 0:
            raise ValueError("domain.subdomains is empty; no element tags found.")

        tagmap = {}
        for tag, elem_idx in domain.subdomains.items():
            chi = basis_0.zeros()                # one DOF per element
            chi[np.asarray(elem_idx, dtype=int)] = 1.0
            tagmap[int(tag)] = chi

        return tagmap #, basis0

    def write_vtk(self):
        """Writes material properties to a VTK file for visualization."""
        # constructing the per-element vector field u for export
        domain = self.getDomain()
        basis_V0 = skfem.Basis(domain, skfem.ElementVector(skfem.ElementTetP0()))
        nelem = domain.nelements
        tags = list(domain.subdomains.keys())
        dim = 3 # for tetrahedral mesh in 3D
        idx_all = basis_V0.get_dofs(elements=tags).all()
        blocks = [idx_all[i*nelem:(i+1)*nelem] for i in range(dim)]
        comps = [self.u[blk] for blk in blocks]
        u_cell = np.column_stack(comps)
        domain.save(self.name + "_mat.vtk", cell_data={"Js": [self.Js], "K": [self.K], "A": [self.A], "u": [u_cell]})


if __name__ == "__main__":
    # print("materials:")
    try:
        name = sys.argv[1]
    except IndexError:
        sys.exit("Argument `name` missing.")
    materials = Materials(name)
    materials.write_vtk()
