#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys

#before: from esys.finley import ReadMesh
from skfem.mesh import Mesh as SkfemMesh
import meshio
import numpy as np
from skfem.io import from_meshio


class Mesh:
    def __init__(self, name):
        # print("read mesh from " + name + ".fly")
        #before: ReadMesh(name + ".fly")
        mm = meshio.read(name + ".med")
        mm.cell_sets = {
            subdomain[0]: [
                np.where(tag == idx.item())[0]
                for tag in mm.cell_data["cell_tags"]
            ]
            for idx, subdomain in mm.cell_tags.items()
        }
        self._domain = from_meshio(mm)

    def getDomain(self):
        return self._domain


if __name__ == "__main__":
    print("mesh:")
    try:
        mesh = Mesh(sys.argv[1])

    except IndexError:
        print("Argument `name` missing.")
