#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys

#before: from esys.finley import ReadMesh
import skfem


class Mesh:
    def __init__(self, name):
        # print("read mesh from " + name + ".fly")
        #before: ReadMesh(name + ".fly")
        self._domain = skfem.mesh.Mesh.load(name + ".med")

    def getDomain(self):
        return self._domain


if __name__ == "__main__":
    print("mesh:")
    try:
        mesh = Mesh(sys.argv[1])

    except IndexError:
        print("Argument `name` missing.")
