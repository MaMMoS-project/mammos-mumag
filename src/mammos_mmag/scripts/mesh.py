#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys

from esys.finley import ReadMesh


class Mesh:
    def __init__(self, name):
        # print("read mesh from " + name + ".fly")
        self._domain = ReadMesh(name + ".fly")

    def getDomain(self):
        return self._domain


if __name__ == "__main__":
    print("mesh:")
    try:
        mesh = Mesh(sys.argv[1])

    except IndexError:
        print("Argument `name` missing.")
