import sys
import math

import esys.escript as e
from esys.finley import ReadMesh
from esys.weipa import saveVTK

from .tools import get_meas, normalize


class Materials:
    def __init__(self, name):
        self.name = name

    def read_mesh(self, mesh_fname):
        print(f"read mesh from {mesh_fname}")
        self.mesh = ReadMesh(mesh_fname)

    def read_materials(self, materials_fname, size=1.0e-9, scale=0.0):
        if not hasattr(self, "mesh"):
            raise AttributeError("Mesh file needs to be read first.")

        self.K = e.Scalar(0, e.Function(self.mesh))
        self.u = e.Vector(0, e.Function(self.mesh))
        self.Js = e.Scalar(0, e.Function(self.mesh))
        self.A = e.Scalar(0, e.Function(self.mesh))
        self.mu0 = 4e-7 * math.pi
        tags = e.Function(self.mesh).getListOfTags()
        krn = open(materials_fname, "r")
        for tag in tags:
            line = krn.readline().split()
            theta, phi = float(line[0]), float(line[1])
            Js = float(line[4])
            if Js > 0:
                self.u.setTaggedValue(
                    tag,
                    [
                        math.sin(theta) * math.cos(phi),
                        math.sin(theta) * math.sin(phi),
                        math.cos(theta),
                    ],
                )
                self.K.setTaggedValue(tag, self.mu0 * float(line[2]))
                self.Js.setTaggedValue(tag, Js)
                self.A.setTaggedValue(tag, self.mu0 * float(line[5]) / (size * size))
        krn.close()
        if scale == 0.0:
            self.volume = e.integrate(e.wherePositive(self.Js))
        else:
            self.volume = scale * scale * scale
        self.size = size
        self.meas = get_meas(self.Js)

    def computeMh(self, m, h):
        return e.integrate(e.inner(m, h) * self.Js) / self.volume

    def get_tags(self):
        return e.makeTagMap(e.Function(self.mesh))

    def write_vtk(self):
        self.u.expand()
        self.K.expand()
        self.Js.expand()
        self.A.expand()
        saveVTK(
            self.name + ".mat",
            tags=self.get_tags(),
            u=self.u,
            K=self.K / self.mu0,
            Js=self.Js,
            A=self.A * (self.size * self.size),
        )


def read_Js(materials_fname):
    with open(materials_fname) as f:
        ll = f.readline().split()
        Js = float(ll[4])
    return Js


def read_A(materials_fname):
    with open(materials_fname) as f:
        ll = f.readline().split()
        A = float(ll[5])
    return A


def read_AnisotropyEnergy(materials_fname, m):
    m = normalize(m)
    with open(materials_fname) as f:
        ll = f.readline().split()
        theta = float(ll[0])
        phi = float(ll[1])
        n0 = math.sin(theta) * math.cos(phi)
        n1 = math.sin(theta) * math.sin(phi)
        n2 = math.cos(theta)
        K1 = float(ll[2])
    mu0 = 4.0e-7 * math.pi
    return -mu0 * K1 * (m[0] * n0 + m[1] * n1 + m[2] * n2) ** 2.0


if __name__ == "__main__":
    print("materials:")
    try:
        name = sys.argv[1]
    except IndexError:
        sys.exit("usage run-escript materials.py modelname")

    from .scripts.materials import main as materials_script

    materials_script(name)
