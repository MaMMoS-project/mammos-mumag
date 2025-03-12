import sys
from math import sin, cos, pi

import esys.escript as e

from esys.weipa import saveVTK

import mesh
from escript_tools import get_meas


class Materials(mesh.Mesh):
    def __init__(self, name, size=1.0e-9, scale=0.0):
        self.name = name
        mesh.Mesh.__init__(self, name)
        domain = self.getDomain()
        self.K = e.Scalar(0, e.Function(domain))
        self.u = e.Vector(0, e.Function(domain))
        self.Js = e.Scalar(0, e.Function(domain))
        self.A = e.Scalar(0, e.Function(domain))
        self.mu0 = 4e-7 * pi
        tags = e.Function(domain).getListOfTags()
        krn = open(name + ".krn", "r")
        for tag in tags:
            line = krn.readline().split()
            theta, phi = float(line[0]), float(line[1])
            Js = float(line[4])
            if Js > 0:
                self.u.setTaggedValue(
                    tag, [sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)]
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
        domain = self.getDomain()
        return e.makeTagMap(e.Function(domain))

    def write_vtk(self):
        self.u.expand()
        self.K.expand()
        self.Js.expand()
        self.A.expand()
        saveVTK(
            name + "_mat",
            tags=self.get_tags(),
            u=self.u,
            K=self.K / self.mu0,
            Js=self.Js,
            A=self.A * (self.size * self.size),
        )


if __name__ == "__main__":
    # print("materials:")
    try:
        name = sys.argv[1]
    except IndexError:
        sys.exit("Argument `name` missing.")
    materials = Materials(name)
    materials.write_vtk()
