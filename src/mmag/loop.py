import sys

import esys.escript as e
from esys.weipa import saveVTK

from .tools import read_params, dot
from .magnetization import getM
from .materials import Materials
from .hmag import Hmag
from .exani import ExAni
from .external import External
from .minimize import Minimize


class Loop:
    def __init__(self, name, outdir=None):
        self.name = name
        if outdir is None:
            self.outname = name
        else:
            outdir.mkdir(exist_ok=True)
            self.outname = str(outdir / name)

    def read_mesh(self, mesh_fname):
        self.mesh_fname = mesh_fname if type(mesh_fname) is str else str(mesh_fname)
        self.materials = Materials(self.name)
        self.materials.read_mesh(self.mesh_fname)

    def read_params(self, params_fname):
        self.params_fname = (
            params_fname if type(params_fname) is str else str(params_fname)
        )
        self.params = read_params(self.params_fname)

    def read_materials(self, materials_fname):
        self.materials_fname = (
            materials_fname if type(materials_fname) is str else str(materials_fname)
        )
        self.materials.read_materials(self.materials_fname)

    def compute_mh(self, m):
        return (
            dot(
                (
                    self._direction[0] * m[0]
                    + self._direction[1] * m[1]
                    + self._direction[2] * m[2]
                ),
                self._meas,
            )
            / self._volume
        )

    def solve(self, m):
        with open(self.outname + ".dat", "w") as f:
            i = 0
            mh_old = 9.99e9
            while self._external.next():
                m, e = self._minimize.solve(m)
                mh = self.compute_mh(m)
                print("-dem->  ", self._external.value, mh, e, flush=True)
                if abs(mh - mh_old) >= self._mstep:
                    mh_old = mh
                    i = i + 1
                    saveVTK(
                        self.outname + f".{i:04}", tags=self.materials.get_tags(), m=m
                    )
                    f.write(f"{i:04} {self._external.value} {mh} {e}\n")
                else:
                    f.write(f"{0:04} {self._external.value} {mh} {e}\n")
                if mh < self._mfinal:
                    break

    def getStatistics(self):
        return self._minimize.getStatistics()

    def run(self):
        if not hasattr(self, "mesh_fname"):
            raise AttributeError("Mesh file has not been read yet.")
        if not hasattr(self, "params_fname"):
            raise AttributeError("Params file has not been read yet.")
        if not hasattr(self, "materials_fname"):
            raise AttributeError("Materials file has not been read yet.")

        m, state, h, start, final, step, self._mstep, self._mfinal, min_params = (
            self.params
        )
        self._external = External(
            start, final, step, h, self.materials.meas, self.materials.volume
        )
        exani = ExAni(
            self.materials.A, self.materials.K, self.materials.u, self.materials.volume
        )
        hmag_on, truncation, tol_u, tol_mxh, precond_iter, iter_max, verbose = (
            min_params
        )
        if hmag_on == 1:
            hmag = Hmag(self.materials.Js, self.materials.volume, tol_u, verbose)
        else:
            hmag = None
        self._direction = h
        self._meas = self.materials.meas
        self._volume = self.materials.volume
        self._minimize = Minimize(self._external, exani, hmag, min_params[1:])
        m = getM(e.wherePositive(self.materials.meas), self.params[0], self.params[1])
        i = 0
        saveVTK(self.outname + f".{i:04}", tags=self.materials.get_tags(), m=m)
        self.solve(m)
        print(self.getStatistics())


if __name__ == "__main__":
    try:
        name = sys.argv[1]
    except IndexError:
        sys.exit("usage mmag run loop -n <name_system>")
    from .scripts.loop import main as loop_script

    loop_script(name)
