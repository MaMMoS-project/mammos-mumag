"""Simulation class."""

import meshio
import os
import shlex
import shutil
import subprocess

from .materials import Materials
from .parameters import Parameters
from .tools import check_dir
from . import _run_escript_bin as run_escript
from . import _scripts_directory as scripts_dir


IS_POSIX = os.name == "posix"


class Simulation:
    """Simulation class."""

    def __init__(self):
        """Initialize class."""
        self.parameters = Parameters()
        self.materials = Materials()

    def run_file(self, file, outdir="out"):
        """Run file using `esys.escript`.

        :param script: path of file.
        :type script: str or pathlib.Path
        :param outdir: Working directory. Defaults to "out"
        :type outdir: str or pathlib.Path, optional
        """
        cmd = shlex.split(
            f"{run_escript} {file}",
            posix=IS_POSIX,
        )
        run_subprocess(cmd, cwd=outdir)

    def run_script(self, script, outdir, name):
        """Run pre-defined script.

        :param script: Name of pre-defined script.
        :type script: str
        :param outdir: Working directory
        :type outdir: str or pathlib.Path
        :param name: System name
        :type name: str
        """
        cmd = shlex.split(
            f"{run_escript} {scripts_dir / script}.py {name}",
            posix=IS_POSIX,
        )
        run_subprocess(cmd, cwd=outdir)

    def run_exani(self, outdir="exani", name="out"):
        """Run "exani" script.

        Test the computation of the exchange and anisotropy energy density.
        This gives the exchange energy density of a vortex in the x-y plane
        and the anistropy energy density in the uniformly magnetized state.
        Here we have placed the anistropy direction paralle to to the z-axis.
        The anisotropy energy density is calculated as -K dot(m,k)^2 where m
        is the unit vector of magnetization and k is the anisotropy direction.
        K is the magnetocrystalline anisotropy constant.

        :param outdir: Working directory, defaults to "exani".
        :type outdir: str or pathlib.Path, optional
        :param name: System name, defaults to "out".
        :type name: str, optional.
        """
        outdir = check_dir(outdir)
        shutil.copyfile(self.mesh_path, outdir / f"{name}.fly")
        self.materials.write_krn(outdir / f"{name}.krn")

        self.run_script(
            script="exani",
            outdir=outdir,
            name=name,
        )

    def run_external(self, outdir="external", name="out"):
        """Run "external" script.

        Compute the Zeemann energy by finite elements and analytically.

        :param outdir: Working directory, defaults to "external".
        :type outdir: str or pathlib.Path, optional
        :param name: System name, defaults to "out".
        :type name: str, optional.
        """
        outdir = check_dir(outdir)
        shutil.copyfile(self.mesh_path, outdir / f"{name}.fly")
        self.materials.write_krn(outdir / f"{name}.krn")
        self.parameters.write_p2(outdir / f"{name}.p2")

        self.run_script(
            script="external",
            outdir=outdir,
            name=name,
        )

    def run_hmag(self, outdir="hmag", name="out"):
        r"""Run "hmag" script.

        This script evaluates the magnetostatic energy density
        and the field of a uniformly magnetized geometry.
        Creates the `vtk` file for visualisation of the magnetic scalar potential
        and the magnetic field. With linear basis function for the magnetic scalar
        potential u, the magnetostatic field h = -grad(u) is defined at the finite
        elements. By smoothing the field can be transfered to the nodes of the
        finite element mesh.

        The software also gives the magnetostatic energy density computed with
        finite elements and compares it with the analytic soluation.
        Three energy values are compared:

        * from field: (Integral over (1/2) field * magnetic_polarization)/volume

        * from gradient: :math:`1/2 \sum_i m_i \cdot g_i`, where :math:`m_i`
          and :math:`g_i` are the unit vector of the magnetization and the gradient
          of the energy normalized by the volume of the energy with respect to
          :math:`m_i` at the nodes of the finite element mesh.

        * analytic: :math:`J_s^2 / (6 \mu_0)`

        :param outdir: Working directory, defaults to "hmag".
        :type outdir: str or pathlib.Path, optional
        :param name: System name, defaults to "out".
        :type name: str, optional.
        """
        outdir = check_dir(outdir)
        shutil.copyfile(self.mesh_path, outdir / f"{name}.fly")
        self.materials.write_krn(outdir / f"{name}.krn")

        self.run_script(
            script="hmag",
            outdir=outdir,
            name=name,
        )
        self.hmag = meshio.read(outdir / f"{name}_hmag.vtu")

    def run_loop(self, outdir="loop", name="out"):
        r"""Run "loop" script.

        Compute demagnetization curves.
        Creates the file `cube.dat` which gives the demagnetization curve.
        The columns of the file are:

        * `vtk number` the number of the `vtk` file that corresponds
          to the field and magnetic polarisation values in the line `mu0 Hext`
          the value of mu_0 Hext (T) where mu_0 is the permability of vacuum
          and Hext is the external value of the external field.

        * `polarisation` the componenent of magnetic polarisation (T)
          parallel to the direction of the external field.

        * `energy density` the energy density (J/m^3) of the current state.

        :param outdir: Working directory, defaults to "loop".
        :type outdir: str or pathlib.Path, optional
        :param name: System name, defaults to "out".
        :type name: str, optional.
        """
        outdir = check_dir(outdir)
        shutil.copyfile(self.mesh_path, outdir / f"{name}.fly")
        self.materials.write_krn(outdir / f"{name}.krn")
        self.parameters.write_p2(outdir / f"{name}.p2")

        self.run_script(
            script="loop",
            outdir=outdir,
            name=name,
        )
        self.loop_vtu_list = [
            meshio.read(outdir / fname)
            for fname in os.listdir(outdir)
            if "vtu" in fname
        ]

    def run_magnetization(self, outdir="magnetization", name="out"):
        """Run "magnetization" script.

        Creates the `vtk` file for the visualisation of the material properties.

        :param outdir: Working directory, defaults to "magnetization".
        :type outdir: str or pathlib.Path, optional
        :param name: System name, defaults to "out".
        :type name: str, optional.
        """
        outdir = check_dir(outdir)
        shutil.copyfile(self.mesh_path, outdir / f"{name}.fly")
        self.materials.write_krn(outdir / f"{name}.krn")
        self.parameters.write_p2(outdir / f"{name}.p2")

        self.run_script(
            script="magnetization",
            outdir=outdir,
            name=name,
        )

    def run_mapping(self, outdir="magnetization", name="out"):
        """Run "mapping" script.

        Test the energy calculations with matrices.
        The module mapping.py contains the tools for mapping from the finite element
        bilinear forms to sparse matrices. We use sparse matrix methods from `jax`.

        :param outdir: Working directory, defaults to "magnetization".
        :type outdir: str or pathlib.Path, optional
        :param name: System name, defaults to "out".
        :type name: str, optional.
        """
        outdir = check_dir(outdir)
        shutil.copyfile(self.mesh_path, outdir / f"{name}.fly")
        self.materials.write_krn(outdir / f"{name}.krn")
        self.parameters.write_p2(outdir / f"{name}.p2")

        self.run_script(
            script="mapping",
            outdir=outdir,
            name=name,
        )

    def run_materials(self, outdir="materials", name="out"):
        """Run "materials" script.

        This script generates a `vtu` file that shows the material.

        :param outdir: Working directory, defaults to "materials".
        :type outdir: str or pathlib.Path, optional
        :param name: System name, defaults to "out".
        :type name: str, optional.
        """
        outdir = check_dir(outdir)
        shutil.copyfile(self.mesh_path, outdir / f"{name}.fly")
        self.materials.write_krn(outdir / f"{name}.krn")

        self.run_script(
            script="materials",
            outdir=outdir,
            name=name,
        )
        self.materials_fields = meshio.read(outdir / f"{name}_mat.vtu")

    def run_store(self, outdir="magnetization", name="out"):
        """Run "store" script.

        The sparse matrices used for computation can be stored
        and reused for simulations with the same finite element mesh.

        :param outdir: Working directory, defaults to "magnetization".
        :type outdir: str or pathlib.Path, optional
        :param name: System name, defaults to "out".
        :type name: str, optional.
        """
        outdir = check_dir(outdir)
        shutil.copyfile(self.mesh_path, outdir / f"{name}.fly")
        self.materials.write_krn(outdir / f"{name}.krn")
        self.parameters.write_p2(outdir / f"{name}.p2")

        self.run_script(
            script="store",
            outdir=outdir,
            name=name,
        )


def run_subprocess(cmd, cwd):
    """Run command using `subprocess`.

    :param cmd: command to execute
    :type cmd: list
    :param cwd: working directory
    :type cwd: str or pathlib.Path
    :raises RuntimeError: Simulation has failed.
    """
    res = subprocess.run(
        cmd,
        cwd=cwd,
        stderr=subprocess.PIPE,
    )
    return_code = res.returncode

    if return_code:
        raise RuntimeError(
            "Simulation has failed. "
            "Exit with error: \n"
            f"{res.stderr.decode('utf-8')}"
        )
