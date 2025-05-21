"""Simulation class."""

import datetime
import json

import os
import shlex
import shutil
import subprocess

import pathlib
from pydantic import Field
from pydantic.dataclasses import dataclass

import mammos_mumag
from mammos_mumag.materials import MaterialDomain, Materials
from mammos_mumag.parameters import Parameters
from mammos_mumag.tools import check_dir, check_esys_escript

IS_POSIX = os.name == "posix"


@dataclass
class Simulation:
    """Simulation class.

    :param materials: class managing materials.
    :type materials: :py:class:`~mammos_mumag.materials.Materials`
    :param parameters: class managing parameters.
    :type parameters: :py:class:`~mammos_mumag.parameters.Parameters`
    """

    material_domain_list: list[MaterialDomain] | None = Field(default=None, repr=False)
    mesh_filepath: pathlib.Path | None = Field(default=None)
    materials_filepath: pathlib.Path | None = Field(default=None, repr=False)
    parameters_filepath: pathlib.Path | None = Field(default=None, repr=False)
    materials: Materials | None = Field(default=None)
    parameters: Parameters | None = Field(default=None)

    def __post_init__(self) -> None:
        """Post-initialization.

        Define `Materials` and `Parameters` instance if they have been defined.
        """
        if self.material_domain_list is not None:
            self.materials = Materials(domains=self.material_domain_list)
        elif self.materials_filepath is not None:
            self.materials = Materials(filepath=self.materials_filepath)
        if self.parameters_filepath is not None:
            self.parameters = Parameters(filepath=self.parameters_filepath)

    def check_attribute(self, *args) -> None:
        """Check existence of attributes.

        :raises AttributeError: Attribute has not been defined yet.
        """
        for attr in args:
            if self.__getattribute__(attr) is None:
                raise AttributeError(f"Attribute `{attr}` has not been defined yet.")

    @classmethod
    def run_file(
        cls, file: str | pathlib.Path, outdir: str | pathlib.Path = "out"
    ) -> None:
        """Run python file using `esys.escript`.

        :param script: path of file.
        :type script: str or pathlib.Path
        :param outdir: Working directory. Defaults to "out"
        :type outdir: str or pathlib.Path, optional
        """
        check_esys_escript()
        cmd = shlex.split(
            f"{mammos_mumag._run_escript_bin} {file}",
            posix=IS_POSIX,
        )
        run_subprocess(cmd, cwd=outdir)

    @classmethod
    def run_script(cls, script: str, outdir: str | pathlib.Path, name: str) -> None:
        """Run pre-defined script.

        :param script: Name of pre-defined script.
        :type script: str
        :param outdir: Working directory
        :type outdir: str or pathlib.Path
        :param name: System name
        :type name: str
        """
        check_esys_escript()
        cmd = shlex.split(
            f"{mammos_mumag._run_escript_bin} {mammos_mumag._scripts_directory / script}.py {name}",
            posix=IS_POSIX,
        )
        run_subprocess(cmd, cwd=outdir)
        with open(outdir / "info.json", "w") as file:
            json.dump(
                {
                    "datetime": datetime.datetime.now(datetime.UTC)
                    .astimezone()
                    .isoformat(timespec="seconds"),
                    "mammos_mumag_version": mammos_mumag.__version__,
                },
                file,
            )

    def run_exani(
        self,
        outdir: str | pathlib.Path = "exani",
        name: str = "out",
    ) -> None:
        r"""Run "exani" script.

        Test the computation of the exchange and anisotropy energy density.
        This gives the exchange energy density of a vortex in the :math:`xy`-plane
        and the anistropy energy density in the uniformly magnetized state.
        Here we have placed the anistropy direction paralle to to the :math:`z`-axis.
        The anisotropy energy density is calculated as :math:`-K (\mathbf{m} \cdot
        \mathbf{k})^2` where :math:`\mathbf{m}` is the unit vector of magnetization
        and :math:`\mathbf{k}` is the anisotropy direction. :math:`K` is the
        magnetocrystalline anisotropy constant.

        This scripts creates the following files in `outdir`:

        * `<name>.fly`: mesh file.

        * `<name>.krn`: materials file.

        * `<name>_uniform.csv`: table containing information about
          the exchange anisotropy energy evaluated with different
          methods on a uniformly  magnetized cube.

        * `<name>_vortex.csv`: table containing information about
          the exchange anisotropy energy evaluated with different
          methods on a vortex.

        :param outdir: Working directory, defaults to "exani".
        :type outdir: str or pathlib.Path, optional
        :param name: System name, defaults to "out".
        :type name: str, optional.
        """
        outdir = check_dir(outdir)
        self.check_attribute("mesh_filepath", "materials")
        shutil.copyfile(self.mesh_filepath, outdir / f"{name}.fly")
        self.materials.write_krn(outdir / f"{name}.krn")

        self.run_script(
            script="exani",
            outdir=outdir,
            name=name,
        )

    def run_external(
        self,
        outdir: str | pathlib.Path = "external",
        name: str = "out",
    ) -> None:
        r"""Run "external" script.

        Compute the Zeemann energy by finite elements and analytically.

        This scripts creates the following files in `outdir`:

        * `<name>.fly`: mesh file.

        * `<name>.krn`: materials file.

        * `<name>.p2`: simulation parameters file.

        * `<name>.csv`: table containing information about
          the Zeeman energy evaluated with different methods.

        :param outdir: Working directory, defaults to "external".
        :type outdir: str or pathlib.Path, optional
        :param name: System name, defaults to "out".
        :type name: str, optional.
        """
        outdir = check_dir(outdir)
        self.check_attribute("mesh_filepath", "materials", "parameters")
        shutil.copyfile(self.mesh_filepath, outdir / f"{name}.fly")
        self.materials.write_krn(outdir / f"{name}.krn")
        self.parameters.write_p2(outdir / f"{name}.p2")

        self.run_script(
            script="external",
            outdir=outdir,
            name=name,
        )

    def run_hmag(self, outdir: str | pathlib.Path = "hmag", name: str = "out") -> None:
        r"""Run "hmag" script.

        This script evaluates the magnetostatic energy density
        and the field of a uniformly magnetized geometry.
        Creates the `vtk` file for visualisation of the magnetic scalar potential
        and the magnetic field. With linear basis function for the magnetic scalar
        potential :math:`u`, the magnetostatic field :math:`h = -\nabla u` is
        defined at the finite elements. By smoothing the field can be transfered
        to the nodes of the finite element mesh.

        This scripts creates the following files in `outdir`:

        * `<name>.fly`: mesh file.

        * `<name>.krn`: materials file.

        * `<name>.csv`: table containing information about the magnetostatic
          energy density evaluated with different methods.
          Three energy values are compared:

          .. math::

            E_{\mathsf{field}} := - \frac{1}{2} \int_\Omega \frac{\mathbf{h} \cdot J_s
            \mathbf{m}}{V} \ \mathrm{d}x

          where :math:`\Omega` is the domain, :math:`\mathbf{h}` is the
          demagnetization field, :math:`J_s` is the spontaneous polarisation,
          :math:`\mathbf{m}` is the magnetization field, and :math:`V` is the volume
          of the domain.

          .. math::

            E_{\mathsf{gradient}} := \frac{1}{2} \sum_i \mathbf{m}_i \cdot \mathbf{g}_i

          where :math:`\mathbf{m}_i` and :math:`\mathbf{g}_i` are the unit vector of
          the magnetization and the gradient of the energy normalized by the volume
          of the energy with respect to :math:`\mathbf{m}_i` at the nodes of the finite
          element mesh.

          .. math::

            E_{\mathsf{analytic}} := J_s^2 / (6 \mu_0)

        :param outdir: Working directory, defaults to "hmag".
        :type outdir: str or pathlib.Path, optional
        :param name: System name, defaults to "out".
        :type name: str, optional.
        """
        outdir = check_dir(outdir)
        self.check_attribute("mesh_filepath", "materials")
        shutil.copyfile(self.mesh_filepath, outdir / f"{name}.fly")
        self.materials.write_krn(outdir / f"{name}.krn")

        self.run_script(
            script="hmag",
            outdir=outdir,
            name=name,
        )

    def run_loop(self, outdir: str | pathlib.Path = "loop", name: str = "out") -> None:
        r"""Run "loop" script.

        Compute demagnetization curves.

        This scripts creates the following files in `outdir`:

        * `<name>.fly`: mesh file.

        * `<name>.krn`: materials file.

        * `<name>.p2`: simulation parameters file.

        * `<name>_{i}.vtu`: saved `vtk` files. TODO

        * `<name>_stats.txt`: memory usage information.

        * `<name>.dat`: table data regarding the demagnetization curve.
          The columns of the file are:

          * the number of the `vtk` file that corresponds
            to the field and magnetic polarisation values in the line.

          * value of :math:`\mu_0 H_{\mathsf{ext}}` in Tesla, where :math:`\mu_0` is
            the permability of vacuum and :math:`H_{\mathsf{ext}}` is the external
            value of the external field.

          * the componenent of magnetic polarisation (in Tesla)
            parallel to the direction of the external field.

          * the energy density (:math:`\mathrm{J}/\mathrm{m}^3`) of the current state.

        :param outdir: Working directory, defaults to "loop".
        :type outdir: str or pathlib.Path, optional
        :param name: System name, defaults to "out".
        :type name: str, optional.
        """
        outdir = check_dir(outdir)
        self.check_attribute("mesh_filepath", "materials", "parameters")
        shutil.copyfile(self.mesh_filepath, outdir / f"{name}.fly")
        self.materials.write_krn(outdir / f"{name}.krn")
        self.parameters.write_p2(outdir / f"{name}.p2")

        self.run_script(
            script="loop",
            outdir=outdir,
            name=name,
        )

    def run_magnetization(
        self,
        outdir: str | pathlib.Path = "magnetization",
        name: str = "out",
    ) -> None:
        """Run "magnetization" script.

        Creates the `vtk` file for the visualisation of the material properties.

        :param outdir: Working directory, defaults to "magnetization".
        :type outdir: str or pathlib.Path, optional
        :param name: System name, defaults to "out".
        :type name: str, optional.
        """
        outdir = check_dir(outdir)
        self.check_attribute("mesh_filepath", "materials", "parameters")
        shutil.copyfile(self.mesh_filepath, outdir / f"{name}.fly")
        self.materials.write_krn(outdir / f"{name}.krn")
        self.parameters.write_p2(outdir / f"{name}.p2")

        self.run_script(
            script="magnetization",
            outdir=outdir,
            name=name,
        )

    def run_mapping(
        self,
        outdir: str | pathlib.Path = "magnetization",
        name: str = "out",
    ) -> None:
        """Run "mapping" script.

        Test the energy calculations with matrices.
        The module mapping.py contains the tools for mapping from the finite element
        bilinear forms to sparse matrices. We use sparse matrix methods from ``jax``.

        :param outdir: Working directory, defaults to "magnetization".
        :type outdir: str or pathlib.Path, optional
        :param name: System name, defaults to "out".
        :type name: str, optional.
        """
        outdir = check_dir(outdir)
        self.check_attribute("mesh_filepath", "materials", "parameters")
        shutil.copyfile(self.mesh_filepath, outdir / f"{name}.fly")
        self.materials.write_krn(outdir / f"{name}.krn")
        self.parameters.write_p2(outdir / f"{name}.p2")

        self.run_script(
            script="mapping",
            outdir=outdir,
            name=name,
        )

    def run_materials(
        self, outdir: str | pathlib.Path = "materials", name: str = "out"
    ) -> None:
        """Run "materials" script.

        This script generates a `vtu` file that shows the material.

        :param outdir: Working directory, defaults to "materials".
        :type outdir: str or pathlib.Path, optional
        :param name: System name, defaults to "out".
        :type name: str, optional.
        """
        outdir = check_dir(outdir)
        self.check_attribute("mesh_filepath", "materials")
        shutil.copyfile(self.mesh_filepath, outdir / f"{name}.fly")
        self.materials.write_krn(outdir / f"{name}.krn")

        self.run_script(
            script="materials",
            outdir=outdir,
            name=name,
        )

    def run_store(
        self, outdir: str | pathlib.Path = "magnetization", name: str = "out"
    ) -> None:
        """Run "store" script.

        The sparse matrices used for computation can be stored
        and reused for simulations with the same finite element mesh.

        :param outdir: Working directory, defaults to "magnetization".
        :type outdir: str or pathlib.Path, optional
        :param name: System name, defaults to "out".
        :type name: str, optional.
        """
        outdir = check_dir(outdir)
        self.check_attribute("mesh_filepath", "materials", "parameters")
        shutil.copyfile(self.mesh_filepath, outdir / f"{name}.fly")
        self.materials.write_krn(outdir / f"{name}.krn")
        self.parameters.write_p2(outdir / f"{name}.p2")

        self.run_script(
            script="store",
            outdir=outdir,
            name=name,
        )


def run_subprocess(cmd: list[str], cwd: str | pathlib.Path) -> None:
    """Run command using `subprocess` in the specified directory.

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
            f"Simulation has failed. Exit with error: \n{res.stderr.decode('utf-8')}"
        )
