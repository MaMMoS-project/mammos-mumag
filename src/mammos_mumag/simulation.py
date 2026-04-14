"""Simulation class."""

import datetime
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass

import mammos_mumag
from mammos_mumag._tools import check_esys_escript
from mammos_mumag.materials import MaterialDomain, Materials
from mammos_mumag.mesh import Mesh
from mammos_mumag.parameters import Parameters

IS_POSIX = os.name == "posix"


@dataclass(config={"arbitrary_types_allowed": True})
class Simulation:
    """Simulation class.

    Three objects are needed in most simulations:

    * a mesh describing the material's geometry,
    * a :py:class:`~mammos_mumag.materials.Materials` instance for the magnetic
      properties of the different domains,
    * a :py:class:`~mammos_mumag.parameters.Parameters` instance for the simulation
      and optimization parameters.

    However, in some cases we are interested in simulations that do not require one or
    more of the objects. For example, evaluating a hysteresis loop using the script
    ``"loop"`` needs all three objects, while the script ``"materials"`` (used for
    outputting a material ``.vtu`` file) only requires the mesh and the materials
    objects.

    Therefore, only :py:attr:`~mammos_mumag.simulation.Simulation.mesh` is a required
    attribute and the other attributes are optional. When executing a script, the
    first step will be checking that the required simulation attributes are defined.
    """

    mesh: mammos_mumag.mesh.Mesh
    """Mesh of the material. The mesh can either be given as a
    :py:class:`~mammos_mumag.mesh.Mesh` instance, as a string for Zenodo meshes
    available through :py:mod:`mammos_mumag`, or as a path
    for local meshes. The only
    recognized mesh format is ``.fly``."""
    materials: Materials | None = Field(default=None)
    """Materials parameters."""
    material_domain_list: list[MaterialDomain] | None = Field(default=None, repr=False)
    """List of :py:class:`~mammos_mumag.materials.MaterialDomain` objects. Each object
    contains the intrinsic properties in any uniform subdomain. If specified, this
    material information overwrites the
    :py:attr:`~mammos_mumag.simulation.Simulation.materials` attribute.
    However, if the attribute
    :py:attr:`~mammos_mumag.simulation.Simulation.materials_filepath`
    is also defined, it will have the priority."""
    materials_filepath: os.PathLike | str | None = Field(default=None, repr=False)
    """Location of materials file to read. If specified, the material parameters read
    from file will overwrite any material information defined via the
    :py:attr:`~mammos_mumag.simulation.Simulation.materials` or
    :py:attr:`~mammos_mumag.simulation.Simulation.material_domain_list` attributes."""
    parameters: Parameters | None = Field(default=None)
    """Simulation parameters."""
    parameters_filepath: os.PathLike | str | None = Field(default=None, repr=False)
    """Location of parameter file to read. If specified, all the parameters stored in
    the :py:attr:`~mammos_mumag.simulation.Simulation.parameters` attribute will be
    overwritten."""

    @field_validator("mesh", mode="before")
    @classmethod
    def _convert_mesh(cls, mesh: Any) -> Any:
        """Convert  string or path to local Mesh instance."""
        if isinstance(mesh, str | os.PathLike):
            mesh = Mesh(mesh)
        return mesh

    def __post_init__(self) -> None:
        """Post-initialization.

        Define `Materials` and `Parameters` instance if they have been defined.

        Reading from files has the precedence over the other attributes. The hierarchy
        for assigning :py:attr:`~mammos_mumag.simulation.Simulation.materials` is:

        * If the attribute :py:attr:`~mammos_mumag.simulation.Simulation.
          materials_filepath` is not ``None``, the materials file is read.
        * If the attribute :py:attr:`~mammos_mumag.simulation.Simulation.
          materials_domain_list` is defined instead, the materials object is assembled
          from the domains.
        * If the other attributes are not set, :py:attr:`~mammos_mumag.simulation.
          Simulation.materials` keeps its assigned value.

        """
        if self.material_domain_list is not None:
            self.materials = Materials(domains=self.material_domain_list)
        elif self.materials_filepath is not None:
            self.materials = Materials(filepath=self.materials_filepath)
        if self.parameters_filepath is not None:
            self.parameters = Parameters(filepath=self.parameters_filepath)

    def check_attribute(self, *args) -> None:
        """Check existence of the attributes.

        Args:
            *args: Attributes to check.

        Raises:
            AttributeError: Attribute has not been defined yet.

        """
        for attr in args:
            if self.__getattribute__(attr) is None:
                raise AttributeError(f"Attribute `{attr}` has not been defined yet.")

    def check_numgrains(self) -> None:
        """Check that the number of grains match for mesh and material class."""
        if (
            "domains" in self.mesh.info
            and len(self.materials.domains) != self.mesh.info["domains"] + 2
        ):
            raise ValueError("Mesh and domains have a different number of grains.")

    @classmethod
    def run_file(
        cls, file: str | os.PathLike, outdir: str | os.PathLike = "out"
    ) -> None:
        """Run python file using ``esys.escript``.

        Args:
            file: Path to simulation script.
            outdir: Working directory.

        """
        check_esys_escript()
        cmd = shlex.split(
            f"{mammos_mumag._run_escript_bin} {file}",
            posix=IS_POSIX,
        )
        _run_subprocess(cmd, cwd=outdir)

    @classmethod
    def _run_script(cls, script: str, outdir: str | os.PathLike, name: str) -> None:
        """Run pre-defined script.

        Args:
            script: Name of pre-defined script.
            outdir: Working directory
            name: System name. This will appear in the output files, if present.

        """
        check_esys_escript()
        cmd = shlex.split(
            f"{mammos_mumag._run_escript_bin} "
            f"{mammos_mumag._scripts_directory / script}.py {name}",
            posix=IS_POSIX,
        )
        _run_subprocess(cmd, cwd=outdir)
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
        outdir: str | os.PathLike = "exani",
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

        This scripts creates the following files in ``outdir``:

        * ``<name>.fly``: mesh file.
        * ``<name>.krn``: materials file.
        * ``<name>_uniform.csv``: table containing information about
          the exchange anisotropy energy evaluated with different
          methods on a uniformly  magnetized cube.
        * ``<name>_vortex.csv``: table containing information about
          the exchange anisotropy energy evaluated with different
          methods on a vortex.

        Args:
            outdir: Working directory.
            name: System name.

        """
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)
        self.check_attribute("mesh", "materials")
        self.check_numgrains()
        self.mesh.write(outdir / f"{name}.fly")
        self.materials.write_krn(outdir / f"{name}.krn")

        self._run_script(
            script="exani",
            outdir=outdir,
            name=name,
        )

    def run_external(
        self,
        outdir: str | os.PathLike = "external",
        name: str = "out",
    ) -> None:
        """Run "external" script.

        Compute the Zeemann energy by finite elements and analytically.

        This scripts creates the following files in `outdir`:

        * ``<name>.fly``: mesh file.
        * ``<name>.krn``: materials file.
        * ``<name>.p2``: simulation parameters file.
        * ``<name>.csv``: table containing information about
          the Zeeman energy evaluated with different methods.

        Args:
            outdir: Working directory.
            name: System name.

        """
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)
        self.check_attribute("mesh", "materials", "parameters")
        self.check_numgrains()
        self.mesh.write(outdir / f"{name}.fly")
        self.materials.write_krn(outdir / f"{name}.krn")
        self.parameters.write_p2(outdir / f"{name}.p2")

        self._run_script(
            script="external",
            outdir=outdir,
            name=name,
        )

    def run_hmag(self, outdir: str | os.PathLike = "hmag", name: str = "out") -> None:
        r"""Run "hmag" script.

        This script evaluates the magnetostatic energy density
        and the field of a uniformly magnetized geometry.
        Creates the `vtk` file for visualisation of the magnetic scalar potential
        and the magnetic field. With linear basis function for the magnetic scalar
        potential :math:`u`, the magnetostatic field :math:`h = -\nabla u` is
        defined at the finite elements. By smoothing the field can be transfered
        to the nodes of the finite element mesh.

        This scripts creates the following files in `outdir`:

        * ``<name>.fly``: mesh file.
        * ``<name>.krn``: materials file.
        * ``<name>.csv``: table containing information about the magnetostatic
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

        Args:
            outdir: Working directory.
            name: System name.

        """
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)
        self.check_attribute("mesh", "materials")
        self.check_numgrains()
        self.mesh.write(outdir / f"{name}.fly")
        self.materials.write_krn(outdir / f"{name}.krn")

        self._run_script(
            script="hmag",
            outdir=outdir,
            name=name,
        )

    def run_loop(self, outdir: str | os.PathLike = "loop", name: str = "out") -> None:
        r"""Run "loop" script.

        Compute demagnetization curves.

        This scripts creates the following files in `outdir`:

        * ``<name>.fly``: mesh file.
        * ``<name>.krn``: materials file.
        * ``<name>.p2``: simulation parameters file.
        * ``<name>_{i}.vtu``: saved configurations. The amount of configurations stored
          depends on the simulation parameter
          :py:attr:`~mammos_mumag.parameters.Parameters.m_step`.
        * ``<name>_stats.txt``: memory usage information.
        * ``<name>.dat``: table data regarding the demagnetization curve.
          The columns of the file are:

          * the number of the `vtk` file that corresponds
            to the field and magnetic polarisation values in the line.
          * value of :math:`\mu_0 H_{\mathsf{ext}}` in Tesla, where :math:`\mu_0` is
            the permability of vacuum and :math:`H_{\mathsf{ext}}` is the external
            value of the external field.
          * the componenent of magnetic polarisation (in Tesla)
            parallel to the direction of the external field.
          * the energy density (:math:`\mathrm{J}/\mathrm{m}^3`) of the current state.

        Args:
            outdir: Working directory.
            name: System name.

        """
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)
        self.check_attribute("mesh", "materials", "parameters")
        self.check_numgrains()
        self.mesh.write(outdir / f"{name}.fly")
        self.materials.write_krn(outdir / f"{name}.krn")
        self.parameters.write_p2(outdir / f"{name}.p2")

        self._run_script(
            script="loop",
            outdir=outdir,
            name=name,
        )

    def run_magnetization(
        self,
        outdir: str | os.PathLike = "magnetization",
        name: str = "out",
    ) -> None:
        """Run "magnetization" script.

        Creates the ``vtk`` file for the visualisation of the material properties.

        Args:
            outdir: Working directory.
            name: System name.

        """
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)
        self.check_attribute("mesh", "materials", "parameters")
        self.check_numgrains()
        self.mesh.write(outdir / f"{name}.fly")
        self.materials.write_krn(outdir / f"{name}.krn")
        self.parameters.write_p2(outdir / f"{name}.p2")

        self._run_script(
            script="magnetization",
            outdir=outdir,
            name=name,
        )

    def run_mapping(
        self,
        outdir: str | os.PathLike = "mapping",
        name: str = "out",
    ) -> None:
        """Run "mapping" script.

        Test the energy calculations with matrices.
        The module mapping.py contains the tools for mapping from the finite element
        bilinear forms to sparse matrices. We use sparse matrix methods from ``jax``.

        Args:
            outdir: Working directory.
            name: System name.

        """
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)
        self.check_attribute("mesh", "materials", "parameters")
        self.check_numgrains()
        self.mesh.write(outdir / f"{name}.fly")
        self.materials.write_krn(outdir / f"{name}.krn")
        self.parameters.write_p2(outdir / f"{name}.p2")

        self._run_script(
            script="mapping",
            outdir=outdir,
            name=name,
        )

    def run_materials(
        self, outdir: str | os.PathLike = "materials", name: str = "out"
    ) -> None:
        """Run "materials" script.

        This script generates a ``vtu`` file that shows the material.

        Args:
            outdir: Working directory.
            name: System name.

        """
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)
        self.check_attribute("mesh", "materials")
        self.check_numgrains()
        self.mesh.write(outdir / f"{name}.fly")
        self.materials.write_krn(outdir / f"{name}.krn")

        self._run_script(
            script="materials",
            outdir=outdir,
            name=name,
        )

    def run_store(self, outdir: str | os.PathLike = "store", name: str = "out") -> None:
        """Run "store" script.

        The sparse matrices used for computation can be stored
        and reused for simulations with the same finite element mesh.

        Args:
            outdir: Working directory.
            name: System name.

        """
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)
        self.check_attribute("mesh", "materials", "parameters")
        self.check_numgrains()
        self.mesh.write(outdir / f"{name}.fly")
        self.materials.write_krn(outdir / f"{name}.krn")
        self.parameters.write_p2(outdir / f"{name}.p2")

        self._run_script(
            script="store",
            outdir=outdir,
            name=name,
        )


def _run_subprocess(cmd: list[str], cwd: str | os.PathLike) -> None:
    """Run command using `subprocess` in the specified directory.

    Args:
        cmd: command to execute
        cwd: working directory

    Raises:
        RuntimeError: Simulation has failed.

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
