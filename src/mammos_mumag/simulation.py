"""Simulation class."""

import datetime
import json
import os
import pathlib
import shlex
import subprocess
from typing import Any

from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass

import mammos_mumag
from mammos_mumag.materials import MaterialDomain, Materials
from mammos_mumag.mesh import Mesh
from mammos_mumag.parameters import Parameters

IS_POSIX = os.name == "posix"


@dataclass(config={"arbitrary_types_allowed": True})
class Simulation:
    """Simulation class.

    Args:
        materials: :py:class:`~mammos_mumag.materials.Materials` instance containing
            information about the material.
        material_domain_list: List of :py:class:`~mammos_mumag.materials.MaterialDomain`
            objects. Each object contains the intrinsic properties in any uniform
            subdomain. If specified, this material information overwrites the
            :py:attr:`~mammos_mumag.simulation.Simulation.materials` attribute.
        materials_filepath: Location of materials file to read. If specified, the
            material parameters read from file will overwrite any material information
            defined via the :py:attr:`~mammos_mumag.simulation.Simulation.materials` or
            :py:attr:`~mammos_mumag.simulation.Simulation.material_domain_list`
            attributes.
        mesh: Mesh object.
        parameters: :py:class:`~mammos_mumag.parameters.Parameters` instance containing
            information about simulation parameters.
        parameters_filepath: Location of parameter file to read. If specified, all the
            parameters stored in the
            :py:attr:`~mammos_mumag.simulation.Simulation.parameters` attribute will be
            overwritten.
    """

    mesh: mammos_mumag.mesh.Mesh
    material_domain_list: list[MaterialDomain] | None = Field(default=None, repr=False)
    materials_filepath: pathlib.Path | None = Field(default=None, repr=False)
    parameters_filepath: pathlib.Path | None = Field(default=None, repr=False)
    materials: Materials | None = Field(default=None)
    parameters: Parameters | None = Field(default=None)

    @field_validator("mesh", mode="before")
    @classmethod
    def _convert_mesh(cls, mesh: Any) -> Any:
        """Convert  string or path to local Mesh instance."""
        if isinstance(mesh, str | pathlib.Path):
            mesh = Mesh(mesh)
        return mesh

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

        Args:
            *args: Attribtes to check.

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

    def run_loop(self, outdir: str | pathlib.Path = "loop", name: str = "out") -> None:
        r"""Run "loop" script.

        Compute demagnetization curves.

        This scripts creates the following files in `outdir`:

        * `<name>.med`: mesh file in med format.

        * `<name>.npz`: mesh file in npz format.

        * `<name>.krn`: materials file.

        * `<name>.p2`: simulation parameters file.

        * `<name>_{i}.vtu`: saved configurations. The amount of configurations stored
          depends on the simulation parameter
          :py:attr:`~mammos_mumag.parameters.Parameters.m_step`.

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

        Args:
            outdir: Working directory.
            name: System name.

        """
        outdir = pathlib.Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)

        # check inputs
        self.check_attribute("mesh", "materials", "parameters")
        self.check_numgrains()

        # copy input files into `outdir`
        self.mesh._write_npz(outdir / f"{name}")
        self.materials.write_krn(outdir / f"{name}.krn")
        self.parameters.write_p2(outdir / f"{name}.p2")

        # call subprocess with loop script
        loop_script = pathlib.Path(__file__).parent / "src" / "loop.py"
        cmd = shlex.split(f"python {loop_script} --mesh {name}.npz", posix=IS_POSIX)
        res = subprocess.run(
            cmd,
            cwd=outdir,
            stderr=subprocess.PIPE,
        )
        return_code = res.returncode
        if return_code:
            raise RuntimeError(
                f"Simulation has failed with error: \n{res.stderr.decode('utf-8')}"
            )

        # write run info
        with open(outdir / "info.json", "w") as f:
            json.dump(
                {
                    "datetime": datetime.datetime.now(datetime.UTC)
                    .astimezone()
                    .isoformat(timespec="seconds"),
                    "mammos_mumag_version": mammos_mumag.__version__,
                },
                f,
            )
