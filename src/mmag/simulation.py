"""Simulation class."""

import json
import os
import pathlib
import shlex
import shutil
import subprocess

from .materials import Materials
from .parameters import Parameters
import mmag


class Simulation:
    """Simulation class."""

    def __init__(self):
        """Initialize class."""
        self.parameters = Parameters()
        self.materials = Materials()

        with open(mmag._conf_dir / "conf.json", "r") as handle:
            conf_dict = json.load(handle)
        self._escript_bin = list(conf_dict["run_escript"].values())[0]

    def run_file(self, file, outdir="out", threads=4):
        """Run file using `esys.escript`.

        :param script: path of file.
        :type script: str or pathlib.Path
        :param outdir: Working directory. Defaults to "out"
        :type outdir: str or pathlib.Path, optional
        :param threads: Number of execution threads. Defaults to 4.
        :type threads: int, optional
        """
        is_posix = os.name == "posix"
        cmd = shlex.split(
            (
                f"{self._escript_bin} "
                f"-t{threads} "
                f"{file}"
            ),
            posix=is_posix,
        )
        run_subprocess(cmd, cwd=outdir)

    def run_script(self, script, outdir, name, threads):
        """Run pre-defined script.

        :param script: Name of pre-defined script.
        :type script: str
        :param outdir: Working directory
        :type outdir: str or pathlib.Path
        :param name: System name
        :type name: str
        :param threads: Number of execution threads
        :type threads: int
        """
        is_posix = os.name == "posix"
        cmd = shlex.split(
            (
                f"{self._escript_bin} "
                f"-t{threads} "
                f"/scripts/{script}.py "
                f"{name}"
            ),
            posix=is_posix,
        )
        run_subprocess(cmd, cwd=outdir)

    def run_exani(self, threads=4, outdir="exani", name="out"):
        """Run "exani" script.

        TODO: Get more info

        :param threads: Number of execution threads, defaults to 4.
        :type threads: int, optional
        :param outdir: Working directory, defaults to "exani".
        :type outdir: str or pathlib.Path, optional
        :param name: System name, defaults to "out".
        :type name: str, optional.
        """
        outdir = pathlib.Path(outdir)
        outdir.mkdir(exist_ok=True)
        shutil.copyfile(self.mesh_path, outdir / f"{name}.fly")
        self.materials.write_krn(outdir / f"{name}.krn")

        self.run_script(
            script="exani",
            outdir=outdir,
            name=name,
            threads=threads,
        )

    def run_external(self, threads=4, outdir="external", name="out"):
        """Run "external" script.

        TODO: Get more info

        :param threads: Number of execution threads, defaults to 4.
        :type threads: int, optional
        :param outdir: Working directory, defaults to "external".
        :type outdir: str or pathlib.Path, optional
        :param name: System name, defaults to "out".
        :type name: str, optional.
        """
        outdir = pathlib.Path(outdir)
        outdir.mkdir(exist_ok=True)
        shutil.copyfile(self.mesh_path, outdir / f"{name}.fly")
        self.materials.write_krn(outdir / f"{name}.krn")
        self.parameters.write_p2(outdir / f"{name}.p2")

        self.run_script(
            script="external",
            outdir=outdir,
            name=name,
            threads=threads,
        )

    def run_hmag(self, threads=4, outdir="hmag", name="out"):
        """Run "hmag" script.

        TODO: Get more info

        :param threads: Number of execution threads, defaults to 4.
        :type threads: int, optional
        :param outdir: Working directory, defaults to "hmag".
        :type outdir: str or pathlib.Path, optional
        :param name: System name, defaults to "out".
        :type name: str, optional.
        """
        outdir = pathlib.Path(outdir)
        outdir.mkdir(exist_ok=True)
        shutil.copyfile(self.mesh_path, outdir / f"{name}.fly")
        self.materials.write_krn(outdir / f"{name}.krn")

        self.run_script(
            script="hmag",
            outdir=outdir,
            name=name,
            threads=threads,
        )

    def run_loop(self, threads=4, outdir="loop", name="out"):
        """Run "loop" script.

        TODO: Get more info

        :param threads: Number of execution threads, defaults to 4.
        :type threads: int, optional
        :param outdir: Working directory, defaults to "loop".
        :type outdir: str or pathlib.Path, optional
        :param name: System name, defaults to "out".
        :type name: str, optional.
        """
        outdir = pathlib.Path(outdir)
        outdir.mkdir(exist_ok=True)
        shutil.copyfile(self.mesh_path, outdir / f"{name}.fly")
        self.materials.write_krn(outdir / f"{name}.krn")
        self.parameters.write_p2(outdir / f"{name}.p2")

        self.run_script(
            script="loop",
            outdir=outdir,
            name=name,
            threads=threads,
        )

    def run_magnetization(self, threads=4, outdir="magnetization", name="out"):
        """Run "magnetization" script.

        TODO: Get more info

        :param threads: Number of execution threads, defaults to 4.
        :type threads: int, optional
        :param outdir: Working directory, defaults to "magnetization".
        :type outdir: str or pathlib.Path, optional
        :param name: System name, defaults to "out".
        :type name: str, optional.
        """
        outdir = pathlib.Path(outdir)
        outdir.mkdir(exist_ok=True)
        shutil.copyfile(self.mesh_path, outdir / f"{name}.fly")
        self.materials.write_krn(outdir / f"{name}.krn")
        self.parameters.write_p2(outdir / f"{name}.p2")

        self.run_script(
            script="magnetization",
            outdir=outdir,
            name=name,
            threads=threads,
        )

    def run_materials(self, threads=4, outdir="materials", name="out"):
        """Run "materials" script.

        TODO: Get more info

        :param threads: Number of execution threads, defaults to 4.
        :type threads: int, optional
        :param outdir: Working directory, defaults to "materials".
        :type outdir: str or pathlib.Path, optional
        :param name: System name, defaults to "out".
        :type name: str, optional.
        """
        outdir = pathlib.Path(outdir)
        outdir.mkdir(exist_ok=True)
        shutil.copyfile(self.mesh_path, outdir / f"{name}.fly")
        self.materials.write_krn(outdir / f"{name}.krn")

        self.run_script(
            script="materials",
            outdir=outdir,
            name=name,
            threads=threads,
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
