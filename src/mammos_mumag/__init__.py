"""Finite-element micromagnetic simulations (hysteresis)."""

import importlib.metadata
import pathlib
import shutil

from mammos_mumag import (
    hysteresis,
    materials,
    mesh,
    parameters,
    simulation,
    tofly,
)

__all__ = [
    "hysteresis",
    "materials",
    "mesh",
    "parameters",
    "simulation",
    "tofly",
]

_run_escript_bin = shutil.which("run-escript")
_scripts_directory = pathlib.Path(__file__).parent.resolve() / "scripts"
__version__ = importlib.metadata.version(__package__)
