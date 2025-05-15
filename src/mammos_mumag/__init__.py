"""mammos_mumag.

MaMMoS micromagnetic simulation software.
"""

import importlib.metadata
import pathlib
import shutil

from mammos_mumag import postprocess
from mammos_mumag.standard_problem import hystloop

__all__ = [
    "hystloop",
    "postprocess",
]

_run_escript_bin = shutil.which("run-escript")
_scripts_directory = pathlib.Path(__file__).parent.resolve() / "scripts"
__version__ = importlib.metadata.version(__package__)
