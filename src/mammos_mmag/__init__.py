"""mammos_mmag.

MaMMoS micromagnetic simulation software.
"""

import importlib.metadata
import pathlib



_base_directory = pathlib.Path(__file__).absolute().parent
_conf_dir = pathlib.Path.home() / ".config" / "mammos" / "mmag"
__version__ = importlib.metadata.version(__package__)
