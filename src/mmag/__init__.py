import importlib.metadata
import pathlib

_base_directory = pathlib.Path(__file__).absolute().parent
_container_scripts = _base_directory.joinpath("container_scripts")
_sim_scripts = _base_directory.joinpath("sim_scripts")
_cache_dir = pathlib.Path.home() / ".cache" / "mammos" / "mmag"
_conf_dir = pathlib.Path.home() / ".config" / "mammos" / "mmag"
__version__ = importlib.metadata.version(__package__)
