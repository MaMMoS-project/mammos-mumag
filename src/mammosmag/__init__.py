import importlib.metadata
import pathlib

_base_directory = pathlib.Path(__file__).absolute().parent.parent.parent
_container_scripts = _base_directory.joinpath("container_scripts")
_sim_scripts = _base_directory / 'src/mammosmag/scripts'
_cache_dir = pathlib.Path.home() / ".cache" / "mammosmag"
_conf_dir = pathlib.Path.home() / ".config" / "mammosmag"
__version__ = importlib.metadata.version(__package__)
