"""Mesh functions."""

from enum import Enum
import pathlib

MESH_DIR = pathlib.Path(__file__).parent / "mesh"


class Mesh(Enum):
    """Class collecting available meshes."""

    CUBE_20_nm: pathlib.Path = MESH_DIR / "CUBE_20_nm.fly"
