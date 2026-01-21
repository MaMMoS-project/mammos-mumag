"""Mesh submodule.

This module reads meshes in the ``.fly`` format. It is possible to generate a mesh
in such format from a ``.unv`` mesh by using the ``unv2fly`` command line tool or
using the Python function :py:func:`~mammos_mumag.tofly.convert`.

Meshes also have a specific geometry, following the shell-trasformation method.
Regardless of the material shape, it is further enclosed by a sphere and
a spherical shell. The volume between the magnetic material and the
firs sphere represents a non-magnetic material (for example, air), while the volume
between the two spheres is a projection to infinity necessary to solve the
demagnetization problem.

For more information, see Imhoff, J. F., et al. “An original solution for unbounded
electromagnetic 2D-and 3D-problems throughout the finite element method.” IEEE
Transactions on Magnetics 26.5 (1990): 1659-1661.

In practical term, this means that the mesh must have tags to divide it in different
regions:

* one or more regions (or domains) of magnetic material;
* the second to last domain is the non-magnetic material between the magnetic material
  and the smaller spherical surface;
* the last domain is also non-magnetic and marks the volume between the two spherical
  shells.

The content of the tags does not matter. What effectively matters is that the
non-magnetic domains are placed last.

"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING
from warnings import warn

import requests
import urllib3

if TYPE_CHECKING:
    import os


def _get_mesh_json() -> dict:
    """Load mesh JSON file.

    The JSON file contains:

    * metadata regarding all meshes (the Zenodo record URL and the backup
      Keeper repository URL);
    * minimal information for the meshes, such as the geometry, a ``length``
      parameter with information about scale, the number of domains and
      numerical information about the mesh.

    """
    with open(Path(__file__).parent / "mesh" / "README.json") as f:
        return json.load(f)


def find_mesh(mesh_name: str | None = None) -> list[str]:
    """Find available meshes containing the given string.

    Args:
        mesh_name: String to be searched. If ``""`` or not given, returns all
            available meshes.

    Returns:
        List of mesh names containing the input string. Empty list if no matches are
        found.

    Examples:
        >>> from mammos_mumag.mesh import find_mesh
        >>> find_mesh("cube50")
        ['cube50_singlegrain_msize1', 'cube50_singlegrain_msize2']
    """
    meshes = _get_mesh_json()["meshes"]
    if mesh_name is None:
        return list(meshes.keys())
    else:
        return [mm for mm in meshes if mesh_name in mm]


class Mesh:
    """Mesh class.

    This class supports both local mesh (i.e. found on disk and defined by the user) and
    remote meshes stored on Zenodo. The Zenodo URL is found in the ``README.json`` and a
    list of all available meshes is found using :py:func:`~mammos_mumag.mesh.find_mesh`.

    Attributes:
        name:

            * If local mesh, path to the file.
            * If remote mesh, its name is as it appears on the Zenodo record or in the
              ``README.json``.

        info: dictionary of available information about the mesh. If the mesh is local,
            this dictionary is initialized with the couple
            `"description": "User defined mesh."`.

    """

    def __init__(self, mesh_name: str | os.PathLike):
        """Initialize Mesh.

        * The input ``mesh_name`` is initially understood as a location and tried for
          matches on disk. If such file exists, the mesh is intended as local.
        * Otherwise, ``mesh_name`` is given to :py:func:`~mammos_mumag.mesh.find_mesh`
          for matches in the Zenodo record. If multiple matches are found, the
          initialization will return an error.

        Args:
            mesh_name: Location of local mesh or name of remote mesh.

        Raises:
            RuntimeError: Multiple matches found in the Zenodo record.
            RuntimeError: No match found, either local or remote.
        """
        if Path(mesh_name).is_file():
            self.name = mesh_name
            self.info = {"description": "User defined mesh."}
            self._local = True
            self._path = Path(mesh_name)
        else:
            matches = find_mesh(mesh_name)
            if len(matches) == 0:
                raise RuntimeError(
                    f"No local or remote matches found with name: {mesh_name}"
                )
            elif len(matches) > 1:
                raise RuntimeError(
                    f"Mesh name ambiguous. More than one match found: {matches}"
                )
            else:
                mesh_json = _get_mesh_json()
                self.name = matches[0]
                self.info = mesh_json["meshes"][self.name]
                self._local = (
                    _path := Path(__file__).parent / "mesh" / f"{self.name}.fly"
                ).is_file()
                if self._local:
                    self._path = _path
                self._url = (
                    f"{mesh_json['metadata']['zenodo_url']}/files/{self.name}.fly"
                )

    def __str__(self) -> str:
        """Implement str dunder."""
        s = f"Mesh: {self.name}\n"
        for k, v in self.info.items():
            s += f"{k}: {v}\n"
        return s

    def __repr__(self) -> str:
        """Implement repr dunder."""
        return f"Mesh('{self.name}')"

    def write(self, dest: os.PathLike | str) -> None:
        """Write mesh to destination."""
        dest = Path(dest).resolve()
        if dest.suffix != ".fly":
            raise ValueError("Only `.fly` meshes are available.")
        if self._local:
            shutil.copy(self._path, dest)
        else:
            s = requests.Session()
            retries = urllib3.util.Retry(
                total=3,
                backoff_factor=0.1,
                status_forcelist=[500, 502, 503, 504],
            )
            s.mount("https://", requests.adapters.HTTPAdapter(max_retries=retries))
            res = s.get(self._url)
            if res.status_code != 200:
                warn(
                    "Unable to download mesh from Zenodo. "
                    f"The request returned with HTTP code: {res.status_code}. "
                    "Downloading the mesh from keeper.",
                    stacklevel=1,
                )
                # Keeper works reliably when downloading 1000 fly mesh in parallel!
                # Think about completely replacing Zenodo downloads with Keeper.
                self._write_from_keeper(dest)
            else:
                with open(dest, "wb") as f:
                    f.write(res.content)

    def _write_from_keeper(self, dest: os.PathLike | str) -> None:
        """Write mesh to destination.

        Load mesh from Keeper rather than from Zenodo.
        This functions is only for developers and should not be used.
        """
        dest = Path(dest).resolve()
        avail_fmts = [".fly", ".med", ".unv"]
        if dest.suffix not in avail_fmts:
            raise ValueError(f"Wrong format. Available formats: {avail_fmts}")
        keeper_url = _get_mesh_json()["metadata"]["keeper_url"]
        mesh_url = f"{keeper_url}files/?p=/{self.name}/mesh{dest.suffix}&dl=1"
        s = requests.Session()
        retries = urllib3.util.Retry(
            total=3,
            backoff_factor=0.1,
            status_forcelist=[500, 502, 503, 504],
        )
        s.mount("https://", requests.adapters.HTTPAdapter(max_retries=retries))
        res = s.get(mesh_url)
        with open(dest, "wb") as f:
            f.write(res.content)


def _get_mesh_json_from_keeper() -> dict:
    """Download mesh.json from Keeper and return dictionary."""
    keeper_url = _get_mesh_json()["metadata"]["keeper_url"]
    res = requests.get(f"{keeper_url}files/?p=/README.json&dl=1")
    if res.status_code != 200:
        raise FileNotFoundError("README.json not found on Keeper.")
    else:
        return json.loads(res.content)
