"""Mesh functions."""

import json
import os
import pathlib
import shutil
from warnings import warn

import meshio as mio
import numpy as np
import requests
import urllib3
from scipy.spatial import KDTree


def _get_mesh_json():
    """Load mesh JSON file."""
    with open(pathlib.Path(__file__).parent / "mesh" / "README.json") as f:
        return json.load(f)


def find_mesh(mesh_name: str | None = None) -> list[str]:
    """Find available meshes matching given name.

    Args:
        mesh_name: Desired mesh name. If None, returns all available meshes.

    Returns:
        List of matches with given name. Empty list if no matches are found.
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

    def __init__(self, mesh_name: str | pathlib.Path):
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
        if pathlib.Path(mesh_name).is_file():
            self.name = mesh_name
            self.info = {"description": "User defined mesh."}
            self._local = True
            self._path = pathlib.Path(mesh_name)
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
                    _path := pathlib.Path(__file__).parent / "mesh" / f"{self.name}.fly"
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

    def write(self, dest: pathlib.Path | str) -> None:
        """Write mesh to destination."""
        dest = pathlib.Path(dest).resolve()
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

    def _write_from_keeper(self, dest: pathlib.Path | str) -> None:
        """Write mesh to destination.

        Load mesh from Keeper rather than from Zenodo.
        This functions is only for developers and should not be used.
        """
        dest = pathlib.Path(dest).resolve()
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

    def _write_npz(self, dest: pathlib.Path | str) -> None:
        """Write mesh in npz format."""
        mesh_path = pathlib.Path(dest).with_suffix(".med")
        self._write_from_keeper(mesh_path)
        _med2npz(pathlib.Path(mesh_path))


def _get_mesh_json_from_keeper() -> dict:
    """Download mesh.json from Keeper and return dictionary."""
    keeper_url = _get_mesh_json()["metadata"]["keeper_url"]
    res = requests.get(f"{keeper_url}files/?p=/README.json&dl=1")
    if res.status_code != 200:
        raise FileNotFoundError("README.json not found on Keeper.")
    else:
        return json.loads(res.content)


def _med2npz(mesh_path: os.PathLike) -> None:
    """Convert med mesh to npz.

    Swapneel's function.
    """
    mesh = mio.read(mesh_path)
    points = mesh.points

    # Remove non-tetrahedral cells
    for i, cell_block in enumerate(mesh.cells):
        if cell_block.type == "tetra":
            required_cell_block = cell_block
            required_cell_data = mesh.cell_data["cell_tags"][i]
            break

    cell_tags = mesh.cell_tags

    # Create new mesh with only tetra cells and tags as cell-sets
    new_mesh = mio.Mesh(
        points=points,
        cells=[required_cell_block],
        cell_data={
            "cell_tags": [np.array([int(cell_tags[i][0]) for i in required_cell_data])]
        },
    )
    new_mesh.unique_tag_values = np.unique(new_mesh.cell_data["cell_tags"][0])
    new_mesh.cell_data_to_sets("cell_tags")

    required_cells = []

    # Remove the air and shell subregions
    for key, val in new_mesh.cell_sets.items():
        if key not in {
            f"set-cell_tags-{new_mesh.unique_tag_values[-1]}",
            f"set-cell_tags-{new_mesh.unique_tag_values[-2]}",
        }:
            required_cells.append(val[0])

    # Select required cells and points without air and shell
    required_cells_arr = np.concatenate(required_cells)
    required_points_arr = np.unique(
        new_mesh.cells[0].data[required_cells_arr].flatten()
    )

    # Going back to cell tags as cell-data instead of cell sets
    new_mesh.cell_sets_to_data(data_name="cell_tags")
    new_mesh.cell_data["cell_tags"][0] += 1

    # Extract required point coordinates and cell connectivity
    # in terms of point coordinates
    new_points = new_mesh.points[required_points_arr]
    new_connectivity_as_points = new_mesh.points[
        new_mesh.cells[0].data[required_cells_arr]
    ]
    # Create tree for re-evaluation of connectivity indices
    tree = KDTree(new_points)

    # Find the new indices
    dist, new_connectivity = tree.query(new_connectivity_as_points)
    if not np.all(dist == 0.0):
        raise RuntimeError("The new connectivity might be wrong!")

    # Create the connectivity array according to Tom's conventions
    new_ijk = np.empty((new_connectivity.shape[0], 5), dtype=np.int_)
    new_ijk[:, 0:-1] = new_connectivity
    new_ijk[:, -1] = new_mesh.cell_data["cell_tags"][0][required_cells_arr]

    # Save the mesh as numpy arrays
    npz_path = mesh_path.with_suffix(".npz")
    np.savez(f"{npz_path}", knt=new_points, ijk=new_ijk)
