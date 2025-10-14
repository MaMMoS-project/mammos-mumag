"""Mesh functions."""

import json
import pathlib
import shutil

import requests
import urllib3


def get_mesh_json():
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
    meshes = get_mesh_json()["meshes"]
    if mesh_name is None:
        return list(meshes.keys())
    else:
        return [mm for mm in meshes if mesh_name in mm]


class Mesh:
    """Mesh class."""

    def __init__(self, mesh_name: str | pathlib.Path):
        """Initialize Mesh with either mesh_name or path for local meshes."""
        matches = (
            find_mesh(mesh_name) if not isinstance(mesh_name, pathlib.Path) else []
        )
        if len(matches) > 1:
            raise ValueError(
                f"Mesh name ambiguous. More than one match found: {matches}"
            )
        elif not matches:
            filepath = pathlib.Path(mesh_name)
            if filepath.is_file():
                self.name = filepath
                self.info = {"description": "User defined mesh."}
                self._local = True
                self._path = filepath
            else:
                raise ValueError("Mesh not found.")
        else:
            mesh_json = get_mesh_json()
            self.name = matches[0]
            self.info = mesh_json["meshes"][matches[0]]
            if _local := (
                _path := pathlib.Path(__file__).parent / "mesh" / f"{self.name}.fly"
            ).is_file():
                self._path = _path
            self._local = _local
            self._url = f"{mesh_json['metadata']['zenodo_url']}/files/{self.name}.fly"

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
        keeper_url = get_mesh_json()["metadata"]["keeper_url"]
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
    keeper_url = get_mesh_json()["metadata"]["keeper_url"]
    res = requests.get(f"{keeper_url}files/?p=/README.json&dl=1")
    if res.status_code != 200:
        raise FileNotFoundError("README.json not found on Keeper.")
    else:
        return json.loads(res.content)
