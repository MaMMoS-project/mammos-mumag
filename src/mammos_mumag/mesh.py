"""Mesh functions."""

import json
import pathlib
import shutil
from warnings import warn

import requests
import urllib3
from platformdirs import user_cache_dir


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
    """Mesh class.

    This class supports both local mesh (i.e. found on disk and defined by the user) and
    remote meshes stored on Zenodo. The Zenodo URL is found in the ``README.json`` and a
    list of all available meshes is found using :py:func:`~mammos_mumag.mesh.find_mesh`.

    Attributes:
        name:

            * If local mesh, path to the file.
            * If remote mesh, its name is as it appears on the Zenodo record or in the
              ``README.json``.

        info: Dictionary of available information about the mesh. If the mesh is local,
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
                mesh_json = get_mesh_json()
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

    def write(self, destination: pathlib.Path | str, use_cache: bool = True) -> None:
        """Write mesh to destination.

        Args:
            destination: Where to save the mesh.
            use_cache: Whether to cache the remote mesh. If True, the mesh gets first
                downloaded to the system cache directory and then copied to destination.
                If the remote mesh with the same name is already in the cache directory,
                the download is skipped. The system cache directory is defined by the
                function :py:func:`platformdirs.user_cache_dir` to ensure compatibility
                with different platforms.

        Raises:
            ValueError: Wrong mesh format. Only `.fly` meshes can be written with this
                function. If the suffix of the destination is different, this error is
                raised.
        """
        destination = pathlib.Path(destination).resolve()
        if suff := destination.suffix != ".fly":
            raise ValueError(
                "Wrong mesh format. "
                "Only `.fly` meshes can be written. "
                f"Given destination suffix: {suff}."
            )

        if self._local:
            shutil.copy(self._path, destination)
        else:
            if use_cache:
                (cached_dest := pathlib.Path(user_cache_dir("mammos_mumag"))).mkdir(
                    exist_ok=True, parents=True
                )
                if not (cached_file := cached_dest / self.name).is_file():
                    self._download_mesh(cached_file)
                shutil.copy(cached_file, destination)
            else:
                self._download_mesh(cached_file)

    def _download_mesh(self, destination: pathlib.Path | str) -> None:
        """Download mesh to destination.

        This function tries to download from Zenodo first. If the request fails, the
        mesh is instead downloaded from Keeper.

        Args:
            destination: Where to save the mesh.

        Raises:
            ValueError: Wrong mesh format. The only available mesh format on
                Zenodo is `.fly`. If the suffix of the destination is different,
                this error is raised.
        """
        destination = pathlib.Path(destination).resolve()
        if destination.suffix != ".fly":
            raise ValueError(
                "Wrong mesh format. "
                "Only `.fly` meshes are available on Zenodo. "
                f"Given destination suffix: {destination.suffix}."
            )

        res = _request(self._url)
        if res.status_code == 200:
            # Download from Zenodo successful
            with open(destination, "wb") as f:
                f.write(res.content)
        else:
            warn(
                "Unable to download mesh from Zenodo. "
                f"The request returned with HTTP code: {res.status_code}. "
                "Downloading the mesh from keeper.",
                stacklevel=1,
            )
            # Keeper works reliably when downloading 1000 fly mesh in parallel!
            # Think about completely replacing Zenodo downloads with Keeper.
            self._download_from_keeper(destination)

    def _download_from_keeper(self, destination: pathlib.Path | str) -> None:
        """Download mesh from Keeper.

        Download from Keeper seems to be more reliable and should be used
        as a fallback in case the download from Zenodo fails.

        Args:
            destination: Where to save the mesh.

        Raises:
            ValueError: Wrong mesh format. The available mesh formats on Keeper are
                `.fly`, `.med`, and `.unv`. If the suffix of the destination is
                different, this error is raised.
        """
        destination = pathlib.Path(destination).resolve()
        avail_fmts = [".fly", ".med", ".unv"]
        if destination.suffix not in avail_fmts:
            raise ValueError(
                "Wrong mesh format. "
                f"Only {avail_fmts} meshes are available on Keeper. "
                f"Given destination suffix: {destination.suffix}."
            )

        keeper_url = get_mesh_json()["metadata"]["keeper_url"]
        mesh_url = f"{keeper_url}files/?p=/{self.name}/mesh{destination.suffix}&dl=1"
        res = _request(mesh_url, destination)
        with open(destination, "wb") as f:
            f.write(res.content)


def _get_mesh_json_from_keeper() -> dict:
    """Download mesh.json from Keeper and return dictionary."""
    keeper_url = get_mesh_json()["metadata"]["keeper_url"]
    res = requests.get(f"{keeper_url}files/?p=/README.json&dl=1")
    if res.status_code != 200:
        raise FileNotFoundError("README.json not found on Keeper.")
    else:
        return json.loads(res.content)


def _request(url) -> requests.Response:
    """Request content from a webpage and get a ``Response`` back.

    If the request fails with codes 50X, it is retried for a total of three times
    with a 0.1 backoff factor. This function is used only to access meshes on Zenodo or
    Keeper.

    Args:
        url: URL of webpage to download
    """
    s = requests.Session()
    retries = urllib3.util.Retry(
        total=3,
        backoff_factor=0.1,
        status_forcelist=[500, 502, 503, 504],
    )
    s.mount("https://", requests.adapters.HTTPAdapter(max_retries=retries))
    res = s.get(url)
    return res
