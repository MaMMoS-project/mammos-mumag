"""Download all available meshes from Keeper."""

import json
import pathlib

from mammos_mumag.mesh import _get_mesh_json_from_keeper, get_mesh_json


def update_mesh_json():
    """Update mesh README.json file containing meshes information.

    This function get the latest up to date JSON file containing names and information
    of all available meshes. It also contains the URL of the Keeper directory where
    the meshes are stored of the Zenodo record where the `fly` formats are published.
    """
    old_json = get_mesh_json()
    new_json = _get_mesh_json_from_keeper()
    if old_json == new_json:
        print("Mesh JSON file already up to date.")
    else:
        print("Mesh JSON was updated.")
        with open(pathlib.Path(__file__).parent / "mesh" / "README.json", "w") as f:
            json.dump(new_json, f, indent=2)


if __name__ == "__main__":
    update_mesh_json()
