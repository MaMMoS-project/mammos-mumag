"""Download all available meshes from Keeper."""

import json
import pathlib

from tqdm import tqdm

from mammos_mumag.mesh import Mesh, _get_mesh_json_from_keeper


def mesh_download_all(
    extension: str = "fly", outdir: str | pathlib.Path = "all_meshes"
) -> None:
    """Helper function to download all available meshes.

    Args:
        extension: Mesh format to download. Available extensions are `fly`, `med`, and
        `unv`.
        outdir: Directory where meshes are downloaded.
    """
    outdir = pathlib.Path(outdir)
    outdir.mkdir(exist_ok=True)
    keeper_json = _get_mesh_json_from_keeper()

    with open(outdir / "README.json", "w") as f:
        json.dump(keeper_json, f, indent=2)

    for mesh_name in (pbar := tqdm(keeper_json["meshes"])):
        pbar.set_description(f"Downloading {mesh_name}")
        Mesh(mesh_name)._write_from_keeper(outdir / f"{mesh_name}.fly")


if __name__ == "__main__":
    mesh_download_all()
