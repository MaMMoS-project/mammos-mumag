"""Script to convert old mesh files to numpy arrays for mumag-matrixfree.

Usage:
python old_mesh_to_matrixfree_mesh.py <folder-with-old-med-meshes>
"""

import pathlib as pl
import sys
from multiprocessing import Pool

import meshio as mio
import numpy as np
from scipy.spatial import KDTree


def save_new_mesh_arrays(mesh_path: pl.Path):
    """Takes in a path to an old med mesh and creates the npz file."""
    mesh = mio.read(mesh_path)
    file_name = mesh_path.name.removesuffix(".med")
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
    np.savez(f"mesh_numpy/{file_name}", knt=new_points, ijk=new_ijk)


if __name__ == "__main__":
    mesh_files = list(pl.Path(sys.argv[1]).glob("*.med"))
    pl.Path("mesh_numpy").mkdir(exist_ok=True)

    with Pool(len(mesh_files)) as p:
        p.map(save_new_mesh_arrays, mesh_files)
