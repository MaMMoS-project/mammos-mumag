"""Script to quickly convert Tom's numpy arrays into vtu files.

The main purpose of this script is to create a vtu file to quickly
visualise the mesh using paraview.

Usage:
python numpy_mesh_to_vtu.py <numpy-mesh-file-path>
"""

import sys
from pathlib import Path

import meshio as mio
import numpy as np

if __name__ == "__main__":
    mesh_path = Path(sys.argv[1])
    mesh = np.load(mesh_path)

    points = mesh["knt"]
    connectivity = mesh["ijk"]

    mio_mesh = mio.Mesh(
        points=points,
        cells=[("tetra", connectivity[:, 0:-1])],
        cell_data={"cell_tags": [connectivity[:, -1]]},
    )
    mio_mesh.write(f"{mesh_path.name.removesuffix('.npz')}.vtu")
