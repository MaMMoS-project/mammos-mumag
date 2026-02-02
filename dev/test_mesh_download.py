from mammos_mumag import mesh
from multiprocessing import Pool
import numpy as np
import pathlib as pl

NUMBER_OF_TRIES = 10

def get_mesh(thread_number):
    print(f"{thread_number=}")
    m = mesh.Mesh(mesh_name="cube40_colu_grains8_gsize20")
    m.write(f"test_mesh/test-{thread_number}.fly")

if __name__ == "__main__":
    pl.Path("test_mesh").mkdir(exist_ok=True)
    with Pool(NUMBER_OF_TRIES) as p:
        p.map(get_mesh, np.arange(0, NUMBER_OF_TRIES))
