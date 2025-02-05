import sys

from mmag.materials import Materials


def main(name):
    materials = Materials(name)
    materials.read_mesh(name + ".fly")
    materials.read_materials(name + ".krn")
    materials.write_vtk()


if __name__ == "__main__":
    print("materials:")
    try:
        name = sys.argv[1]
    except IndexError:
        sys.exit("mmag run materials -n name_system")
    main(name)
