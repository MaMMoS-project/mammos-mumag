import sys
from pathlib import Path
import os

from mammosmag.loop import Loop

def check_input_files(name):
    pp = Path(name)
    if (
        pp.with_suffix('.fly').is_file()
        and (pp.with_suffix('.p2')).is_file()
        and (pp.with_suffix('.krn')).is_file()
    ):
        return
    raise Exception(f"""
        Necessary input files missing.
        Current workdir: {os.getcwd()}
        Files here: {os.listdir()}
    """)

def main(name):
    loop = Loop(name)
    check_input_files(name)
    loop.read_mesh(name + ".fly")
    loop.read_params(name + ".p2")
    loop.read_materials(name + ".krn")
    loop.run()

if __name__ == "__main__":
    try:
        name = sys.argv[1]
    except IndexError:
        sys.exit("usage: mammosmag run loop -n name_system")
    main(name)
