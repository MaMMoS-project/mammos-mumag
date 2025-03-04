import numpy as np
import os
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import sys

try:
    mesh_size = float(sys.argv[1])
except IndexError:
    sys.exit("usage python mumag3.py mesh_size\n      where mesh_size is given in units of the exchange length")


salome = "/home/tom/pd/SALOME-9.13.0/salome"
tofly = "../demagnetization/tofly3"

escript = "run-escript"
path_to_loop = "../../src/mmag/sim_scripts/loop.py"
path_to_store = "../../src/mmag/sim_scripts/store.py"


matplotlib.rcParams.update({"font.size": 14})

mu0 = 4e-7 * np.pi
Js = 1.05    # T     mu0Ms
A = 1.3e-11  # J/m   exchange constant
Km = 0.5 * Js * Js / mu0
lex = np.sqrt(A / Km)
Ku = 0.1 * Km


def write_krn(name, K1, Js, A):
    with open(name + ".krn", "w") as f:
        f.write(f"0.0 0.0 {K1:.8e} 0.0 {Js} {A}\n")
        f.write("0.0 0.0 0.0  0.0 0.0  0.0\n")
        f.write("0.0 0.0 0.0  0.0 0.0  0.0\n")


def write_p2(name, state):
    with open(name + ".p2", "w") as f:
        s = f"""[mesh]
size = 1.e-9
scale = 0.0

[initial state]
mx = 0.
my = 0.
mz = 1.
state = {state}

[field]
hstart = 0.0
hfinal = 0.0
hstep = -0.02
hx = 0.
hy = 0.
hz = 1.

[minimizer]
tol_fun = 1e-8
tol_hmag_factor = 0.01
precond_iter = 4\n"""
        f.write(s)


def read_last_float(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
        last_line = lines[-1]
        last_column_value = last_line.split()[-1]
        return float(last_column_value)


def convert_energies(e):
    return (e + Ku) / Km

Path("results").mkdir(exist_ok=True)

# write material properties
write_krn("cube", Ku, Js, A)

lengths = np.array([8.5,8.55,8.6])
energies_flower = []
energies_twisted = []
energies_vortex = []


for s in lengths:

    # (TWISTED) FLOWER STATE
    print(f"\nflower : {s:.2f} lex\n")

    # create mesh
    cmd = f"{salome} -t -w1 cube.py args:{s*lex*1e9:.2f},{mesh_size*lex*1e9:.2f},4"
    print(cmd)
    os.system(cmd)

    # convert mesh to fly format
    cmd = f"{tofly} -e 1,2 cube.unv cube.fly"
    print(cmd)
    os.system(cmd)
    
    # write parameter file    
    write_p2("cube", "flower")

    # store sparse matrices
    cmd = f"{escript} {path_to_store} cube"
    print(cmd)
    os.system(cmd)

    # run simulation
    cmd = f"{escript} {path_to_loop} cube"
    print(cmd)
    os.system(cmd)
    
    energies_flower.append(read_last_float("cube.dat"))
    os.rename("cube.dat", Path("results") / f"cube_flower_{s:.2f}.dat")
    os.rename("cube_0000.vtu", Path("results") / f"cube_flower_{s:.2f}.vtu")

    # VORTEX STATE
    print(f"\nvortex : {s:.2f} lex\n")
    
    # write parameter file
    write_p2("cube", "vortex")
    
    # run simulation
    cmd = f"{escript} {path_to_loop} cube"
    print(cmd)
    os.system(cmd)
    
    energies_vortex.append(read_last_float("cube.dat"))
    os.rename("cube.dat", Path("results") / f"cube_vortex_{s:.2f}.dat")
    os.rename("cube_0000.vtu", Path("results") / f"cube_vortex_{s:.2f}.vtu")

# plot results

energies_flower = np.array(energies_flower)
energies_vortex = np.array(energies_vortex)
energies_flower = convert_energies(energies_flower)
energies_vortex = convert_energies(energies_vortex)

plt.plot(lengths, energies_flower, marker="o", label="twisted flower")
plt.plot(lengths, energies_vortex, marker="s", label="vortex")
plt.xlabel(r"size (l$_\mathrm{ex}$)")
plt.ylabel(r"energy density ($\mu_0 M_\mathrm{s}/2$)")
plt.legend()
plt.tight_layout()
plt.savefig(Path("results") / "energies.png")

# clean directory

os.remove("cube.unv")
os.remove("cube.fly")
os.remove("cube.krn")
os.remove("cube.pkl")
os.remove("cube.p2")
