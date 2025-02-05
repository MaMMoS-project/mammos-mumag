# Info
`mmag` is a finite-element micromagnetic simulation tool capable of simulating hysteresis loops of magnetic materials with multiple grains, developed and maintained by Thomas Schrefl at Zentrum für Modellierung und Simulation, Universität für Weiterbildung Krems.


# Install package
To install the package, run:
```console
pip install .
```

This will install `mmag` as a command line executable. It requires either `apptainer` or `podman` Installed on the system in order to build `esys-escript` container which a dependency of `mmag`.

> **_NOTE:_**  Make sure to install `fuse-overlayfs` from conda-forge or any other package manager if the Linux kernel version is below 5.11.


# Usage
To get a quick summary of all the options available with `mmag`, run:
```console
$ mmag --help
usage: mmag [-h] [-v] {build-escript,unvtofly,run} ...

positional arguments:
  {build-escript,unvtofly,run}
    build-escript       Option to build esys-escript container using apptainer or podman.
                        mmag depends on esys-escript for the simulations.
                        The definition files to build the container are provided with the package.
    unvtofly            Convert unv files to the fly format. Elements that belong to a group called
                        'contact' will be converted to their contact counterparts. First and secound
                        order meshes are supported.
    run                 Run the mmag simulation based on the pre-defined scripts.

options:
  -h, --help            show this help message and exit
  -v, --version         show version of the package and exit
```

`mmag` takes three sub-commands: `build-escript`, `unvtofly`, and `run`.

## sub-command `build-escript`
`mmag`'s simulation scripts depend on `esys-escript` for the FEM simulation. The repository comes with `esys-escript` container definition files for [apptainer](https://apptainer.org/) and [podman](https://podman.io/) and the building is handled by `build-escript` sub-command. To build the container, run:
```console
mmag build-escript --threads 10 --container <apptainer or podman>
```

This will build the `esys-escript` container for the selected program and configure `mmag` to use it. Please use help for further options:
```console
$ mmag build-escript --help
usage: mmag build-escript [-h] -c {apptainer,podman} [-t THREADS]

options:
  -h, --help            show this help message and exit
  -c, --container {apptainer,podman}
                        Specify the container program to use. It can be either apptainer or podman.
  -t, --threads THREADS
                        Specify the number of build threads (Defaults to 4)
```

## sub-command `unvtofly`
`mmag` uses `fly` mesh file format; `mamossmag unvtofly` convert standard `unv` mesh files to `fly` format.
```console
mmag unvtofly <unv-flile-name> <fly-file-name>
```

Further options are:
```console
$ mmag unvtofly --help
usage: mmag unvtofly [-h] [-e DIMENSIONS] [UNV] [FLY]

positional arguments:
  UNV                   Path to the input file or '-' for stdin.
                        It must already exist and be stored in the unv format.
                        If ommited stdin will be used instead.
  FLY                   Path to the output file or '-' for stdout.
                        Overridden if it already exists.
                        If ommited stdout will be used instead.

options:
  -h, --help            show this help message and exit
  -e, --exclude DIMENSIONS
                        Comma separated list of DIMENSIONS that shall be ignored
                        while converting (e.g. '-e 1,2' only converts 3D elements).
```

## sub-command `run`
This sub-command is used to actually run `mmag` simulations with a user-defined python script or with pre-defined simulation scripts, for example:
```console
mmag run -c apptainer -t 5 examples/demagnetization/cube_simulation.py
```
runs the script `cube_simulation.py` in `examples/demagnetization/`, and
```console
mmag run -c apptainer -t 5 loop -n <name-system>
```
runs the `loop` script with the specified `name-system`.

The possible pre-defined scripts are `exani`, `external`, `hmag`, `loop`, `magnetisation`, and `materials`.
Some user-defined scripts require one or more of the following configuration files in the working directory:
1. `<name-system>.fly`: the mesh file.
2. `<name-system>.krn`: defines material parameters of each grain in the magnetic material.
3. `<name-system>.p2`: simulation parameters, such as, external field range, step size of hysteresis, size of the geometry, initial magnetisation, etc.

For all the options, run:
```console
$ mmag run --help
usage: mmag run [-h] [-t THREADS] [-c {apptainer,podman}] [-n NAME_SYSTEM] script

positional arguments:
  script                Python file or pre-defined simulation script to execute.
                        The name must be one of loop, exani,
                        external, hmag, magnetisation, or materials.

options:
  -h, --help            show this help message and exit
  -t, --threads THREADS
                        Specify the number of runtime threads for esys-escript (mmag).
  -c, --container {apptainer,podman}
                        Choose the container program to use for running esys-escript.
  -n, --name-system NAME_SYSTEM
                        The name given to the simulation configuration files in the present working directory.
```


# Python API

`mmag` is also a Python package.
For its use consider `examples/demagnetization/cube_simulation.py`:
```python
from pathlib import Path
from mmag.loop import Loop

here = Path(__file__).absolute().parent
loop = Loop('cube', outdir=here/'out')
loop.read_mesh(here/'cube.fly')
loop.read_params(here/'cube.p2')
loop.read_materials(here/'cube.krn')
loop.run()
```

So, one can run
```console
mmag run examples/demagnetization/cube_simulation.py
```
and the input files do not need to be in the same folder, as the command
```console
mmag run loop -n cube
```
would require. In particular, using the simulation script `loop` requires:
- Files `cube.fly`, `cube.krn`, `cube.p2` are already in the same folder
- I expect the output file to be `cube.dat` in the same folder.


# Example

## Create a finite element mesh for micromagnetic simulations

We use Salome https://www.salome-platform.org/ for geometry and mesh generation.
Please see `mumag/examples/standard_problem_3` for an example.

To create the mesh file of a cube with an edge length of 40 nm and a mesh size of 2 nm
```console
cd examples/standard_problem_3
salome_install_path/SALOME-9.12.0/salome -t cube.py args:40,2
mmag unvtofly -e 1,2 cube.unv cube.fly
```

In order to define space dependent material properties, groups are created in the Salome geometry and the Salome mesh. Group names are 1, 2, 3, .....
These numbers refer to the line numbers of the *.krn file.

## Show the materials

The file cube.krn contains one line per mesh region.
Each line contains

theta phi K1 0 Js A

where

|    |    |
| -------- | ----------- |
| theta | is the angle of the magnetocrystalline anisotropy axis from the z-direction in radians |
| phi | is the angle of the magnetocrystalline anisotropy axis from the x-direction in radians |
| K1 | is the uniaxial anisotropy constant in J/m3 | 
| 0  | is a placeholder for K2 (currently not implemented) |  
| Js | is the magnetic polarization in T | 
| A  | is the exchange constant in J/m  |

The last two lines denote a sphere enclosing the magnetic region and a spherical shell. 

To create a vtu file that shows the materials use   
```console
mmag run -c apptainer -t 5 materials -n cube
```

## Run the standard problem 3

for details see https://www.ctcms.nist.gov/~rdm/spec3.html

```console
cd mumag/examples/standard_problem_3
python mumag3.py
```

This creates a file results/energies.png which give the energy of the flower and the vortex state as function of cube size. The critical lenght is 8.51 lex is close to that obtained by Hertel and Kronmüller https://www.ctcms.nist.gov/~rdm/results3.html#hertel .

## Open boundary problem

The magnetostatic potential is set to zero at infinity. To treat this boundary condition numerically, we can enclose the magnetic materials in an airbox or apply the spherical shell transformation.

### Airbox

The air box has to be around 5 times larger than the extension of the magnet. See
Chen, Qiushi, and Adalbert Konrad. "A review of finite element open boundary techniques for static and quasi-static electromagnetic field problems." IEEE Transactions on Magnetics 33.1 (1997): 663-676.

The example in pymag/t uses an airbox.

### Spherical shell transformation

Alternatively we can apply a transformation. A shell sourrounding the magnet is transformed to fill the space between the magnet and infinity. The most straight forward geometry is a spherical shell. See
Imhoff, J. F., et al. "An original solution for unbounded electromagnetic 2D-and 3D-problems throughout the finite element method." IEEE Transactions on Magnetics 26.5 (1990): 1659-1661.  

The two last groups defined in Salome are a sphere sourrounding the magnetic material and the spherical shell.

The example in pymag/meshing uses the spherical shell transformation.

### Magnetostatic energy

To test the magnetostatic field computation you can calculate the magnetostatic energy density and the field of a uniformly magnetized cube.

```console
mmag run -c apptainer -t 5 hmag -n cube
```

### Solver parameters

The minimizer section in the *.p2 file gives the paramters for the LBFGS algorithm. The algorithm uses the preconditioner described in
Exl, Lukas, et al. "Preconditioned nonlinear conjugate gradient method for micromagnetic energy minimization." Computer Physics Communications 235 (2019): 179-186.

| parameter |  usage  | default value |
| --------- | ------- | ------------- |
| tol_fun | set the tolerance of the total energy | 1e-10 |
| tol_hmag_factor | tol_fun\*tol_hmag_factor is the tolerance for the magnetostatic scalar potential | 1 |
| truncation | number of history vectors stored | 5 | 
| precond_iter | number of conjugate gradient iterations for inverse Hessian approximation | 10 |  
| verbose  | verbosity level of output  | 1 |
