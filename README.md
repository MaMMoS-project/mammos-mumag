# Info
Repository contains a DockerFile to create a docker image for `hystmag`. `hystmag` is a finite-element micromagnetic simulation tool capable of simulating hysteresis loops of magnetic materials with multiple grains, developed and maintained by Thomas Schrefl at Zentrum für Modellierung und Simulation, Universität für Weiterbildung Krems.

# Build the docker image
To build the image, clone and change directory (`cd`) to the repository. Once in the repository, run:
```bash
docker build -t hystmag --build-arg BUILD_THREADS=20 --build-arg RUN_THREADS=1 .
```
This will build a `hystmag:latest` docker image. The argument `BUILD_THREADS` defines the number of threads used by `scons` to build `escript`, whereas the argument `RUN_THREADS` defines the number threads used by `escript` to run the simulation.

>**_NOTE:_** Once the number of `escript` run threads are fixed at the compile time, they cannot be changed later.

# Run the docker image
To run the simulation, one needs to have following configuration files in the working directory:
1. `<system-name>.fly` which is the mesh file.
2. `<system-name>.krn` which defines material parameters of each grain in the magnetic material.
3. `<system-name>.p2` which defines the simulation parameters, such as, external field range, step size of hysteresis, size of the geometry, initial magnetisation, etc.

# Build Apptainer image
In order to run `hystmag` on HPC, one needs to build an Apptainer `.sif` file from the provided `Apptainer.def` file. To do the same, run:
```bash
apptainer build --build-arg BUILD_THREADS=20 --build-arg RUN_THREADS=1 hystmag.sif Apptainer.def
```
>**_NOTE:_** Once the number of `escript` run threads are fixed at the compile time, they cannot be changed later.

# Example

## Create a finite element mesh for micromagnetic simulations

We use Salome https://www.salome-platform.org/ for geometry and mesh generation.
Please see `mumag/examples/standard_problem_3` for an example.

To create the mesh file of a cube with an edge length of 40 nm and a mesh size of 2 nm
```bash
cd mumag/examples/standard_problem_3
salome_install_path/SALOME-9.12.0/salome -t cube.py args:40,2
../../py/tofly3 -e 1,2 cube.unv cube.fly
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
```bash
run-escript ../../py/materials.py cube
```

## Run the standard problem 3

for details see https://www.ctcms.nist.gov/~rdm/spec3.html

```bash
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

```bash
run-escript ../../py/hmag.py cube
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



###  To run the docker image use:

```bash
docker run --volume $(pwd):/io --user="$(id -u):$(id -g)" hystmag <system-name>
```

###  To run the apptainer container use:

```bash
apptainer run hystmag.sif <system-name>
```
Since `hystmag.sif` is also an executable, simply run:

```bash
hystmag.sif <system-name>
```
>**_NOTE:_** The configuration files must be in the current working directory.
