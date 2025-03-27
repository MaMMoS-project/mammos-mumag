# demagnetization of a permanent magnet cube

## mesh generation

generate a mesh with salome

`/salome_install_path/salomeSALOME-9.13.0/salome -t -w1 cube.py args:20,2,4`

The above command creates a mesh in the universial mesh format: https://victorsndvg.github.io/FEconv/formats/unv.xhtml

## convert the mesh to the fly format of esys-escript used by mammos mag 

`./tofly3 -e 1,2 cube.unv cube.fly`

## show the mesh and the materials

`run-escript materials.py cube`

creates the file cube_mat.vtk for the visualisation of the material properties. 

The cube is enclosed with a sphere. The sphere is embedded in spherical shell.
The `cube.krn` file contains to extra lines with zeros for those two regions.

## compute the magnetostatic field

`run-escript hmag.py cube`

creates the file cube_hmag.vtk for visualisation of the magnetic scalar potential and the magnetic field.
With linear basis function for the magnetic scalar potential u, the magnetostatic field h = -grad(u) is defined at the finite elements. By smoothing the field can be transfered to the nodes of the finite element mesh. This is `h_at_nodes`.

### magnetostatic energies

The software also gives the magnetostatic energy density computed with finite elements and compares it with the analytic soluation. 
Three energy values are compared:

from field: (Integral over (1/2) field * magnetic_polarization)/volume

from gradient:  (1/2) sum_i dot(m_i,g_i), where m_i and g_i are the unit vector of the magnetization and the gradient of the energy normalized by the volume
of the energy with respect to m_i at the nodes of the finite element mesh 

analytic: Js^2 / (6 mu_0)

## exchange and anisotropy energy

to test the computation of the exchange and anisotropy energy density

`run-escript exani.py cube`

this gives the exchange energy density of a vortex in the x-y plane and the anistropy energy density in the uniformly magnetized state.
Here we have placed the anistropy direction paralle to to the z-axis. The anisotropy energy density is calculated as 
-K dot(m,k)^2  where m is the unit vector of magnetization and k is the anisotropy direction. K is the magnetocrystalline anisotropy constant

## zeeman energy

to the the Zeeman energy density in the external field

`run-escript external.py cube`

computes the Zeemann energy by finite elements and analytically.

## jax implmentation

The above tools checked the energy calculation with the finite element backend. 
From the finite element backend system matrices are generated for micromagnetic simulations. 
To test the energy calculations with matrices use the following command:

`run-escript mapping.py cube`

The module `mapping.py` contains the tools for mapping from the finite element bilinear forms to sparse matrices. Here we use sparse matrix methods from jax.

### storing sparse matrices

The sparse matrices used for computation can be stored and reused for simulations with the same finite element mesh. To store the matrices use the command

`run-escript store.py cube`

This features is used in the example for the standard problem 3

## demagnetization curve

To compute demagnetization curves use

`run-escript loop.py cube`

creates the file `cube.dat` which gives the demagnetization curve. The columns of the file are 

`vtk number`      the number of the vtk file that corresponds to the field and magnetic polarisation values in the line  
`mu0 Hext`        the value of mu_0 Hext (T) where mu_0 is the permability of vacuum and Hext is the external value of the external field  
`polarisation`    the componenent of magnetic polarisation (T) parallel to the direction of the external field  
`energy density`  the energy density (J/m^3) of the current state  



