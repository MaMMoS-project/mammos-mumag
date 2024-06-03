# Info
Repository contains a DockerFile to create a docker image for `hystmag`. `hystmag` is a finite-element micromagnetic simulation tool capable of simulating hysteresis loops of magnetic materials with multiple grains, developed and maintained by Thomas Schrefl at Zentrum für Modellierung und Simulation, Universität für Weiterbildung Krems.

# Build the docker image
To build the image, clone and change directory (`cd`) to the repository. Once in the repository, run:
```bash
docker build -t hystmag .
```
This will build a `hystmag:latest` docker image.

# Run the docker image
To run the simulation, one needs to have following configuration files in the working directory:
1. `<system-name>.fly` which is the mesh file.
2. `<system-name>.krn` which defines material parameters of each grain in the magnetic material.
3. `<system-name>.p2` which defines the simulation parameters, such as, external field range, step size of hysteresis, size of the geometry, initial magnetisation, etc.

Please refer to `mumag/pymag/t` repository for an example of these configuration files.

To run the docker image use:
```bash
docker run --volume $(pwd):/io --user="$(id -u):$(id -g)" hystmag <system-name>
```
