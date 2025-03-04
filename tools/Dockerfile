# syntax=docker/dockerfile:1

# Create a base image
FROM docker.io/condaforge/miniforge3 AS base
COPY ./scripts /scripts
COPY ./patches/escript.yaml ./escript.yaml
RUN conda env create -f ./escript.yaml

# Create build stage (image)
FROM base AS build
RUN apt-get update -y
RUN apt-get install build-essential -y
RUN git clone https://github.com/LutzGross/esys-escript.github.io.git escript
WORKDIR escript/
RUN git reset --hard 6d6f21f9b77078e186ab0bdc10b49a97c0688f3c
RUN git clone https://github.com/trilinos/Trilinos.git
WORKDIR Trilinos/
RUN git reset --hard ee6599d413f7ef5786a5ad1f719bbfe3a4d22300
WORKDIR /escript
COPY ./patches/dependencies.py ./site_scons/
COPY ./patches/SConstruct ./
COPY ./patches/hp_options.py scons/
RUN conda install -y -n escript -c conda-forge cmake
ARG BUILD_THREADS
RUN conda run --no-capture-output -n escript scons -j$BUILD_THREADS options_file=scons/hp_options.py

# Create the final reduced image
FROM base AS final
COPY --from=build /escript /escript
WORKDIR /io
ENTRYPOINT OMP_PROC_BIND=spread OMP_PLACES=threads PYTHONPATH=$PYTHONPATH:/escript/escript LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/escript/lib:/escript/escript_trilinos/lib /escript/bin/run-escript "$0" "$@"
