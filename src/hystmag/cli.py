import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys

import hystmag
from hystmag.tofly import dimension


def install_escript_podman(threads):
    podman_path = shutil.which("podman")
    if not podman_path:
        raise OSError("podman cannot be accessed through PATH variable.")

    cmd = shlex.split(
        (
            "podman build -t escript "
            f"--build-arg BUILD_THREADS={threads} "
            f"{hystmag._container_scripts/"Dockerfile"}"
        ),
        posix=(os.name == "posix"),
    )

    res = subprocess.run(cmd, capture_output=True)

    if res.returncode != 0:
        raise OSError("Unable to install the podman container.")
    else:
        hystmag._conf_dir.mkdir(parents=True, exist_ok=True)
        with open("conf.json", "w") as handle:
            json.dump({"escript_container_program": "podman"}, handle)


def install_escript_apptainer(threads):
    podman_path = shutil.which("apptainer")
    if not podman_path:
        raise OSError("apptainer cannot be accessed through PATH variable.")

    hystmag._cache_dir.mkdir(exist_ok=True, parents=True)

    cmd = shlex.split(
        (
            "apptainer build "
            f"--build-arg BUILD_THREADS={threads} "
            f"{hystmag._cache_dir/"escript.sif"} "
            f"{hystmag._container_scripts/"Apptainer.def"}"
        ),
        posix=(os.name == "posix"),
    )

    res = subprocess.run(cmd, capture_output=False)

    if res.returncode != 0:
        raise OSError(
            "Unable to install the apptainer container. Output error:\n\n"
            f"{res.stderr}"
        )
    else:
        hystmag._conf_dir.mkdir(parents=True, exist_ok=True)
        with open("conf.json", "w") as handle:
            json.dump({"escript_container_program": "apptainer"}, handle)


def main():
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest="sub_parser")
    parser.add_argument(
        "-v", "--version", action="version", version=f"hystmag {hystmag.__version__}"
    )

    escript_build_parser = sub_parsers.add_parser(
        name="escript",
        help=(
            "Option to build esys-escript container using apptainer or podman. Hystmag "
            "depends on esys-escript for the simulations. The definition files to "
            "build the container are provided with the package."
        ),
    )

    escript_build_parser.add_argument(
        "-p",
        "--program",
        type=str,
        required=True,
        help=(
            "Specify the container program to use. It can be either apptainer or "
            "podman."
        ),
    )

    escript_build_parser.add_argument(
        "-t",
        "--threads",
        type=int,
        required=False,
        default=4,
        help=("Specify the number of build threads."),
    )

    unvtofly_parser = sub_parsers.add_parser(
        name="unvtofly",
        help=(
            "Convert unv files to the fly format. Elements that "
            "belong to a group called 'contact' will be converted to their "
            "contact counterparts. First and secound order meshes are supported."
        ),
    )

    unvtofly_parser.add_argument(
        "infile",
        nargs="?",
        type=argparse.FileType("r", 1000),
        default=sys.stdin,
        metavar="UNV",
        help=(
            "Path to the input file "
            "or '-' for stdin. It must already exist and be stored in the "
            "unv format. If ommited stdin will be used instead. "
        ),
    )

    unvtofly_parser.add_argument(
        "outfile",
        nargs="?",
        type=argparse.FileType("w", 1000),
        default=sys.stdout,
        metavar="FLY",
        help=(
            "Path to the output file "
            "or '-' for stdout. Overridden if it already exists. If ommited "
            "stdout will be used instead."
        ),
    )

    unvtofly_parser.add_argument(
        "-e",
        "--exclude",
        type=dimension,
        default=set(),
        metavar="DIMENSIONS",
        help=(
            "Comma separated list of DIMENSIONS "
            "that shall be ignored while converting (e.g. '-e 1,2' only "
            "converts 3D elements)."
        ),
    )

    run_parser = sub_parsers.add_parser(
        name="run",
        help=("Run the hystmag simulation based on the pre-defined scripts."),
    )

    run_parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=4,
        required=False,
        help=("Specify the number of runtime threads for esys-escript (hystmag)."),
    )

    run_parser.add_argument(
        "-s",
        "--script",
        type=str,
        required=True,
        help=(
            "Name the pre-defined simulation script to use. The name must be one of "
            "loop, exani, external, hmag, magnetisation, or materials."
        ),
    )

    args = parser.parse_args()

    if args.sub_parser == "escript":
        if args.program not in {"apptainer", "podman"}:
            raise ValueError("Specified program must be either apptainer or podman.")

    exec(f"install_escript_{args.program}(args.threads)")
