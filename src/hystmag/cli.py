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
            f"{hystmag._container_scripts}"
        ),
        posix=(os.name == "posix"),
    )

    res = subprocess.run(cmd, capture_output=False)

    if res.returncode == 0:
        hystmag._conf_dir.mkdir(parents=True, exist_ok=True)
        with open(f"{hystmag._conf_dir/"conf.json"}", "w") as handle:
            json.dump({"escript_container_program": "podman"}, handle)
    else:
        raise OSError(
            "Unable to install the podman container. Exit with error:\n"
            f"{res.stderr}"
        )


def install_escript_apptainer(threads):
    podman_path = shutil.which("apptainer")
    if not podman_path:
        raise OSError("apptainer cannot be accessed through PATH variable.")

    hystmag._cache_dir.mkdir(exist_ok=True, parents=True)

    cmd = shlex.split(
        (
            "apptainer build "
            f"--build-arg BUILD_THREADS={threads} "
            f"--build-arg PATCH_DIR={hystmag._container_scripts/"patches"} "
            f"{hystmag._cache_dir/"escript.sif"} "
            f"{hystmag._container_scripts/"Apptainer.def"}"
        ),
        posix=(os.name == "posix"),
    )

    res = subprocess.run(cmd, capture_output=False)

    if res.returncode == 0:
        hystmag._conf_dir.mkdir(parents=True, exist_ok=True)
        with open(f"{hystmag._conf_dir/"conf.json"}", "w") as handle:
            json.dump({"escript_container_program": "apptainer"}, handle)
    else:
        raise OSError(
            "Unable to install the apptainer container."
        )


def run_hystmag(threads, script, system):
    if not (hystmag._conf_dir/"conf.json").exists():
        raise RuntimeError(
            f"Cannot find a configuration file in {hystmag._conf_dir}. "
            "Make sure to build and install escript container, for example: "
            "hystmag build-escript --threads 8 --program apptainer"
        )
    with open(hystmag._conf_dir/"conf.json", "r") as handle:
        config = json.load(handle)

    escript_container = config["escript_container_program"]

    if escript_container == "apptainer":
        cmd = shlex.split(
            (
                "apptainer run "
                f"{hystmag._cache_dir/"escript.sif"} -t{threads} "
                f"{hystmag._sim_scripts/(script+".py")} {system}"
            ),
            posix=(os.name == "posix")
        )

        res = subprocess.run(cmd, capture_output=False)

        if res.returncode != 0:
            raise RuntimeError(
                f"Hystmag {script} execution for {system} failed "
                f"using {escript_container} escript container."
            )
    elif escript_container == "podman":
        cmd = shlex.split(
            (
                f"podman run -v .:/io -v {hystmag._sim_scripts}:/sim_scripts "
                f"escript -t{threads} /sim_scripts/{script}.py {system}"
            ),
            posix=(os.name == "posix")
        )

        res = subprocess.run(cmd, capture_output=False)

        if res.returncode != 0:
            raise RuntimeError(
                f"Hystmag {script} execution for {system} failed "
                f"using {escript_container} escript container."
            )
    else:
        raise NotImplementedError(
            f"{escript_container} escript container is not supported."
        )


def main():
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest="sub_parser")
    parser.add_argument(
        "-v", "--version", action="version", version=f"hystmag {hystmag.__version__}"
    )

    escript_build_parser = sub_parsers.add_parser(
        name="build-escript",
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
        choices=("apptainer", "podman"),
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
        choices=("loop", "exani", "external", "hmag", "magnetisation", "materials"),
        help=(
            "Name the pre-defined simulation script to use. The name must be one of "
            "loop, exani, external, hmag, magnetisation, or materials."
        ),
    )

    run_parser.add_argument(
        "system",
        type=str,
        help=(
            "The name given to the simulation configuration files in the present "
            "working directory."
        )
    )

    args = parser.parse_args()

    if args.sub_parser == "build-escript":
        exec(f"install_escript_{args.program}(args.threads)")

    if args.sub_parser == "run":
        run_hystmag(args.threads, args.script, args.system)
