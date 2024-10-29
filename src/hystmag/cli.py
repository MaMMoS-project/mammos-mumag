import argparse

import sys

import hystmag
from hystmag import tofly, cli_helpers


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
        type=tofly.dimension,
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
        ),
    )

    args = parser.parse_args()

    if args.sub_parser == "build-escript":
        exec(f"cli_helpers.install_escript_{args.program}(args.threads)")

    if args.sub_parser == "run":
        cli_helpers.run_hystmag(args.threads, args.script, args.system)
