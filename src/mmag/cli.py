"""CLI functions."""

import argparse
import sys

import mmag
from . import cli_helpers, tofly


def main():
    """Execute CLI commands."""
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest="sub_parser")
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"mmag {mmag.__version__}",
    )

    escript_build_parser = sub_parsers.add_parser(
        name="build-escript",
        help=(
            "Option to build esys-escript container using apptainer or podman. "
            "mmag depends on esys-escript for the simulations. "
            "The definition files to build the container are provided with the package."
        ),
    )

    escript_build_parser.add_argument(
        "-c",
        "--container",
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
        help=("Run the mmag simulation based on the pre-defined scripts."),
    )

    run_parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=4,
        required=False,
        help=("Specify the number of runtime threads for esys-escript (mmag)."),
    )

    run_parser.add_argument(
        "-c",
        "--container",
        type=str,
        default=None,
        required=False,
        choices=("apptainer", "podman"),
        help=("Choose the container program to use for running esys-escript."),
    )

    run_parser.add_argument(
        "script",
        type=str,
        help=(
            "Python file or pre-defined simulation script to execute. "
            "The name must be one of loop, exani, "
            "external, hmag, magnetisation, or materials."
        ),
    )

    run_parser.add_argument(
        "-n",
        "--name-system",
        type=str,
        default=None,
        required=False,
        help=(
            "The name given to the simulation configuration files in the present "
            "working directory."
        ),
    )

    args = parser.parse_args()

    if args.sub_parser == "build-escript":
        cli_helpers.install_escript(args.container, args.threads)

    if args.sub_parser == "unvtofly":
        nodes, index, groups, contact = tofly.scanUnv(args.infile, args.exclude)
        tofly.writeFly(
            nodes, groups, index, contact, args.infile, args.outfile, args.exclude
        )
        args.infile.close()
        args.outfile.close()

    if args.sub_parser == "run":
        cli_helpers.run_mmag(
            args.threads, args.container, args.script, args.name_system
        )
