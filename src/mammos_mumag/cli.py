"""CLI tools.

The command `unv2fly` converts a mesh in the `unv` format to `fly`.
"""

import argparse
import pathlib

from mammos_mumag import tofly


def convert_mesh():
    """Command-line entry point to convert unv mesh to fly format."""
    parser = argparse.ArgumentParser(
        prog="unv2fly",
        usage="%(prog)s -d [outdir]",
        description="Convert unv files to the fly format.",
    )
    parser.add_argument(
        "infile",
        type=pathlib.Path,
        help=("path of mesh in unv format"),
    )
    parser.add_argument(
        "outfile",
        nargs="?",
        type=pathlib.Path,
        default=None,
        help=(
            "path of mesh in fly format. If not defined the name will be inferred from "
            "the unv mesh"
        ),
    )
    parser.add_argument(
        "-e",
        "--exclude",
        type=str,
        default="",
        help=(
            "Comma separated list of dimension integers that shall be ignored in the "
            "conversion (e.g. '-e 1,2' only converts 3D elements)."
        ),
    )

    args = parser.parse_args()
    exclude_list = [int(i) for i in args.exclude.split(",") if i != ""]
    outfile = (
        args.outfile
        if args.outfile is not None
        else args.infile.with_suffix(".fly").name
    )
    tofly.convert(args.infile, outfile, exclude_list=exclude_list)
