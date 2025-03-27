"""Tool functions."""

import pathlib


def check_path(fname):
    """Check that file exists.

    :param fname: File path.
    :type fname: str or pathlib.Path
    :raises FileNotFoundError: File not found.
    :return: File path.
    :rtype: pathlib.Path
    """
    path = pathlib.Path(fname).resolve()
    if not path.is_file():
        raise FileNotFoundError("File not found.")
    return path


def check_dir(outdir):
    """Check that directory exists.

    :param outdir: Directory path.
    :type outdir: str or pathlib.Path
    :return: Checked directory path.
    :rtype: pathlib.Path
    """
    outdir = pathlib.Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    return outdir
