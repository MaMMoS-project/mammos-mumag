"""Build mmag/escript container with apptainer or podman."""

import json
import os
import pathlib
import shlex
import subprocess
import sys
import warnings

import mmag


IS_POSIX = os.name == "posix"


def main(container, threads, location):
    """Build `mmag`/`escript` container.

    :param container: Container where to build `mmag`.
        Accepted containers are `apptainer` and `podman`.
    :type container: str
    :param threads: Number of building threads.
    :type threads: int
    :param location: If the container `apptainer` is chosen,
        the location of the `.sif` file can be specified.
    :type location: str or pathlib.Path
    :raises ValueError: Value of `container` is not recognized.
    :raises RuntimeError: Unable to build the container.
    """
    config_path = mmag._conf_dir / "conf.json"
    if config_path.exists():
        with open(config_path, "r") as handle:
            config_dict = json.load(handle)
        container_dict = config_dict["run_escript"]
        if container in container_dict:
            installed_dir = container_dict[container].split()[-1]
            warnings.warn(
                (
                    f"The escript {container} container is already installed. "
                    f"Current installed directory: {installed_dir}"
                ),
                stacklevel=2,
            )
    else:
        config_dict = {}

    if container == "apptainer":
        if location is None:
            location = pathlib.Path.home() / ".cache" / "mammos" / "mmag" / "mmag.sif"
        else:
            location = pathlib.Path(location)
        location.parent.mkdir(exist_ok=True, parents=True)
        container_text = f"apptainer run {location}"
        cmd = get_cmd_apptainer(threads=threads, location=location)
    elif container == "podman":
        container_text = "podman run -v .:/io escript"
        cmd = get_cmd_podman(threads=threads)
    else:
        raise ValueError("Value of `container` is not recognized.")

    res = subprocess.run(cmd, stderr=subprocess.PIPE)
    if res.returncode == 0:
        config_dict["run_escript"][container] = container_text
        with open(config_path, "w") as handle:
            json.dump(config_dict, handle)
    else:
        raise RuntimeError(
            f"Unable to build the {container} container."
            "Exit with error:\n"
            f"{res.stderr.decode('utf-8')}"
        )


def get_cmd_apptainer(threads, location):
    """Get command building apptainer image.

    :param threads: Number of building threads.
    :type threads: int
    :param location: The location of the created `.sif` file.
    :type location: str or pathlib.Path
    :return: Building command to be executed.
    :rtype: list[str]
    """
    cmd = shlex.split(
        (
            "apptainer build "
            "--fakeroot "
            f"--build-arg BUILD_THREADS={threads} "
            f"{location} "
            "Apptainer.def"
        ),
        posix=IS_POSIX,
    )
    return cmd


def get_cmd_podman(threads):
    """Get command building podman image.

    :param threads: Number of building threads.
    :type threads: int
    :return: Building command to be executed.
    :rtype: list[str]
    """
    cmd = shlex.split(
        (
            "podman build "
            "-t escript "  # specify tag
            f"--build-arg BUILD_THREADS={threads} "
            "."
        ),
        posix=IS_POSIX,
    )
    return cmd


if __name__ == "__main__":
    try:
        container = sys.argv[1]
        threads = int(sys.argv[2])
    except IndexError:
        sys.exit("Wrong usage.")
    try:
        location = sys.argv[3]
    except IndexError:
        location = None

    main(container, threads, location)
