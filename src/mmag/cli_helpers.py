"""CLI Helper functions."""

import json
import os
import shlex
import shutil
import subprocess
import warnings

import mmag

SIMULATION_SCRIPTS = [
    "loop",
    "exani",
    "external",
    "hmag",
    "magnetisation",
    "materials",
]


def install_escript(container, threads):
    """Install mmag software.

    :param threads: Number of building threads
    :type threads: int
    :param container: Container name
    :type container: str
    :raises FileNotFoundError: Container not found
    :raises RuntimeError: Build failed
    """
    container_path = shutil.which(container)
    if not container_path:
        raise FileNotFoundError(
            f"{container} cannot be accessed through PATH variable."
        )

    config_path = mmag._conf_dir.joinpath("conf.json")
    if config_path.exists():
        with open(config_path, "r") as handle:
            config_dict = json.load(handle)
        container_list = config_dict["escript_containers"]
        if container in container_list:
            warnings.warn(
                (f"The escript {container} container is already installed."),
                stacklevel=2,
            )
        else:
            container_list.append(container)
    else:
        config_dict = {"escript_containers": [container]}

    is_posix = os.name == "posix"

    if container == "podman":
        cmd = shlex.split(
            (
                "podman build -t escript "
                f"--build-arg BUILD_THREADS={threads} "
                f"{mmag._container_scripts}"
            ),
            posix=is_posix,
        )
    else:
        temp_dir = mmag._cache_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        cmd = shlex.split(
            (
                "apptainer build -Fs --ignore-fakeroot-command "
                f"--tmpdir {temp_dir} "  # NOTE: needed when temp is mounted with nodev
                f"--build-arg BUILD_THREADS={threads} "
                f"--build-arg PATCH_DIR={mmag._container_scripts/'patches'} "
                f"--build-arg MMAG_DIR={mmag._base_directory} "
                f"{mmag._cache_dir/'escript'} "
                f"{mmag._container_scripts/'Apptainer.def'}"
            ),
            posix=is_posix,
        )

    res = subprocess.run(cmd, stderr=subprocess.PIPE)

    if res.returncode == 0:
        mmag._conf_dir.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as handle:
            json.dump(config_dict, handle)
    else:
        raise RuntimeError(
            f"Unable to install the {container} container. Exit with error:\n"
            f"{res.stderr.decode('utf-8')}"
        )


def run_mmag(threads, container, script, name_system):
    """Run mmag software.

    :param threads: Number of running threads
    :type threads: int
    :param container: Container name
    :type container: str
    :param script: Python file or pre-defined script name to execute
    :type script: str
    :param name_system: Name of simulation system
    :type name_system: str
    :raises RuntimeError: Container not configured
    :raises RuntimeError: Configuration file not found
    :raises RuntimeError: Execution failed
    """
    config_path = mmag._conf_dir.joinpath("conf.json")
    if config_path.exists():
        with open(config_path, "r") as handle:
            config_dict = json.load(handle)
            container_list = config_dict["escript_containers"]
        if container is None:
            container = container_list[0]
        elif container not in container_list:
            raise RuntimeError(
                f"{container} escript container not configured. "
                "Make sure to build and install escript container, for example: "
                f"mmag build-escript --threads 8 --container {container}"
            )
    else:
        raise RuntimeError(
            f"Cannot find a configuration file in {mmag._conf_dir}. "
            "Make sure to build and install escript container, for example: "
            f"mmag build-escript --threads 8 --container {container}"
        )

    is_posix = os.name == "posix"

    if container == "apptainer":
        cmd = shlex.split(
            (
                f"apptainer run "
                f"{mmag._cache_dir/'escript'} -t{threads} "
            ),
            posix=is_posix,
        )

    elif container == "podman":
        cmd = shlex.split(
            (
                f"podman run "
                "-v .:/io "
                "-v $PWD:/sim_scripts "
                f"escript -t{threads}"
            ),
            posix=is_posix,
        )

    if script in SIMULATION_SCRIPTS:
        cmd.append(
            f"{mammosmag._sim_scripts / (script+'.py')} {name_system}"
        )
    else:
        cmd.append(
            f"{script}"
        )

    res = subprocess.run(cmd, stderr=subprocess.PIPE)

    if res.returncode != 0:
        raise RuntimeError(
            f"mmag {script} execution for {name_system} failed "
            f"using {container} escript container with error:\n"
            f"{res.stderr.decode('utf-8')}"
        )
