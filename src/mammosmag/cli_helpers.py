import json
import os
import shlex
import shutil
import subprocess
import warnings

import mammosmag

SIMULATION_SCRIPTS = [
    "loop",
    "exani",
    "external",
    "hmag",
    "magnetisation",
    "materials",
]


def install_escript(container, threads):
    program_path = shutil.which(container)
    if not program_path:
        raise FileNotFoundError(f"{container} cannot be accessed through PATH variable.")

    config_path = mammosmag._conf_dir.joinpath("conf.json")
    if config_path.exists():
        with open(config_path, "r") as handle:
            config_dict = json.load(handle)
        container_list = config_dict["escript_container_programs"]
        if container in container_list:
            warnings.warn(
                (f"The {container} escript container is already installed. "),
                stacklevel=2,
            )
        else:
            container_list.append(container)
    else:
        config_dict = {"escript_container_programs": [container]}

    is_posix = os.name == "posix"

    if container == "podman":
        cmd = shlex.split(
            (
                "podman build -t escript "
                f"--build-arg BUILD_THREADS={threads} "
                f"{mammosmag._container_scripts}"
            ),
            posix=is_posix,
        )
    else:
        temp_dir = mammosmag._cache_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        cmd = shlex.split(
            (
                "apptainer build -Fs --ignore-fakeroot-command "
                f"--tmpdir {temp_dir} "  # NOTE: needed when temp is mounted with nodev
                f"--build-arg BUILD_THREADS={threads} "
                f"--build-arg PATCH_DIR={mammosmag._container_scripts/'patches'} "
                f"--build-arg MAMMOSMAG_DIR={mammosmag._base_directory} "
                f"{mammosmag._cache_dir/'escript'} "
                f"{mammosmag._container_scripts/'Apptainer.def'}"
            ),
            posix=is_posix,
        )

    res = subprocess.run(cmd, stderr=subprocess.PIPE)

    if res.returncode == 0:
        mammosmag._conf_dir.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as handle:
            json.dump(config_dict, handle)
    else:
        raise RuntimeError(
            f"Unable to build and install the {container} container. Exit with error:\n"
            f"{res.stderr.decode('utf-8')}"
        )


def run_mammosmag(threads, container, script, name_system):
    config_path = mammosmag._conf_dir.joinpath("conf.json")
    if config_path.exists():
        with open(config_path, "r") as handle:
            config_dict = json.load(handle)
            container_list = config_dict["escript_container_programs"]
        if container is None:
            container = container_list[0]
        elif container not in container_list:
            raise RuntimeError(
                f"{container} escript container not configured. "
                "Make sure to build and install escript container, for example: "
                f"mammosmag build-escript --threads 8 --container {container}"
            )
    else:
        raise RuntimeError(
            f"Cannot find a configuration file in {mammosmag._conf_dir}. "
            "Make sure to build and install escript container, for example: "
            f"mammosmag build-escript --threads 8 --container {container}"
        )

    is_posix = os.name == "posix"

    if container == "apptainer":
        cmd = shlex.split(
            (
                f"apptainer run "
                f"{mammosmag._cache_dir/'escript'} -t{threads} "
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
            f"mammosmag {script} execution for {name_system} failed "
            f"using {container} escript container with error:\n"
            f"{res.stderr.decode('utf-8')}"
        )
