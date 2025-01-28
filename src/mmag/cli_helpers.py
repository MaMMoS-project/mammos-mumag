import json
import os
import shlex
import shutil
import subprocess
import warnings

import mmag


def install_escript(program, threads):
    program_path = shutil.which(program)
    if not program_path:
        raise FileNotFoundError(f"{program} cannot be accessed through PATH variable.")

    config_path = mmag._conf_dir.joinpath("conf.json")
    if config_path.exists():
        with open(config_path, "r") as handle:
            config_dict = json.load(handle)
        container_list = config_dict["escript_container_programs"]
        if program in container_list:
            warnings.warn(
                (f"The {program} escript container is already installed. "),
                stacklevel=2,
            )
        else:
            container_list.append(program)
    else:
        config_dict = {"escript_container_programs": [program]}

    is_posix = os.name == "posix"

    if program == "podman":
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
            f"Unable to build and install the {program} container. Exit with error:\n"
            f"{res.stderr.decode('utf-8')}"
        )


def run_mmag(threads, program, script, system):
    config_path = mmag._conf_dir.joinpath("conf.json")
    if config_path.exists():
        with open(config_path, "r") as handle:
            config_dict = json.load(handle)
            container_list = config_dict["escript_container_programs"]
        if program is None:
            program = container_list[0]
        elif program not in container_list:
            raise RuntimeError(
                f"{program} escript container not configured. "
                "Make sure to build and install escript container, for example: "
                f"mmag build-escript --threads 8 --container {program}"
            )
    else:
        raise RuntimeError(
            f"Cannot find a configuration file in {mmag._conf_dir}. "
            "Make sure to build and install escript container, for example: "
            f"mmag build-escript --threads 8 --container {program}"
        )

    is_posix = os.name == "posix"

    if program == "apptainer":
        cmd = shlex.split(
            (
                f"apptainer run "
                f"{mmag._cache_dir/'escript'} -t{threads} "
                f"{mmag._sim_scripts/(script+'.py')} {system}"
            ),
            posix=is_posix,
        )

    elif program == "podman":
        cmd = shlex.split(
            (
                f"podman run -v .:/io -v {mmag._sim_scripts}:/sim_scripts "
                f"escript -t{threads} /sim_scripts/{script}.py {system}"
            ),
            posix=is_posix,
        )

    res = subprocess.run(cmd, stderr=subprocess.PIPE)

    if res.returncode != 0:
        raise RuntimeError(
            f"mmag {script} execution for {system} failed "
            f"using {program} escript container with error:\n"
            f"{res.stderr.decode('utf-8')}"
        )
