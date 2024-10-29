import json
import os
import shlex
import shutil
import subprocess
import hystmag


def install_escript_podman(threads):
    podman_path = shutil.which("podman")
    if not podman_path:
        raise FileNotFoundError("podman cannot be accessed through PATH variable.")

    cmd = shlex.split(
        (
            "podman build -t escript "
            f"--build-arg BUILD_THREADS={threads} "
            f"{hystmag._container_scripts}"
        ),
        posix=(os.name == "posix"),
    )

    res = subprocess.run(cmd, stderr=subprocess.PIPE)

    if res.returncode == 0:
        hystmag._conf_dir.mkdir(parents=True, exist_ok=True)
        with open(f"{hystmag._conf_dir/"conf.json"}", "w") as handle:
            json.dump({"escript_container_program": "podman"}, handle)
    else:
        raise RuntimeError(
            "Unable to install the podman container. Exit with error:\n"
            f"{res.stderr.decode("utf-8")}"
        )


def install_escript_apptainer(threads):
    apptainer_path = shutil.which("apptainer")
    if not apptainer_path:
        raise FileNotFoundError("apptainer cannot be accessed through PATH variable.")

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

    res = subprocess.run(cmd, stderr=subprocess.PIPE)

    if res.returncode == 0:
        hystmag._conf_dir.mkdir(parents=True, exist_ok=True)
        with open(f"{hystmag._conf_dir/"conf.json"}", "w") as handle:
            json.dump({"escript_container_program": "apptainer"}, handle)
    else:
        raise RuntimeError(
            "Unable to install the apptainer container. Exit with error:\n"
            f"{res.stderr.decode("utf-8")}"
        )


def run_hystmag(threads, script, system):
    if not (hystmag._conf_dir / "conf.json").exists():
        raise RuntimeError(
            f"Cannot find a configuration file in {hystmag._conf_dir}. "
            "Make sure to build and install escript container, for example: "
            "hystmag build-escript --threads 8 --program apptainer"
        )
    with open(hystmag._conf_dir / "conf.json", "r") as handle:
        config = json.load(handle)

    escript_container = config["escript_container_program"]

    if escript_container == "apptainer":
        cmd = shlex.split(
            (
                "apptainer run "
                f"{hystmag._cache_dir/"escript.sif"} -t{threads} "
                f"{hystmag._sim_scripts/(script+".py")} {system}"
            ),
            posix=(os.name == "posix"),
        )

        res = subprocess.run(cmd, stderr=subprocess.PIPE)

        if res.returncode != 0:
            raise RuntimeError(
                f"Hystmag {script} execution for {system} failed "
                f"using {escript_container} escript container with error:\n"
                f"{res.stderr.decode("utf-8")}"
            )
    elif escript_container == "podman":
        cmd = shlex.split(
            (
                f"podman run -v .:/io -v {hystmag._sim_scripts}:/sim_scripts "
                f"escript -t{threads} /sim_scripts/{script}.py {system}"
            ),
            posix=(os.name == "posix"),
        )

        res = subprocess.run(cmd, stderr=subprocess.PIPE)

        if res.returncode != 0:
            raise RuntimeError(
                f"Hystmag {script} execution for {system} failed "
                f"using {escript_container} escript container with error:\n"
                f"{res.stderr.decode("utf-8")}"
            )
    else:
        raise NotImplementedError(
            f"{escript_container} escript container is not supported."
        )
