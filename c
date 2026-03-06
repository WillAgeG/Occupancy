#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

from pathlib import Path
import os
import click
from git import Repo

WORKDIRPATH = Path("./").resolve()
DOCKERFILE = Path("docker/Dockerfile")
REGISTRY_PREFIX = "registry.sberautotech.ru:5574/sd/dl/"
# Setting repo_name
REPO_NAME = Repo("./").remotes.origin.url.split(".git")[0].split("/")[-1].lower()
BRANCH_NAME = Repo("./").active_branch.name.split("/")[-1].replace("-", "_").lower()
DOCKER_TAG = REPO_NAME

_yellow = "\033[1;93m"
_green = "\033[1;96m"
_red = "\033[1;91m"
_reset = "\033[0m"


@click.group()
def cli():
    pass


# -------- Building and pushing dockers -------- #
@cli.command(help="Build agent version of docker and push it to the registry")
@click.option(
    "--push", is_flag=True, default=False, help="Push docker image to registry"
)
@click.option(
    "--tag", type=str, default=BRANCH_NAME, help="image label", show_default=True
)
@click.option(
    "--verbose", "verbose", is_flag=True, default=False, help="Additional output"
)
def docker_agent(push, tag, verbose):
    # Build base docker image
    client, image_name = _build_agent(tag, verbose)

    # Push to registry
    if push:
        _push_agent(client, image_name, verbose)
    return 0


def _build_agent(tag, verbose):
    import docker

    # Create a Docker client instance
    client = docker.from_env()

    if DOCKERFILE.is_file():
        print(f"{_yellow}Building {DOCKERFILE.name}, please wait...{_reset}")
        im, response = client.images.build(
            path=str(WORKDIRPATH),
            tag=DOCKER_TAG + ":" + tag,
            dockerfile=str(DOCKERFILE),
            target="base",
            quiet=False,
        )
        for output in response:
            if "stream" in output and verbose:
                print(output["stream"].strip())
        return client, DOCKER_TAG + ":" + tag
    else:
        raise Exception(f"{_red}{DOCKERFILE} do not exist{_reset}")


def _push_agent(client, image_name, verbose):
    print(f"{_yellow}Pushing docker image{_reset}")
    # Push the tagged image to the registry
    print("New tag", REGISTRY_PREFIX + image_name)
    
    import subprocess # docker api .tag() function is broken 
    subprocess.run(["docker", "tag", image_name, REGISTRY_PREFIX + image_name])
    
    push_response = client.images.push(
        repository=REGISTRY_PREFIX + image_name, stream=True, decode=True
    )
    # Print the output from pushing the image
    for line in push_response:
        if "errorDetail" in line:
            raise Exception(f'{line["errorDetail"]}')
        if  verbose:
            if "status" in line:
                print(line["status"])

    print(f"{_green} Pushed {image_name} {_reset}")


# -------- Running local docker instance -------- #
@cli.command(help="Build and start local instance of Docker (or stop it)")
@click.option("--stop", "-s", is_flag=True, default=False, help="Stop container")
@click.option(
    "--volume",
    "-v",
    "volumes",
    multiple=True,
    help="additional mount options for container (in form /foo:/bar)",
)
@click.option(
    "--verbose", "verbose", is_flag=True, default=False, help="Additional output"
)
def docker_local(stop, volumes, verbose):
    if stop:
        _stop_local()
        return 0
    print(f"{_yellow}Running local docker instance...{_reset}")
    client, image_name = _build_local(verbose)
    _run_local(client, image_name, volumes, verbose)
    return 0


def _build_local(verbose):
    import docker

    # Create a Docker client instance
    client = docker.from_env()

    if DOCKERFILE.is_file():
        # Define the parameters
        build_args = {"UID": str(os.getuid()), "GID": str(os.getgid())}
        print(f"{_yellow}Building {DOCKERFILE.name}, please wait...{_reset}")
        im, response = client.images.build(
            path=str(WORKDIRPATH),
            tag=DOCKER_TAG + ":local",
            dockerfile=str(DOCKERFILE),
            target="local",
            quiet=False,
            buildargs=build_args,
        )
        for output in response:
            if "stream" in output and verbose:
                print(output["stream"].strip())
        return client, DOCKER_TAG + ":local"
    else:
        raise Exception(f"{_red}{DOCKERFILE} do not exist{_reset}")


def _run_local(client, image_name, volumes, verbose):
    from docker.types import DeviceRequest

    # Importing dvc config
    from configparser import ConfigParser

    DVC_CONFIG_PATH = Path(os.path.expanduser("~") + "/.config/dvc/config")
    if os.path.exists(DVC_CONFIG_PATH):
        print(f"Found DVC config by: {DVC_CONFIG_PATH}, importing cache path")
        _dvc_config = ConfigParser()
        _dvc_config.read(DVC_CONFIG_PATH)
        if "cache" in _dvc_config:
            DVC_CACHE_PATH = _dvc_config["cache"]["dir"]
            print(f"Found DVC cache: {DVC_CACHE_PATH}")
        else:
            print("Could not find cache path")
            DVC_CACHE_PATH = None
    else:
        print("Could not find dvc config file, please check your config")
        DVC_CONFIG_PATH = None

    # Importing clearml config
    CLEARML_CONFIG_PATH = Path(os.path.expanduser("~") + "/clearml.conf")
    if os.path.exists(CLEARML_CONFIG_PATH):
        print(f"Found ClearML config by: {CLEARML_CONFIG_PATH}")
    else:
        print("Could not find ClearML config file, please check your config")
        CLEARML_CONFIG_PATH = None

    mount_volumes = [
        f"{WORKDIRPATH}:/workspaces",
    ]
    if DVC_CONFIG_PATH is not None:
        mount_volumes.append(f"{DVC_CONFIG_PATH}:/home/clearml/.config/dvc/config")
        if DVC_CACHE_PATH is not None:
            mount_volumes.append(f"{DVC_CACHE_PATH}:{DVC_CACHE_PATH}")
    if CLEARML_CONFIG_PATH is not None:
        mount_volumes.append(f"{CLEARML_CONFIG_PATH}:/home/clearml/clearml.conf")
    for vol in volumes:
        # check for correct volume path
        if ":" not in vol or not Path(vol.split(":")[0]).exists():
            print(f'{_red}Mount option "{vol}" is not correct{_reset}')
            exit()
        else:
            mount_volumes.append(vol)
    print(mount_volumes)

    # Run the Docker command
    try:
        container = client.containers.run(
            image_name,  # Image name
            detach=True,
            volumes=mount_volumes,  # Mounting paths and volumes
            network="host",
            device_requests=[DeviceRequest(device_ids=["all"], capabilities=[["gpu"]])],
            name=REPO_NAME + "_" + os.environ["USER"],
            stdin_open=True,  # Open stdin for interactive mode
            ipc_mode="host"

        )
        print(f"\n {_yellow}Successfully Started continer {container.name}{_reset}\n")
        attach_command = f"docker exec -it {container.name} /bin/bash"
        print(
            f"You can attach to container using this command:\n {_green}{attach_command}{_reset}"
        )
    except Exception as e:
        print(f"\nCould not start container:\n {_red}{str(e)}{_reset}\n")
        attach_command = (
            f'docker exec -it {REPO_NAME + "_" + os.environ["USER"]} /bin/bash'
        )
        print(
            f"If container already started, you can attach to it using:\n {_green}{attach_command}{_reset}"
        )
        exit()
    return 0


def _stop_local():
    import docker

    # Create a Docker client instance
    client = docker.from_env()
    try:
        # Find the container by name and stop it
        container = client.containers.get(REPO_NAME + "_" + os.environ["USER"])
        container.stop()
        container.remove()
    except Exception as e:
        print(f"Error stopping container: {e}")


# ---------- Sending task to ClearML agents ---------- #
@click.option(
    "--queue",
    "-q",
    type=str,
    required=True,
    prompt=True,
    help="Name or ID оf requied queue, If not provided, a task is created but not launched",
)
@click.option(
    "--script",
    "-s",
    type=str,
    required=False,
    default="run.py",
    help="Path to python script to be executed (Relative to repo!)",
)
@click.option(
    "--project",
    "-p",
    type=str,
    required=True,
    prompt=True,
    help="ClearML project name (e.g. perception/examples)",
)
@click.option(
    "--name",
    "-n",
    type=str,
    required=True,
    prompt=True,
    help="Name of your experinment",
)
@click.option(
    "--docker_tag",
    type=str,
    default=BRANCH_NAME,
    help="docker image label",
    show_default=True,
)
@click.option(
    "--docker_init",
    type=str,
    default="docker/init.sh",
    help="docker pre-run init script",
    show_default=True,
)
@click.option(
    "--flag", "-f",
    type = str,
    multiple=True,
    help="additional flags to your python script (foo=bar -> -foo bar)",
)
@cli.command(help="Send task to clearml")
def send_task(docker_init, docker_tag, queue, script, project, name, flag):
    import subprocess

    print("Sending task...")

    ##TODO: check for docker existance in registry
    print("\033[72m Dont forget to push commited changes \033[0m")
    image_name = REGISTRY_PREFIX + DOCKER_TAG + ":" + docker_tag
    command = f'clearml-task --packages "" --project {project} --name {name} --script {script} --queue {queue} --docker {image_name} --docker_bash_setup_script {docker_init}'
    if len(flag) > 0:
        command += " --args"
        for f in flag:
            command += f' {f}'

    print(command)
    subprocess.run(command, shell=True)
    return 0


if __name__ == "__main__":
    cli()
