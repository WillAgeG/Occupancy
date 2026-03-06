import argparse
import os
import shlex
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Running DDP training")
    parser.add_argument(
        "--cfg",
        default="",
        help="path to config file",
        type=str,
    )
    return parser.parse_args()


def run_cmd(cmd: str, env: dict[str, str] | None = None) -> None:
    assert isinstance(cmd, str), cmd
    env = env or {}
    subprocess.check_call(shlex.split(cmd), env=(os.environ | env))


def main() -> None:
    """Launch distributed training using all GPUs."""
    args = parse_args()
    command = f"torchrun --nproc_per_node=2 src/train.py --cfg {args.cfg}"
    run_cmd(command)


if __name__ == "__main__":
    main()
