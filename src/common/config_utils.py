import importlib.util
import json
import pathlib
import shutil
from dataclasses import asdict

from src.common import base_config


def load_config(cfg_path: str) -> base_config.ExperimentConfig:
    cfg_path = pathlib.Path(cfg_path).resolve()

    spec = importlib.util.spec_from_file_location(cfg_path.stem, cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.cfg


def _get_dst_path(cfg_path: str, experiment_dir: str) -> pathlib.Path:
    cfg_path = pathlib.Path(cfg_path).resolve()
    experiment_dir = pathlib.Path(experiment_dir).resolve()

    experiment_dir.mkdir(parents=True, exist_ok=True)
    dst_path = experiment_dir / cfg_path.name
    return dst_path


def copy_config_to_experiment(cfg_path: str, experiment_dir: str) -> pathlib.Path:
    """
    Coppies config in the experiment dir.
    """

    dst_path = _get_dst_path(cfg_path, experiment_dir)
    shutil.copy2(cfg_path, dst_path)

    return dst_path


def copy_json_config_to_experiment(cfg_path: str, experiment_dir: str) -> pathlib.Path:
    """
    Coppies config as a json dict to experiment.
    """
    dst_path = _get_dst_path(cfg_path, experiment_dir)
    json_dst_path = dst_path.with_suffix(".json")
    cfg = load_config(cfg_path)
    dict_cfg = asdict(cfg)

    with pathlib.Path(json_dst_path).open("w", encoding="utf-8") as f:
        json.dump(dict_cfg, f, indent=4, ensure_ascii=False, default=repr)

    return json_dst_path
