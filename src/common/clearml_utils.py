import pathlib

import clearml
from sat_mlops import experiments


def is_running_on_clearml_agent() -> bool:
    try:
        task = clearml.Task.current_task()
        if task is None:
            return False
        return not task.running_locally()
    except Exception as e:  # noqa: BLE001
        print(e)
        return False


def init_task(task_type=clearml.TaskTypes.training, tags=None, name="NavioOccupancy"):
    if tags is None:
        tags = ["train"]
    task, cml_logger = experiments.clearml_init_task(
        name=name,  # ignored if started via cli
        team=experiments.Team.PERCEPTION,
        project="perception/occupancy/NavioOccupancy",  # ignored if started via cli
        task_type=task_type,
        tags=tags,
    )

    return task, cml_logger


def get_remote_weights(model_id: str) -> pathlib.Path:
    weights_f = clearml.InputModel(model_id).get_weights()
    weights_f = pathlib.Path(weights_f)
    return weights_f


def get_task():
    try:
        return clearml.Task.current_task()
    except Exception as e:  # noqa: BLE001
        print(e)
        return None
