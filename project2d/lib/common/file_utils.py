import logging
import os
import pathlib
import re

rover_name_reg = re.compile(r"(?:^|[^a-z])([a-z]{1,2}\d{0,2}-\d{2,3}).*")
date_reg = re.compile(r".*[^a-z](\d{4}-\d{2}-\d{2}).*")
time_reg = re.compile(r".*[^a-z](\d{2}-\d{2}-\d{2}).*")
DEFAULT_ROVER_NAME = "RT_UNK"
DEFAULT_DATE = "D_UNK"
DEFAULT_TIME = "T_UNK"


def re_find(pattern: re.Pattern, string: str, default_value: str, logger: logging.Logger,
            message_to_log: str = "") -> str:
    result = pattern.findall(string)
    if result:
        return result[0]
    if message_to_log:
        logger.error(f"regexp cannot find {message_to_log} for {string}")
    return default_value


def scene_path_to_ride_id(scene_path: pathlib.Path | str) -> str:
    scene_path = str(scene_path)
    rover_name = re_find(rover_name_reg, scene_path, DEFAULT_ROVER_NAME, "rover_name")
    date = re_find(date_reg, scene_path, DEFAULT_DATE, "date")
    time_str = re_find(time_reg, scene_path, DEFAULT_TIME, "time")
    return f"{rover_name}_logs_{date}_{time_str}"


def _index_frames(dir_path: str, suffix: str) -> dict[int, str]:
    """
    Build {frame_number: full_path} from files like frame_000123.<suffix>.
    Ignores non-matching names.
    """
    out: dict[int, str] = {}

    if not os.path.isdir(dir_path):
        return out

    wanted_ext = f".{suffix}"

    for name in os.listdir(dir_path):
        full_path = os.path.join(dir_path, name)
        if not os.path.isfile(full_path):
            continue

        root, ext = os.path.splitext(name)
        if ext != wanted_ext:
            continue

        # Expect: frame_000123
        if not root.startswith("frame_"):
            continue

        num_str = root[6:]  # after "frame_"
        if len(num_str) != 6 or not num_str.isdigit():
            continue

        out[int(num_str)] = full_path

    return out

def generate_paths_in_window(root_dir: str, start: int, end: int, suffix1: str = "lidar", suffix2: str = "meta", ext1: str = "pcd", ext2: str = "json") -> tuple[list[str], list[str]]:
    paths1, paths2 = [], []
    
    for num in range(start, end):
        
        pcd_by_frame = _index_frames(os.path.join(root_dir, suffix1), ext1)
        pose_by_frame = _index_frames(os.path.join(root_dir, suffix2),ext2)

        common_frames = sorted(pcd_by_frame.keys() & pose_by_frame.keys())
        common_frames = [f for f in common_frames if start <= f < end]

        paths1 = [str(pcd_by_frame[f]) for f in common_frames]
        paths2 = [str(pose_by_frame[f]) for f in common_frames]

    return paths1, paths2
