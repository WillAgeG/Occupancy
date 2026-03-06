import hashlib
import os
import pathlib

from common.enums import Modality


def get_md5(filename):
    """ """
    hash_obj = hashlib.md5()
    with pathlib.Path(filename).read_bytes() as f:
        hash_obj.update(f)
    return hash_obj.hexdigest()


def dict_to(_dict, device):
    for key, value in _dict.items():
        if type(value) is dict:
            _dict[key] = dict_to(_dict[key], device)
        if type(_dict[key]) is list:
            _dict[key] = [v.to(device, non_blocking=True) for v in _dict[key]]
        else:
            _dict[key] = _dict[key].to(device, non_blocking=True)

    return _dict


def remove_recursively(folder_path):
    """
    Remove directory recursively
    """
    if pathlib.Path(folder_path).is_dir():
        filelist = list(os.listdir(folder_path))  # noqa: PTH208
        for f in filelist:
            pathlib.Path(os.path.join(folder_path, f)).unlink()


def create_directory(directory):
    """
    Create directory if doesn't exists
    """
    if not pathlib.Path(directory).exists():
        pathlib.Path(directory).mkdir(parents=True)
