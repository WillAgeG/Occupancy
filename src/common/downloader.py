import pathlib
import subprocess

import yaml

from src.common import base_config


class DatasetDownloader:
    def __init__(self, cfg: base_config.DownloadConfig):
        self.entries = cfg.entries

    def __call__(self):
        for entry in self.entries:
            if entry.source_type == "s3":
                self.s3_download(entry.source, entry.target)
            elif entry.source_type == "lakefs":
                self.lakefs_download(entry.source, entry.target, entry.overwrite)
            else:
                err = f"Specified unsupported source type {entry.source_type}"
                raise NotImplementedError(err)

    @staticmethod
    def s3_download(data_path_url: str, local_path: str) -> pathlib.Path:
        print(f"Downloading s3 data {data_path_url} in {local_path} using obsutil")
        command = ["obsutil", "sync", data_path_url, local_path]  # Can put /dataset here persistency
        _ = subprocess.check_call(command)
        return pathlib.Path(local_path).resolve()

    @staticmethod
    def lakefs_download(data_path_url: str, local_path: str, overwrite: bool) -> pathlib.Path:
        if is_non_empty_dir(local_path) and not overwrite:
            print(f"Skipping download of {data_path_url}. {local_path} is not empty")
            return pathlib.Path(local_path).resolve()
        print(f"Downloading lakefs data {data_path_url} in {local_path} using lakectl")
        comand = ["lakectl", "fs", "download", "--recursive", data_path_url, local_path]
        _ = subprocess.check_call(comand)
        return pathlib.Path(local_path).resolve()

    @staticmethod
    def lakefs_pull(
        data_path_url: str,
        local_path: str,
        commit_hash: str,
    ) -> pathlib.Path:
        """
        Warning! Erases other files (that are not present in the lakefs uri) in the local_path
        """
        dest_dir = pathlib.Path(local_path).resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)

        ref_file = dest_dir / ".lakefs_ref.yaml"

        ref_data = {
            "src": data_path_url,
            "at_head": commit_hash,
            "active_operation": "",
        }

        # write .lakefs_ref.yaml
        with ref_file.open("w") as f:
            yaml.safe_dump(ref_data, f, sort_keys=False)

        print(ref_file.read_text())
        print(f"Pulling lakefs data in {dest_dir} using .lakefs_ref.yaml")

        _ = subprocess.check_call(["lakectl", "local", "checkout", "-y"], cwd=dest_dir)  # noqa: S607
        return pathlib.Path(local_path).resolve()


def download_data(cfg: base_config.DownloadConfig):
    if cfg is not None:
        downloader = DatasetDownloader(cfg)
        downloader()


def is_non_empty_dir(path: str) -> bool:
    ppath = pathlib.Path(path)
    return ppath.exists() and ppath.is_dir() and any(ppath.iterdir())
