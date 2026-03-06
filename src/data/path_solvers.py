import abc
import collections
import glob
import os


class AbstractPathSolver(abc.ABC):
    def __init__(self, key: str, dir_name: str, extension: str):
        super().__init__()
        self.key = key
        self.dir_name = dir_name
        self.extension = extension

    @abc.abstractmethod
    def get_filepaths(self, filepaths: dict[str: dict[str: list[str]]], sequences: list[str]):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}: key={self.key} dir_name={self.dir_name} extension={self.extension}"


class ScaledPathSolver(AbstractPathSolver):
    def get_filepaths(self, filepaths, sequences):
        for sequence in sequences:
            base = os.path.join(sequence, self.dir_name)
            all_files = sorted(glob.glob(os.path.join(base, f"*.{self.extension}*")))

            for f in sorted(all_files):
                if f.endswith((f"_1_2.{self.extension}", "_1_2")):
                    filepaths[f"{self.key}_1_2"][sequence].append(f)
                elif f.endswith((f"_1_4.{self.extension}", "_1_4")):
                    filepaths[f"{self.key}_1_4"][sequence].append(f)
                elif f.endswith((f"_1_8.{self.extension}", "_1_8")):
                    filepaths[f"{self.key}_1_8"][sequence].append(f)
                else:
                    filepaths[f"{self.key}_1_1"][sequence].append(f)


class SimplePathSolver(AbstractPathSolver):
    def get_filepaths(self, filepaths, sequences):
        for sequence in sequences:
            filepaths[f"{self.key}"][sequence] = sorted(
                glob.glob(os.path.join(sequence, self.dir_name, f"*.{self.extension}")))


class ReplacePathSolver(AbstractPathSolver):
    def __init__(self, key, dir_name, extension, replaces: list[tuple[str, str]] | None = None):
        super().__init__(key, dir_name, extension)
        self.replaces = replaces or []

    def __repr__(self):
        return super().__repr__() + f" replaces={self.replaces}"

    def _apply_replaces(self, path: str) -> str:
        for old, new in self.replaces:
            path = path.replace(old, new)
        return path

    def get_filepaths(self, filepaths, sequences):
        for sequence in sequences:
            files = sorted(glob.glob(os.path.join(sequence, self.dir_name, f"*.{self.extension}")))
            filepaths[f"{self.key}"][sequence] = [self._apply_replaces(p) for p in files]


def strip_n(filepaths: dict[str: dict[str: list[str]]], n: int, front: bool = False):
    assert n >= 0, "You should pass a non-negative number of samples to strip"
    for key in filepaths:
        for seq in filepaths[key]:
            if front:
                filepaths[key][seq] = filepaths[key][seq][n:]
            else:
                filepaths[key][seq] = filepaths[key][seq][:-n]


def strip_using_ref(filepaths: dict[str: dict[str: list[str]]], ref_key: str):
    ref_lens = {}
    for seq in filepaths[ref_key]:
        ref_lens[seq] = len(filepaths[ref_key][seq])

    for key in filepaths:
        for seq in filepaths[key]:
            filepaths[key][seq] = filepaths[key][seq][:ref_lens[seq]]


def merge_sequences(filepaths: dict[str: dict[str: list[str]]]) -> dict[str: list[str]]:
    shallow_filepaths = collections.defaultdict(list)
    for key in filepaths:
        for seq in filepaths[key]:
            shallow_filepaths[key] += filepaths[key][seq]

    return shallow_filepaths
