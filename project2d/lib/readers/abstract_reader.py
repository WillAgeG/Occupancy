import abc


class AbstractCloudReader(abc.ABC):

    @classmethod
    @abc.abstractmethod
    def read_cloud(cls, file_path, xyz=True, **kwargs):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def read_pose(cls, file_path, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def read_label(file_path, *args, **kwargs):
        raise NotImplementedError
