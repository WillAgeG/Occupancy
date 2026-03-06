import errno
import logging
import os
import pathlib


def get_logger(path, filename):

    # Create the folder where the training information is to be saved if it doesn't exist
    if not pathlib.Path(path).exists():
        try:
            pathlib.Path(path).mkdir(parents=True)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # Create the logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # In order to store logs of level INFO and above
    # create file handler which logs even debug messages into logs file
    fh = logging.FileHandler(os.path.join(path, filename))
    fh.setLevel(logging.INFO)
    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s -- %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


class DummySummaryWriter:
    def add_scalar(self, *args, **kwargs): pass
    def add_scalars(self, *args, **kwargs): pass
    def add_histogram(self, *args, **kwargs): pass
    def add_image(self, *args, **kwargs): pass
    def add_images(self, *args, **kwargs): pass
    def add_figure(self, *args, **kwargs): pass
    def flush(self): pass
    def close(self): pass
