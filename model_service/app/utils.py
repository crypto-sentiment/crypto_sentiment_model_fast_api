import logging
import logging.config
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Callable
from importlib import import_module


def get_project_root() -> Path:
    """
    Return a Path object pointing to the project root directory.

    :return: Path
    """
    return Path(__file__).parent.parent.parent


def get_logger(name) -> logging.Logger:
    """

    :param name: name – any string
    :return: Python logging.Logger object
    """

    return logging.getLogger(name)


@contextmanager
def timer(name, logger):
    """
    A context manager to report running times.
    Example usage:
        ```
        with timer("Any comment here"):
            pass
        ```
    :param name: any string
    :param logger: logging object
    :return: None
    """
    t0 = time.time()
    yield
    logger.info(f"[{name}] done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    print(get_project_root().absolute())
