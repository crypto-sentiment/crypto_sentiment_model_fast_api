import logging
import logging.config
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict
from enum import Enum

import pydoc

class ModelConfigFields(str, Enum):
    PIPELINE = "model_pipeline"
    MODEL_ENGINE = "model_engine"
    MODEL_PREFIX = "model_"
    CROSS_VAL_PARAMS = "cross_val_params"

def object_from_dict(d, default_kwargs=None):
    kwargs = d.copy()
    object_type = kwargs.pop("type")
        
    if isinstance(default_kwargs, dict):
        kwargs.update(default_kwargs)

    # support nested constructions
    for key, value in kwargs.items():
        if isinstance(value, dict) and "type" in value:
            value = object_from_dict(value)
            kwargs[key] = value

    return pydoc.locate(object_type)(**kwargs)


def get_project_root() -> Path:
    """
    Return a Path object pointing to the project root directory.

    :return: Path
    """
    return Path(__file__).parent.parent


def load_config_params() -> Dict[str, Any]:
    """
    Loads global project configuration params defined in the `config.yaml` file.

    :return: a nested dictionary corresponding to the `config.yaml` file.
    """
    project_root: Path = get_project_root()
    with open(project_root / "config.yaml") as f:
        params: Dict[str, Any] = yaml.load(f, Loader=yaml.FullLoader)
    return params


def get_logger(name) -> logging.Logger:
    """

    :param name: name – any string
    :return: Python logging.Logger object
    """

    params = load_config_params()
    log_config = params["logging"]
    logging.config.dictConfig(config=log_config)

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
