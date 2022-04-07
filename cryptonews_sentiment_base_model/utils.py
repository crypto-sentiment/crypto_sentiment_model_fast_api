import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

import yaml


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


@contextmanager
def timer(name):
    """
    A context manager to report running times.
    Example usage:
        ```
        with timer("Any comment here"):
            pass
        ```
    :param name: any string
    :return: None
    """
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    print(get_project_root().absolute())
    params = load_config_params()
    print(params)
