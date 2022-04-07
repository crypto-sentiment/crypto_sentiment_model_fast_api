import time
from contextlib import contextmanager
from pathlib import Path


def get_project_root() -> Path:
    """
    Return a Path object pointing to the project root directory.

    :return: Path
    """
    return Path(__file__).parent.parent


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
