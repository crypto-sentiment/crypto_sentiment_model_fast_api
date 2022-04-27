from pathlib import Path


def get_project_root() -> Path:
    """
    Return a Path object pointing to the project root directory.

    :return: Path
    """
    return Path(__file__).parent.parent.parent
