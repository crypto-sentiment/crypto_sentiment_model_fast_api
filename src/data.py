from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from src.utils import get_project_root

# loading config params
project_root: Path = get_project_root()
with open(str(project_root / "config.yml")) as f:
    params: Dict[str, Any] = yaml.load(f, Loader=yaml.FullLoader)


def read_train_data(params: Dict[str, Any]) -> pd.DataFrame:
    """
    A custom function that reads training data from CSV files into Pandas DataFrames
    :param params: a dictionary read from the config.yml file
    :return: Dataframe with training data
    """

    # read and process the training set
    # TODO apply trimming up to `max_text_length_words` param
    df = pd.read_csv(
        Path(project_root / params["data"]["path_to_data"])
        / params["data"]["train_filename"]
    )

    return df


if __name__ == "__main__":
    train_df = read_train_data(params=params)
    print(train_df.head(2))
