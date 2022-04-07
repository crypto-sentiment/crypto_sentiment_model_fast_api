from pathlib import Path

import yaml

from cryptonews_sentiment_base_model.data import read_train_data
from cryptonews_sentiment_base_model.utils import get_project_root

# loading config params
project_root: Path = get_project_root()

with open(project_root / "config.yaml") as f:
    params = yaml.load(f, Loader=yaml.FullLoader)


def test_presence_of_basic_data_fields():
    """
    Reads training data and checks whether in has the text fields listed in the `config.yaml` file.
    :return: None
    """

    train_df = read_train_data(params=params)

    assert params["data"]["text_field_name"] in train_df.columns
    assert params["data"]["label_field_name"] in train_df.columns
