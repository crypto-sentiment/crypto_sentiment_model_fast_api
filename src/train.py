import pickle
from pathlib import Path
from typing import List

import yaml
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.data import read_train_data
from src.model import initialize_model
from src.utils import get_project_root, timer

# loading config params
project_root: Path = get_project_root()

with open(str(project_root / "config.yml")) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
    cross_val_params = params["cross_validation"]


def train_model(
    train_texts: List[str], train_targets: List[int], cross_val: bool = False
) -> Pipeline:
    """
    Trains the model defined in model.py with the optional flag to add cross-validation.
    :param train_texts: a list of texts to train the model, the model is an sklearn Pipeline
                        with tf-idf as a first step, so raw texts can be fed into the model
    :param train_targets: a list of targets (ints)
    :param cross_val: whether to perform cross-validation
    :return: model – the trained model, an Sklearn Pipeline object
    """

    with timer("Training the model"):
        model = initialize_model(params)
        model.fit(X=train_texts, y=train_targets)

    if cross_val:
        with timer("Cross-validation"):
            skf = StratifiedKFold(
                n_splits=cross_val_params["cv_n_splits"],
                shuffle=cross_val_params["cv_shuffle"],
                random_state=cross_val_params["cv_random_state"],
            )

            # Running cross-validation
            cv_results = cross_val_score(
                estimator=model,
                X=train_texts,
                y=train_targets,
                cv=skf,
                n_jobs=cross_val_params["cv_n_jobs"],
                scoring=cross_val_params["cv_avg_f1_scoring"],
            )

            avg_cross_score = round(100 * cv_results.mean(), 2)
            print(
                "Average cross-validation {}: {}%.".format(
                    cross_val_params["cv_avg_f1_scoring"], avg_cross_score
                )
            )
    return model


if __name__ == "__main__":

    with timer("Reading and processing data"):
        train_df = read_train_data(params=params)

    with timer("Training the model"):
        model = train_model(
            train_texts=train_df[params["data"]["text_field_name"]],
            train_targets=train_df[params["data"]["label_field_name"]],
            cross_val=cross_val_params["cv_perform_cross_val"],
        )

    # Saving the model as a pickle file
    with open(params["model"]["path_to_model"], "wb") as f:
        pickle.dump(model, f)
