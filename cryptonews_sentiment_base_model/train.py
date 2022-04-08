import pickle
from pathlib import Path
from typing import Any, Dict, List

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from .data import read_train_data
from .model import initialize_model
from .utils import get_logger, get_project_root, load_config_params, timer

logger = get_logger(Path(__file__).name)


def train_model(
    train_texts: List[str], train_targets: List[int], model_params: Dict[str, Any], cross_val_params: Dict[str, Any]
) -> Pipeline:
    """
    Trains the model defined in model.py with the optional flag to add cross-validation.
    :param train_texts: a list of texts to train the model, the model is an sklearn Pipeline
                        with tf-idf as a first step, so raw texts can be fed into the model
    :param train_targets: a list of targets (ints)
    :param model_params: a dictionary with model parameters, see the "model" section of the `config.yaml` file
    :param cross_val_params: a dictionary with cross-validation parameters,
                             see the "cross_validation" section of the `config.yaml` file
    :return: model – the trained model, an Sklearn Pipeline object
    """

    with timer("Training the model", logger=logger):
        model = initialize_model(model_params=model_params)
        model.fit(X=train_texts, y=train_targets)

    if cross_val_params["cv_perform_cross_val"]:
        with timer("Cross-validation", logger=logger):
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
                scoring=cross_val_params["cv_scoring"],
            )

            avg_cross_score = round(100 * cv_results.mean(), 2)
            logger.info("Average cross-validation {}: {}%.".format(cross_val_params["cv_scoring"], avg_cross_score))
    return model


if __name__ == "__main__":

    # loading project-wide configuration params
    params: Dict[str, Any] = load_config_params()
    project_root = get_project_root()

    with timer("Reading and processing data", logger=logger):
        path_to_data = project_root / params["data"]["path_to_data"] / params["data"]["train_filename"]
        train_df = read_train_data(path_to_data=path_to_data)

    model = train_model(
        train_texts=train_df[params["data"]["text_field_name"]],
        train_targets=train_df[params["data"]["label_field_name"]],
        model_params=params["model"],
        cross_val_params=params["cross_validation"],
    )

    # Saving the model as a pickle file
    with open(params["model"]["path_to_model"], "wb") as f:
        pickle.dump(model, f)
