from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml
from sklearn.pipeline import Pipeline

from .utils import get_project_root

# loading config params
project_root: Path = get_project_root()

with open(project_root / "config.yaml") as f:
    params: Dict[str, Any] = yaml.load(f, Loader=yaml.FullLoader)


def model_inference(
    model: Pipeline,
    input_text: str,
    class_names: List[str] = params["data"]["class_names"],
):
    """
    Run model inference with the given model

    :param model: Sklearn Pipeline defined in model.py
    :param input_text: any string
    :param class_names: a list of strings defining class names in the classification task
    :return: a dictionary with class names as keys and predicted scores as values.
    """

    # TODO apply processing, e.g. trimming up to `max_text_length_words` param
    pred_probs: np.array = model.predict_proba([input_text]).squeeze().round(4)
    response_dict: Dict[str, str] = dict(zip(class_names, map(str, pred_probs.tolist())))

    return response_dict
