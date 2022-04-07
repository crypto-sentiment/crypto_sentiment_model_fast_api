import pickle
from pathlib import Path
from typing import Any, Dict

import yaml
from fastapi import FastAPI
from fastapi_health import health
from sklearn.pipeline import Pipeline

from .inference import model_inference
from .utils import get_project_root

# loading config params
project_root: Path = get_project_root()

with open(project_root / "config.yaml") as f:
    params: Dict[str, Any] = yaml.load(f, Loader=yaml.FullLoader)


# TODO move the following two methods to the model wrapper to be defined in model.py
# loading the model into memory
def load_model(model_path):
    with open(model_path, "rb") as f:
        model: Pipeline = pickle.load(f)
    return model


model = load_model(params["model"]["path_to_model"])


# check if model is loaded correctly
def is_model_loaded() -> bool:
    return model is not None


# initializing the API, see https://fastapi.tiangolo.com/tutorial/first-steps/
app = FastAPI()
app.add_api_route("/health", health([is_model_loaded]))


@app.get("/")
def get_classifier_details() -> Dict[str, Any]:
    """
    Gets classifier's name and model version by reading it from the config file.
    :return: Classifier's name and model version
    """

    return {
        "name": params["model"]["name"],
        "model_version": params["model"]["version"],
    }


@app.post("/classify", status_code=200)
def classify_content(
    input_json: Dict[str, str], text_field_name: str = params["data"]["text_field_name"]
) -> Dict[str, str]:
    """
    Gets a JSON with text fields, processes them, runs model prediction
    and returns the resulting predicted probabilities for each class.
    :return: a Response with a dictionary mapping class names to predicted probabilities
    """

    # TODO implement error handling, see https://fastapi.tiangolo.com/tutorial/handling-errors/
    assert text_field_name in input_json

    return model_inference(model=model, input_text=input_json.get(text_field_name, ""))
