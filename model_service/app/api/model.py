from typing import Dict, List, Tuple, cast

from app.api.engine import ModelEngine, ModelsRegistry
from app.api.news import News
from app.models import *  # noqa
from fastapi import APIRouter
from hydra import compose, initialize
from omegaconf import OmegaConf


def input2model_data(input_data: List[News]) -> Tuple[List[str], List[str]]:
    data: List[str] = []
    labels: List[str] = []

    for sample in input_data:
        data.append(sample.title)

        if sample.label is not None:
            labels.append(sample.label)

    return data, labels


def load_ml_model() -> ModelEngine:
    initialize(config_path="../../conf")
    cfg = compose(config_name="config", return_hydra_config=True)

    dict_cfg = cast(dict, OmegaConf.to_container(cfg))

    model_choice = cfg.hydra.runtime.choices.model
    del dict_cfg["hydra"]

    ml_model = ModelsRegistry.get_model(model_choice, dict_cfg)

    ml_model.load()

    return ml_model


ml_model = load_ml_model()

model = APIRouter()


@model.post("/predict", status_code=200)
def predict(input_data: List[News]) -> Dict[str, str]:
    data, _ = input2model_data(input_data)

    return ml_model.predict(data)
