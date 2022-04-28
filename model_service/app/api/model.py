from app.models import *
from app.api.engine import ModelEngine, ModelsRegistry
from app.api.news import News
from fastapi import Header, APIRouter
import mmh3
from typing import List, Dict, Tuple

from hydra import compose, initialize
from omegaconf import OmegaConf
from typing import cast

fake_db = [
    {
        "title_id": mmh3.hash("BTC dropbs by 10% today.", seed=17),
        "title": "BTC dropbs by 10% today.",
        "source": "bitcointicker",
        "pub_time": "2022-04-28",
        "label": "Positive",
    },
    {
        "title_id": mmh3.hash("BTC dropbs by 10% today.", seed=17),
        "title": "BTC dropbs by 10% today.",
        "source": "bitcointicker",
        "pub_time": "2022-04-28",
        "label": "Neutral",
    },
    {
        "title_id": mmh3.hash("BTC dropbs by 10% today.", seed=17),
        "title": "BTC dropbs by 10% today.",
        "source": "bitcointicker",
        "pub_time": "2022-04-28",
        "label": "Neutral",
    },
]


def input2model_data(input_data: List[News]) -> Tuple[List[str], List[str]]:
    data: List[str] = []
    labels: List[str] = []

    for sample in input_data:
        data.append(sample.title)

        if sample.label is not None:
            labels.append(sample.label)

    return data, labels


def load_ml_model(pretrained: bool = False) -> ModelEngine:
    initialize(config_path="../../../conf")
    cfg = compose(config_name="config", return_hydra_config=True)

    dict_cfg = cast(dict, OmegaConf.to_container(cfg))

    model_choice = cfg.hydra.runtime.choices.model
    del dict_cfg["hydra"]

    ml_model = ModelsRegistry.get_model(model_choice, dict_cfg)

    ml_model.load()

    return ml_model


ml_model = load_ml_model()

model = APIRouter()


@model.get("/", response_model=List[News])
async def index():
    return fake_db


@model.post("/predict", status_code=200)
def predict(input_data: List[News]) -> Dict[str, str]:
    data, _ = input2model_data(input_data)

    return ml_model.predict(data)
