from typing import Callable, Dict, Iterable, List, Optional
from abc import ABC, abstractmethod
import pickle
from cryptonews_sentiment_analysis_app.utils import object_from_dict, get_project_root, ModelConfigFields

CLASS_NAME = str
CLASS_PROBABILITY = float


def build_model_from_config(cfg: Dict) -> Dict[str, Callable]:
    models = {}
    for model_name in cfg:
        model_cfg = cfg[model_name]
        default_kwargs = None
        if "type" in model_cfg and model_name.startswith(ModelConfigFields.MODEL_PREFIX):
            if model_name == ModelConfigFields.PIPELINE:
                default_kwargs = {"steps": list(models.items())}
            if model_name == ModelConfigFields.MODEL_ENGINE:
                default_kwargs = {"models_from_config": models}
            model = object_from_dict(model_cfg, default_kwargs=default_kwargs)
            models[model_name] = model
    return models


class ModelEngine(ABC):
    @abstractmethod
    def fit(self, X: Iterable, y: Iterable, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def predict(self, X: Iterable) -> Dict[CLASS_NAME, CLASS_PROBABILITY]:
        pass

    @abstractmethod
    def save(self, path: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def load(self, path: Optional[str] = None) -> None:
        pass


def initialize_model(cfg: Dict) -> ModelEngine:
    model = build_model_from_config(cfg)['model_engine']
    return model


class TfidfLogisticRegression(ModelEngine):
    def __init__(
        self,
        model: str,
        class_names: List[CLASS_NAME],
        models_from_config: Dict[str, Callable],
        path_to_model: str,
        name: str,
        version: str,
        round_up_to: int
    ):
        self.sklearn_estimator: Callable = models_from_config[model]
        self.class_names: List[CLASS_NAME] = class_names
        self.path_to_model = get_project_root() / path_to_model
        self.name = name
        self.version = version
        self.round_up_to = round_up_to

    def fit(self, X: Iterable, y: Iterable, cfg: Dict) -> None:
        self.sklearn_estimator.fit(X, y)

        if ModelConfigFields.CROSS_VAL_PARAMS in cfg:
            cross_val_params = cfg[ModelConfigFields.CROSS_VAL_PARAMS]
            default_kwargs = {"estimator": self.sklearn_estimator,
                              "X": X, "y": y}
            cv_results = object_from_dict(
                cross_val_params,
                default_kwargs=default_kwargs
            )
            scoring = cross_val_params["scoring"]
            avg_cross_score = round(100 * cv_results.mean(), 2)
            print("Average cross-validation {}: {}%.".format(
                scoring, avg_cross_score))

    def predict(self, X: Iterable) -> Dict[CLASS_NAME, CLASS_PROBABILITY]:
        prediction = self.sklearn_estimator.predict_proba(
            X).squeeze().round(self.round_up_to)
        response_dict = dict(
            zip(self.class_names, map(str, prediction.tolist()))
        )
        return response_dict

    def save(self, path: Optional[str] = None) -> None:
        path_to_saved_model = self.path_to_model or path
        with open(path_to_saved_model, "wb") as f:
            pickle.dump(self.sklearn_estimator, f)

    def load(self, path: Optional[str] = None) -> None:
        path_to_saved_model = self.path_to_model or path
        with open(path_to_saved_model, "wb") as f:
            self.sklearn_estimator = pickle.load(f)
