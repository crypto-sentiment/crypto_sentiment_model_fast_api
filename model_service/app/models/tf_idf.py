from typing import Dict, Any, Iterable, Optional, cast
from app.api.engine import ModelEngine, ModelsRegistry

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from model_service.app.utils import get_logger, timer
from pathlib import Path
import pickle

logger = get_logger(Path(__file__).name)


@ModelsRegistry.register("tf_idf")
class TfidfLogisticRegression(ModelEngine):
    def __init__(self, cfg: Dict[str, Any]):

        self.cfg = cfg["model"]
        self.class_names = cfg["data"]["class_names"]
        self.model: Pipeline = self._initialize_model(self.cfg)

    def fit(self, X: Iterable, y: Iterable, *args, **kwargs) -> None:
        self.model.fit(X, y)

        if self.cfg["cross_validation"]["cv_perform_cross_val"]:
            cross_val_params = self.cfg["cross_validation"]

            with timer("Cross-validation", logger=logger):
                skf = StratifiedKFold(
                    n_splits=cross_val_params["cv_n_splits"],
                    shuffle=cross_val_params["cv_shuffle"],
                    random_state=cross_val_params["cv_random_state"],
                )

                # Running cross-validation
                cv_results = cross_val_score(
                    estimator=self.model,
                    X=X,
                    y=y,
                    cv=skf,
                    n_jobs=cross_val_params["cv_n_jobs"],
                    scoring=cross_val_params["cv_scoring"],
                )

                avg_cross_score = round(100 * cv_results.mean(), 2)
                print("Average cross-validation {}: {}%.".format(cross_val_params["cv_scoring"], avg_cross_score))

    def predict(self, X: Iterable) -> Dict[str, str]:
        prediction = self.model.predict_proba(X).squeeze().round(4)
        response_dict: Dict[str, str] = dict(zip(self.class_names, map(str, prediction.tolist())))

        return response_dict

    def save(self, path: Optional[str] = None) -> None:
        path_to_saved_model = cast(str, self.cfg["path_to_model"] or path)

        with open(path_to_saved_model, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: Optional[str] = None) -> None:
        path_to_saved_model = cast(str, self.cfg["path_to_model"] or path)

        with open(path_to_saved_model, "wb") as f:
            self.model = pickle.load(f)

    def _initialize_model(self, cfg: Dict[str, Any]) -> Pipeline:
        """
        Initializes the model, an Sklearn Pipeline with two steps: tf-idf and logreg.
        :param model_params: a dictionary read from the `config.yml` file, section "model"
        :return: an Sklearn Pipeline object
        """

        # TODO define a model wrapper class instead
        tf_idf_params = cfg["tfidf"]
        logreg_params = cfg["logreg"]

        # initialize TfIdf, logreg, and the Pipeline with the params from a config file
        # TODO support arbitrary params, not only the listed ones.
        text_transformer = TfidfVectorizer(
            stop_words=tf_idf_params["stop_words"],
            ngram_range=eval(tf_idf_params["ngram_range"]),
            lowercase=bool(tf_idf_params["lowercase"]),
            analyzer=tf_idf_params["analyzer"],
            max_features=int(tf_idf_params["max_features"]),
        )

        logreg = LogisticRegression(
            C=int(logreg_params["C"]),
            solver=logreg_params["solver"],
            multi_class=logreg_params["multi_class"],
            random_state=int(logreg_params["random_state"]),
            max_iter=int(logreg_params["max_iter"]),
            n_jobs=int(logreg_params["n_jobs"]),
            fit_intercept=bool(logreg_params["fit_intercept"]),
        )

        model = Pipeline([("tfidf", text_transformer), ("logreg", logreg)])

        return model
