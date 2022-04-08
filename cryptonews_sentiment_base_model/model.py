from typing import Any, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def initialize_model(model_params: Dict[str, Any]) -> Pipeline:
    """
    Initializes the model, an Sklearn Pipeline with two steps: tf-idf and logreg.
    :param model_params: a dictionary read from the `config.yml` file, section "model"
    :return: an Sklearn Pipeline object
    """

    # TODO define a model wrapper class instead
    tf_idf_params = model_params["tfidf"]
    logreg_params = model_params["logreg"]

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
