import hydra
from omegaconf import DictConfig, OmegaConf

from cryptonews_sentiment_analysis_app.data import read_train_data
from cryptonews_sentiment_analysis_app.model.engine import initialize_model
from cryptonews_sentiment_analysis_app.utils import get_project_root


@hydra.main(config_path="../../", config_name="config")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg)
    project_root = get_project_root()
    cfg_data = cfg['data']

    model = initialize_model(cfg)

    # with timer("Reading and processing data", logger=logger):
    path_to_data = project_root / \
        cfg_data["path_to_data"] / cfg_data["train_filename"]
    train_df = read_train_data(path_to_data=path_to_data)
    train_texts = train_df[cfg_data["text_field_name"]]
    train_targets = train_df[cfg_data["label_field_name"]]

    # with timer("Training the model", logger=logger):
    model.fit(train_texts, train_targets, cfg)
    model.save()

if __name__ == '__main__':
    main()
