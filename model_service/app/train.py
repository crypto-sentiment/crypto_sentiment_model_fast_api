import hydra
from omegaconf import DictConfig, OmegaConf
from app.data import read_train_data
from app.utils import get_project_root
from app.api.engine import ModelsRegistry
from typing import cast
from hydra.core.hydra_config import HydraConfig
from app.models import *


@hydra.main(config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    hydra_cfg = HydraConfig.get()
    dict_cfg = cast(dict, OmegaConf.to_container(cfg))

    project_root = get_project_root()
    cfg_data = dict_cfg["data"]

    model = ModelsRegistry.get_model(hydra_cfg.runtime.choices.model, dict_cfg)

    print(f"project_root: {project_root}")
    path_to_data = project_root / cfg_data["path_to_data"] / cfg_data["train_filename"]
    train_df = read_train_data(path_to_data=path_to_data)
    train_texts = train_df[cfg_data["text_field_name"]]
    train_targets = train_df[cfg_data["label_field_name"]]

    model.fit(train_texts, train_targets, cfg)
    model.save(project_root / dict_cfg["model"]["path_to_model"])


if __name__ == "__main__":
    main()
