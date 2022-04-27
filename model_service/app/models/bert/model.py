from typing import Dict, Any, Iterable, Optional
from app.api.engine import ModelEngine, ModelsRegistry
from .dataset import prepare_dataset, split_train_val
from .pipeline import SentimentPipeline, MetricTracker

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
import torch
import numpy as np


@ModelsRegistry.register("bert")
class Bert(ModelEngine):
    def __init__(self, cfg: Dict[str, Any]):

        self.cfg = cfg["model"]
        self.class_names = cfg["data"]["class_names"]

        self.model: Optional[SentimentPipeline] = None
        self.trainer: Optional[Trainer] = None

    def fit(self, X: Iterable, y: Iterable, *args, **kwargs) -> None:
        train_data, val_data, train_labels, val_labels = split_train_val(X, y)

        train_dataset = prepare_dataset(self.cfg, train_data, train_labels)
        val_dataset = prepare_dataset(self.cfg, val_data, val_labels)

        train_dataloader = DataLoader(train_dataset, batch_size=self.cfg["train_batch_size"], shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.cfg["val_batch_size"], shuffle=False)

        seed_everything(self.cfg["seed"])

        num_training_steps = self.cfg["epochs"] * len(train_dataloader)

        self.model = SentimentPipeline(self.cfg, num_training_steps)

        metric_tracker = MetricTracker()

        self.trainer = Trainer(
            max_epochs=self.cfg["epochs"],
            gpus=1,
            callbacks=[metric_tracker],
            num_sanity_val_steps=0,
            enable_checkpointing=False,
            logger=False,
        )

        self.trainer.fit(
            self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    def predict(self, X: Iterable) -> Dict[str, str]:

        pred_numpy = self._predict(X)

        return None

    def save(self, path: Optional[str] = None) -> None:
        filepath = path or self.cfg["path_to_model"]
        self.trainer.save_checkpoint(filepath)

    def load(self, path: Optional[str] = None) -> None:
        filepath = path or self.cfg["path_to_model"]
        self.model = SentimentPipeline.load_from_checkpoint(filepath)

    @torch.no_grad()
    def _predict(
        self,
        dataloader: DataLoader,
    ) -> np.ndarray:

        assert self.trainer is not None, "Trainer is none"

        outputs = self.trainer.predict(self.model, dataloader)

        logits = torch.cat([p.logits for p in outputs])

        pred_labels = torch.argmax(logits, dim=-1).numpy()
        torch.cuda.empty_cache()

        return pred_labels

    def _get_dataloader(self, dataset: Iterable) -> DataLoader:
        dataset = prepare_dataset(self.cfg, dataset)

        return DataLoader(dataset, batch_size=self.cfg["val_batch_size"], shuffle=False)
