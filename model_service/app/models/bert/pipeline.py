from .utils import build_object
from datasets import load_metric
from typing import Dict, Any
from torch import Tensor
import torch

from transformers import get_scheduler
from transformers.modeling_outputs import SequenceClassifierOutput
import pytorch_lightning as pl


class SentimentPipeline(pl.LightningModule):
    """Class for training text classification models"""

    def __init__(self, cfg: Dict[str, Any], num_training_steps: int):
        super().__init__()

        self.cfg = cfg
        self.model = build_object(cfg["model"], is_hugging_face=True)
        self.metric = load_metric("accuracy")
        self.num_training_steps = num_training_steps

    def configure_optimizers(self):
        optimizer = build_object(self.cfg["optimizer"], params=self.model.parameters())

        lr_scheduler = get_scheduler(
            optimizer=optimizer,
            num_training_steps=self.num_training_steps,
            **self.cfg["scheduler"]["params"],
        )

        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, batch: Dict[str, Tensor]):
        return self.model(**batch)

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:

        outputs: SequenceClassifierOutput = self.model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        self.metric.add_batch(predictions=predictions, references=batch["labels"])

        self.log(
            "train_acc",
            self.metric.compute()["accuracy"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )

        self.log(
            "train_loss",
            outputs.loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )

        return outputs.loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:

        with torch.no_grad():
            outputs = self.model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        self.metric.add_batch(predictions=predictions, references=batch["labels"])

        self.log(
            "val_acc",
            self.metric.compute()["accuracy"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )

        self.log(
            "val_loss",
            outputs.loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )
