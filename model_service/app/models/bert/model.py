from typing import Any, Dict, Iterable, Optional, cast

import numpy as np
import torch
from app.api.engine import ModelEngine, ModelsRegistry
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .dataset import prepare_dataset
from .pipeline import SentimentPipeline


@ModelsRegistry.register("bert")
class Bert(ModelEngine):
    def __init__(self, cfg: Dict[str, Any]):

        self.cfg = cfg["model"]
        self.class_names = cfg["data"]["class_names"]
        self.device = torch.device(self.cfg["device"])

    def fit(self, X: Iterable, y: Iterable, *args, **kwargs) -> None:
        pass

    def predict(self, X: Iterable) -> Dict[str, str]:

        dataloader = self._get_dataloader(X)
        prediction = self._predict(dataloader)
        response_dict = dict(zip(self.class_names, map(str, prediction.tolist())))

        return response_dict

    def save(self, path: Optional[str] = None) -> None:
        pass

    def load(self, path: Optional[str] = None) -> None:
        filepath = path or self.cfg["path_to_model"]
        self.model = SentimentPipeline.load_from_checkpoint(filepath, cfg=self.cfg)
        self.model = self.model.to(self.device)

    @torch.no_grad()
    def _predict(
        self,
        dataloader: DataLoader,
    ) -> np.ndarray:
        self.model.eval()

        outputs = []
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            output = cast(SentimentPipeline, self.model).model(**batch)

            outputs.append(output)

        logits = [torch.cat([p.logits for p in outputs])]

        prediction = F.softmax(torch.stack(logits, dim=0), dim=-1).cpu().numpy().squeeze()

        return prediction

    def _get_dataloader(self, data: Iterable) -> DataLoader:
        dataset = prepare_dataset(self.cfg, data)

        return DataLoader(dataset, batch_size=self.cfg["val_batch_size"], shuffle=False)
