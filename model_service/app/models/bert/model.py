from typing import Dict, Any, Iterable, Optional
from app.api.engine import ModelEngine, ModelsRegistry
from .dataset import prepare_dataset
from .pipeline import SentimentPipeline

from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.nn import functional as F


@ModelsRegistry.register("bert")
class Bert(ModelEngine):
    def __init__(self, cfg: Dict[str, Any]):

        self.cfg = cfg["model"]
        self.class_names = cfg["data"]["class_names"]
        self.model: Optional[SentimentPipeline] = None

    def fit(self, X: Iterable, y: Iterable, *args, **kwargs) -> None:
        pass

    def predict(self, X: Iterable) -> Dict[str, str]:

        if self.model is None:
            raise ValueError("Bert model schould be loaded before predict with load() method.")

        dataloader = self._get_dataloader(X)
        prediction = self._predict(dataloader)
        response_dict = dict(zip(self.class_names, map(str, prediction.tolist())))

        return response_dict

    def save(self, path: Optional[str] = None) -> None:
        pass

    def load(self, path: Optional[str] = None) -> None:
        filepath = path or self.cfg["path_to_model"]
        self.model = SentimentPipeline.load_from_checkpoint(filepath)

    @torch.no_grad()
    def _predict(
        self,
        dataloader: DataLoader,
    ) -> np.ndarray:

        outputs = []
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            output = self.model(**batch)

            outputs.append(output)

        logits = [torch.cat([p.logits for p in outputs])]

        prediction = F.softmax(torch.stack(logits, dim=0), dim=-1).cpu().numpy()

        torch.cuda.empty_cache()

        return prediction

    def _get_dataloader(self, dataset: Iterable) -> DataLoader:
        dataset = prepare_dataset(self.cfg, dataset)

        return DataLoader(dataset, batch_size=self.cfg["val_batch_size"], shuffle=False)
