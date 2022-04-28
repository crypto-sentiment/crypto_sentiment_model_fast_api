import torch
from typing import Dict, Any, Iterable

from torch.utils.data import Dataset
from .utils import build_object
from typing import Optional
from sklearn.preprocessing import LabelEncoder


class FinNewsDataset(Dataset):
    def __init__(self, encodings: Dict[str, Any], labels: Optional[list] = None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self) -> int:
        return len(self.encodings)


def prepare_dataset(cfg: Dict[str, Any], data: Iterable[Any], labels: Optional[list] = None) -> Dataset:

    tokenizer = build_object(cfg["tokenizer"], is_hugging_face=True)

    encodings = tokenizer(data, truncation=True, padding=True)

    if labels is not None:
        le = LabelEncoder()

        labels = le.fit_transform(labels)

    return FinNewsDataset(encodings, labels)
