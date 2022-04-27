# from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine
from pathlib import Path
from typing import List, Dict, Any
import mmh3
from app.utils import get_project_root
import pandas as pd


class MockDb:
    def __init__(self) -> None:
        self.data: List[Dict[str, Any]] = [
            {
                "title_id": mmh3.hash("BTC dropbs by 10% today.", seed=17),
                "title": "BTC dropbs by 10% today.",
                "source": "bitcointicker",
                "pub_time": "2022-04-28",
                "label": "Positive",
            },
            {
                "title_id": mmh3.hash("BTC dropbs by 10% today.", seed=17),
                "title": "BTC dropbs by 10% today.",
                "source": "bitcointicker",
                "pub_time": "2022-04-28",
                "label": "Neutral",
            },
            {
                "title_id": mmh3.hash("BTC dropbs by 10% today.", seed=17),
                "title": "BTC dropbs by 10% today.",
                "source": "bitcointicker",
                "pub_time": "2022-04-28",
                "label": "Neutral",
            },
        ]

        self._load_init_data()

    def _load_init_data(self):
        project_root = get_project_root()

        toy_data_filepath = project_root / "data/toy_train.csv"
        if toy_data_filepath.exists():
            df = pd.read_csv(toy_data_filepath)

            for _, row in df.iterrows():
                self.data.append(
                    {
                        "title_id": mmh3.hash(row["title"], seed=17),
                        "title": row["title"],
                        "pub_time": "2022-04-28",
                        "source": "bitcointicker",
                        "label": row["sentiment"],
                    }
                )

    def add_data(self):
        pass
