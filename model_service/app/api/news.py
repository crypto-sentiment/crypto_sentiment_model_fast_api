from pydantic import BaseModel
from typing import Optional


class News(BaseModel):
    title_id: str
    title: str
    source: str
    pub_time: str
    label: Optional[str]
