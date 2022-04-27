from app.api.news import News
from typing import List
from fastapi import APIRouter
from app.db import MockDb


db = APIRouter()

mock_db = MockDb()


@db.get("/", response_model=List[News])
async def index():
    return mock_db.data
