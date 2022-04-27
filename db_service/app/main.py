from fastapi import FastAPI

from app.api.db import db

app = FastAPI()

app.include_router(db)
