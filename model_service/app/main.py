from fastapi import FastAPI

from app.api.model import model

app = FastAPI()

app.include_router(model)
