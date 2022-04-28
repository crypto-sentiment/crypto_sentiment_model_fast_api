from app.api.model import model
from fastapi import FastAPI

app = FastAPI()

app.include_router(model)
