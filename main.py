from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import List
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import pandas as pd
import uvicorn
from pick_regno import pick_regno
import os

MODEL_PATH = "micromodel.cbm"

app = FastAPI(title="PickRegno API")


# Входные данные (один элемент)
class InputData(BaseModel):
    camera_regno: str
    nn_regno: str
    camera_score: float
    nn_score: float
    nn_sym_scores: str
    nn_len_scores: str
    camera_type: str
    camera_class: str
    time_check: str
    direction: str

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )

@app.post("/predict")
async def predict(data: List[InputData]):
    results = []

    for item in data:
        prob = pick_regno(
            camera_regno=item.camera_regno,
            nn_regno=item.nn_regno,
            camera_score=item.camera_score,
            nn_score=item.nn_score,
            nn_sym_scores=item.nn_sym_scores,
            nn_len_scores=item.nn_len_scores,
            camera_type=item.camera_type,
            camera_class=item.camera_class,
            time_check=item.time_check,
            direction=item.direction,
            model_name=MODEL_PATH
        )
        results.append(prob.tolist())

    return {"results": results}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)