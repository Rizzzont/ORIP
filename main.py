from contextlib import asynccontextmanager
from typing import List

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel, Field


onnx_session = None


class IrisRequestData(BaseModel):
    sepal_length: float = Field(
        ..., gt=0, description="Длина чашелистика в см", examples=[5.1]
    )
    sepal_width: float = Field(
        ..., gt=0, description="Ширина чашелистика в см", examples=[3.5]
    )
    petal_length: float = Field(
        ..., gt=0, description="Длина лепестка в см", examples=[1.4]
    )
    petal_width: float = Field(
        ..., gt=0, description="Ширина лепестка в см", examples=[0.2]
    )


class IrisResponseData(BaseModel):
    predicted_class: int = Field(
        ..., description="Индекс предсказанного класса: 0, 1 или 2"
    )
    probabilities: List[float] = Field(
        ..., description="Вероятности принадлежности к классам"
    )
    model_status: str = Field(
        ..., description="Версия модели, использованная для инференса"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global onnx_session

    print("Инициализация сервера: загрузка графа ONNX в оперативную память...")
    onnx_session = ort.InferenceSession(
        "iris_model.onnx",
        providers=["CPUExecutionProvider"],
    )

    yield

    print("Остановка сервера: освобождение ресурсов памяти...")
    onnx_session = None


app = FastAPI(lifespan=lifespan)


@app.post("/v1/predict", response_model=IrisResponseData)
def predict(data: IrisRequestData):
    input_array = np.array(
        [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width,
        ]],
        dtype=np.float32,
    )

    input_name = onnx_session.get_inputs()[0].name
    raw_predictions = onnx_session.run(None, {input_name: input_array})

    predicted_class = int(raw_predictions[0][0])
    raw_probabilities = raw_predictions[1][0]

    # ONNX-классификатор может вернуть вероятности словарём или массивом.
    if isinstance(raw_probabilities, dict):
        probabilities = [
            float(raw_probabilities[key])
            for key in sorted(raw_probabilities)
        ]
    else:
        probabilities = np.asarray(raw_probabilities, dtype=float).tolist()

    return IrisResponseData(
        predicted_class=predicted_class,
        probabilities=probabilities,
        model_status="iris_onnx_v1.0",
    )