from fastapi import FastAPI
from fastapi import Request
from fastapi import HTTPException

from pydantic import BaseModel

import pandas as pd
import time

from model import model
from middleware import logging_middleware
from logging_config import logger
from tracing import tracer

from prometheus_fastapi_instrumentator import Instrumentator


app = FastAPI(
    title="IRIS Prediction API",
    version="2.0"
)

Instrumentator().instrument(app).expose(app)

app.middleware("http")(logging_middleware)


class IrisRequest(BaseModel):

    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def root():

    return {
        "message": "IRIS Prediction API"
    }


@app.get("/live")
def live():

    return {
        "status": "alive"
    }


@app.get("/ready")
def ready():

    return {
        "status": "ready"
    }


@app.post("/predict")
def predict(req: IrisRequest, request: Request):

    start = time.time()

    with tracer.start_as_current_span("prediction") as span:

        try:

            X = pd.DataFrame([{
                "sepal_length": req.sepal_length,
                "sepal_width": req.sepal_width,
                "petal_length": req.petal_length,
                "petal_width": req.petal_width
            }])

            prediction = model.predict(X)

            latency = round(
                (time.time() - start) * 1000,
                2
            )

            span.set_attribute(
                "latency_ms",
                latency
            )

            span.set_attribute(
                "prediction",
                int(prediction[0])
            )

            logger.info(
                f"Prediction={prediction[0]} "
                f"Latency={latency}ms "
                f"Client={request.client.host}"
            )

            return {
                "prediction": int(prediction[0])
            }

        except Exception as e:

            logger.exception(str(e))

            raise HTTPException(
                status_code=500,
                detail="Prediction failed"
            )