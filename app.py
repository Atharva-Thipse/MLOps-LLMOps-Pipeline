from fastapi import FastAPI, Request, HTTPException, Response, status
from pydantic import BaseModel
import pandas as pd
import time
from model import model
from middleware import logging_middleware
from logging_config import logger
from tracing import tracer
import mlflow.pyfunc
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title="IRIS Prediction API",
    version="2.0"
)

Instrumentator().instrument(app).expose(app) # Prometheus Metrics

FastAPIInstrumentor.instrument_app(app) # OpenTelemetry

app.middleware("http")(logging_middleware) # Logging Middleware

app_state = {"is_ready": False, "is_alive": True}
model = None

class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.on_event("startup")
async def startup_event():
    global model
    logger.info("Loading MLflow model...")
    try:
        model = mlflow.pyfunc.load_model("mlruns/1/models/m-39d3297caad94e22a2f4ca42d973501c/artifacts")
        app_state["is_ready"] = True
        logger.info("Model loaded successfully.")

    except Exception as e:
        logger.exception(f"Model loading failed: {e}")
        app_state["is_alive"] = False
        raise
        
@app.get("/")
def root():
    return {"message": "IRIS Prediction API"}

@app.get("/live", tags=["Probe"])
async def live():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/ready", tags=["Probe"])
async def ready():
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

@app.post("/predict")
def predict(req: IrisRequest, request: Request):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is still loading."
        )

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
            latency = round((time.time() - start) * 1000, 2)

            span.set_attribute("latency_ms", latency)
            span.set_attribute("prediction", int(prediction[0]))
            span.set_attribute("client.ip", request.client.host)

            logger.info(
                f"Prediction={prediction[0]} "
                f"Latency={latency}ms "
                f"Client={request.client.host}"
            )

            return {"prediction": int(prediction[0])}

        except Exception as e:
            span.record_exception(e)
            logger.exception(str(e))
            raise HTTPException(status_code=500, detail="Prediction failed")