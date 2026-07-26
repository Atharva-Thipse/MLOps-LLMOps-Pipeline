import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = mlflow.pyfunc.load_model(
    "mlruns/1/models/m-39d3297caad94e22a2f4ca42d973501c/artifacts"
)

class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def root():
    return {"message": "IRIS Prediction API"}


@app.post("/predict")
def predict(req: IrisRequest):

    X = pd.DataFrame([{
        "sepal_length": req.sepal_length,
        "sepal_width": req.sepal_width,
        "petal_length": req.petal_length,
        "petal_width": req.petal_width
    }])

    prediction = model.predict(X)

    return {
        "prediction": prediction[0]
    }
