import mlflow.pyfunc

# Load the Model
model = mlflow.pyfunc.load_model(
    "mlruns/1/models/m-39d3297caad94e22a2f4ca42d973501c/artifacts"
)