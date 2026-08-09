# 21F3002319_MLOPS_WEEKLY_ASSIGNMENT Week 8
data folder - contains iris.csv, train/test datasets, and poisoned datasets.

app.py - FastAPI application exposing the IRIS prediction API, health probes, Prometheus metrics, and prediction endpoint.

prepare_data.ipynb - splits the dataset into train and test datasets.

train.ipynb - trains decision tree model on train dataset.

middleware.py - HTTP middleware for request/response logging.

logging_config.py - Configures application logging for Google Cloud Logging.

inference.ipynb - evaluates the model on test dataset.

tracing.py - Configures OpenTelemetry tracing and Google Cloud Trace integration.

Dockerfile - Builds the container image for deploying the FastAPI application to GKE.

scripts/post.lua - Lua script used by wrk to generate POST requests to the /predict endpoint for high-concurrency stress testing.

scripts/poison_data.py - Generates poisoned IRIS training datasets at different corruption levels (5%, 10%, and 50%) while maintaining a clean baseline.

scripts/train_mlflow.py - Trains models on the clean and poisoned datasets, calculates accuracy, precision, recall, and F1 score, and logs the experiments to MLflow.

### Feast Components
#### Entity
iris_id: Unique identifier for each iris sample.

#### Data Source
Local Parquet file (iris.parquet): Contains all feature values and event timestamps.

#### Feature View
Features registered:
  * sepal_length
  * sepal_width
  * petal_length
  * petal_width

### Execute unit tests
`pytest`
The test suite validates:
* Dataset schema
* Missing values
* Feature ranges
* Model accuracy
* Model precision

### Continuous Integration Workflow

The GitHub Actions workflow is automatically triggered on:
* Every push
* Every pull request

The workflow performs the following steps:
1. Checkout repository
2. Install Python dependencies
3. Authenticate with Google Cloud
4. Pull versioned data and models using DVC
5. Execute pytest
6. Generate a CML report
7. Publish the report as a Pull Request comment

### Week 8 MLflow

The poisoning experiments are tracked in MLflow under the iris-week8-data-poisoning experiment.

The following metrics are recorded:
* Accuracy
* Precision
* Recall
* F1 score

This allows the impact of increasing data poisoning severity to be compared across experiments.

### k8s
* deployment.yaml - Defines Docker image, replicas, resource requests, CPU limits, etc.
* service.yaml - Creates a Kubernetes LoadBalancer that exposes the application externally.
* hpa.yaml - Defines the Horizontal Pod Autoscaler. It specifies min replicas, max replicas, and CPU utilization threshold.

### Output
#### Poisoned Datasets
The Week 8 experiment generates:

* train_0pct_poisoned.csv - Clean training dataset.
* train_5pct_poisoned.csv - 5% poisoned training dataset.
* train_10pct_poisoned.csv - 10% poisoned training dataset.
* train_50pct_poisoned.csv - 50% poisoned training dataset.

The poisoned samples contain randomly generated feature values and randomly assigned class labels. The clean test dataset is kept unchanged so that all models are evaluated against the same baseline.
