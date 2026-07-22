# 21F3002319_MLOPS_WEEKLY_ASSIGNMENT Week 4
data folder - contains iris.csv.

prepare_data.ipynb - splits the dataset into train and test datasets.

train.ipynb - trains decision tree model on train dataset.

inference.ipynb - evaluates the model on test dataset.

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

### Running the Project
Install dependencies
`pip install feast pandas scikit-learn pyarrow`

Apply Feast definitions
`cd feature_repo`
`feast apply`

Materialize features
`feast materialize 2020-01-01T00:00:00 <current_timestamp>`

Run the notebooks
prepare_data.ipynb
train.ipynb
inference.ipynb

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

### MLflow
mlflow.db:
* Stores experiment metadata.
mlruns/:
* Stores trained models.
* Maintains the Model Registry.

### Output
The pipeline produces:
* Integrated MLflow into an end-to-end ML pipeline.
* Performed hyperparameter tuning across multiple training runs.
* Logged parameters, metrics, and model artifacts using MLflow.
* Compared experiments through the MLflow Tracking UI.
* Managed models using the MLflow Model Registry.