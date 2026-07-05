# 21F3002319_MLOPS_WEEKLY_ASSIGNMENT Week 3
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

### Output
The pipeline produces:
  * feature_repo: Feast Registry
  * online_store.db: SQLite Online Store
  * Trained Decision Tree model
  * Predictions generated using online features fetched from Feast
