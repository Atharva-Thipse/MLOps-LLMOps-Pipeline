# --------DEFINE ENTITY-------- #
from feast import Entity

iris = Entity(
    name="iris_id",
    join_keys=["iris_id"],
)

# --------DEFINE DATA SOURCE-------- #
from feast import FileSource

iris_source = FileSource(
    path="data/iris.parquet",
    event_timestamp_column="event_timestamp",
)

# --------DEFINE FEATURE VIEW-------- #
from feast import FeatureView
from feast import Field
from feast.types import Float32

iris_view = FeatureView(
    name="iris_features",

    entities=[iris],

    ttl=None,

    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
    ],

    source=iris_source,
)