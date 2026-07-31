from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

trace.set_tracer_provider(TracerProvider())

processor = BatchSpanProcessor(
    CloudTraceSpanExporter()
)

trace.get_tracer_provider().add_span_processor(processor)

tracer = trace.get_tracer("iris-api")