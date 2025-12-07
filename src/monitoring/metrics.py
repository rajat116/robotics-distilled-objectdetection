from prometheus_client import Counter, Histogram

REQUEST_COUNTER = Counter(
    "inference_requests_total",
    "Total inference requests",
)

LATENCY_HISTOGRAM = Histogram(
    "inference_latency_seconds",
    "Inference latency in seconds",
)
