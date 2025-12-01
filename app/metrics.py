from prometheus_client import Counter, Histogram

# Counts how many prediction requests happened
REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
)

# Measures the latency of each prediction
REQUEST_LATENCY = Histogram(
    "prediction_request_latency_seconds",
    "Latency (seconds) for prediction requests",
)
