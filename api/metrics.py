"""
Prometheus 指标定义与中间件
"""

from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter(
    "agent_requests_total",
    "Total number of analysis requests",
    ["mode", "status"],
)

REQUEST_DURATION = Histogram(
    "agent_request_duration_seconds",
    "Request processing duration in seconds",
    ["mode"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

TOOL_CALLS = Counter(
    "agent_tool_calls_total",
    "Total number of tool invocations",
    ["tool_name", "success"],
)

ACTIVE_REQUESTS = Gauge(
    "agent_active_requests",
    "Number of requests currently being processed",
)
