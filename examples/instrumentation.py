"""
OpenTelemetry Instrumentation for DSPy

This module provides optional configuration for OpenTelemetry instrumentation
to trace DSPy operations and LLM calls using Phoenix.

To use this module, install the examples dependencies:
    uv sync --group examples
"""

import logging
import socket
import warnings
from typing import Any

_INSTRUMENTATION_AVAILABLE = False

try:
    import phoenix as px
    from openinference.instrumentation.dspy import DSPyInstrumentor
    from openinference.instrumentation.litellm import LiteLLMInstrumentor
    from openinference.semconv.resource import ResourceAttributes
    from opentelemetry import trace as trace_api
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk import trace as trace_sdk
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    _INSTRUMENTATION_AVAILABLE = True
except ImportError:
    pass


def is_instrumentation_available() -> bool:
    return _INSTRUMENTATION_AVAILABLE


def configure_instrumentation(
    project_name: str = "dspy-react-machina",
    endpoint: str = "http://localhost:6006/v1/traces",
) -> tuple[Any, Any] | None:
    if not _INSTRUMENTATION_AVAILABLE:
        return None

    _suppress_warnings()
    session = _launch_phoenix_if_needed()

    tracer_provider = _setup_tracer_provider(project_name, endpoint)
    _instrument_dspy()

    return session, tracer_provider


def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def _suppress_warnings() -> None:
    warnings.filterwarnings("ignore", category=Warning, module="sqlalchemy")
    logging.getLogger("phoenix").setLevel(logging.ERROR)


def _launch_phoenix_if_needed() -> Any | None:
    if _is_port_in_use(6006):
        return None
    return px.launch_app()


def _setup_tracer_provider(project_name: str, endpoint: str) -> Any:
    resource = Resource({ResourceAttributes.PROJECT_NAME: project_name})
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    trace_api.set_tracer_provider(tracer_provider=tracer_provider)
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter=OTLPSpanExporter(endpoint)))

    return tracer_provider


def _instrument_dspy() -> None:
    DSPyInstrumentor().instrument(skip_dep_check=True)
    LiteLLMInstrumentor().instrument(skip_dep_check=True)
