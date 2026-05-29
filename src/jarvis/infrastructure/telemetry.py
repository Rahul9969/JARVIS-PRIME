"""
JARVIS-PRIME Telemetry & Observability
=========================================

Structured logging + metrics collection:
- Hierarchical logger with JSON output
- Request/response timing
- Agent performance metrics
- System resource monitoring
- Event tracing with span IDs

Phase 3: Local file-based telemetry
Phase 4+: OpenTelemetry export to Grafana/Jaeger
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Span:
    """A single operation span for distributed tracing."""
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    trace_id: str = ""
    parent_id: str | None = None
    operation: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    tags: dict[str, str] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "ok"

    def end(self, status: str = "ok") -> None:
        self.end_time = time.time()
        self.status = status

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return (time.time() - self.start_time) * 1000

    def add_event(self, name: str, **kwargs: Any) -> None:
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            **kwargs,
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "operation": self.operation,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "tags": self.tags,
            "events": self.events,
        }


class MetricsCollector:
    """
    Collects and aggregates performance metrics.
    """

    def __init__(self):
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = {}

    def increment(self, name: str, value: float = 1.0) -> None:
        """Increment a counter metric."""
        self._counters[name] = self._counters.get(name, 0) + value

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge (point-in-time) metric."""
        self._gauges[name] = value

    def record(self, name: str, value: float) -> None:
        """Record a value in a histogram."""
        if name not in self._histograms:
            self._histograms[name] = []
        self._histograms[name].append(value)
        # Keep last 1000 values
        if len(self._histograms[name]) > 1000:
            self._histograms[name] = self._histograms[name][-1000:]

    def get_counter(self, name: str) -> float:
        return self._counters.get(name, 0)

    def get_gauge(self, name: str) -> float:
        return self._gauges.get(name, 0)

    def get_histogram_stats(self, name: str) -> dict[str, float]:
        """Get statistics for a histogram."""
        values = self._histograms.get(name, [])
        if not values:
            return {"count": 0}

        import numpy as np
        arr = np.array(values)

        return {
            "count": len(values),
            "mean": round(float(arr.mean()), 4),
            "median": round(float(np.median(arr)), 4),
            "p95": round(float(np.percentile(arr, 95)), 4),
            "p99": round(float(np.percentile(arr, 99)), 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
        }

    def snapshot(self) -> dict[str, Any]:
        """Get all metrics."""
        result: dict[str, Any] = {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {},
        }
        for name in self._histograms:
            result["histograms"][name] = self.get_histogram_stats(name)
        return result


class JarvisLogger:
    """
    Structured JSON logger with severity levels.
    """

    LEVELS = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3, "CRITICAL": 4}

    def __init__(
        self,
        name: str = "jarvis",
        level: str = "INFO",
        log_dir: Path | None = None,
    ):
        self.name = name
        self.level = self.LEVELS.get(level, 1)
        self.log_dir = log_dir
        self._log_file = None

        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log_file = log_dir / f"jarvis_{time.strftime('%Y%m%d')}.jsonl"

    def _log(self, level: str, message: str, **kwargs: Any) -> None:
        if self.LEVELS.get(level, 0) < self.level:
            return

        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "level": level,
            "logger": self.name,
            "message": message,
            **kwargs,
        }

        if self._log_file:
            try:
                with open(self._log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, default=str) + "\n")
            except Exception:
                pass

    def debug(self, message: str, **kwargs: Any) -> None:
        self._log("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        self._log("INFO", message, **kwargs)

    def warn(self, message: str, **kwargs: Any) -> None:
        self._log("WARN", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self._log("ERROR", message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        self._log("CRITICAL", message, **kwargs)


class Telemetry:
    """
    Unified telemetry system.
    Combines logging, metrics, and tracing.
    """

    def __init__(self, log_dir: Path | None = None):
        self.logger = JarvisLogger(log_dir=log_dir)
        self.metrics = MetricsCollector()
        self._active_spans: dict[str, Span] = {}
        self._completed_spans: list[Span] = []
        self._start_time = time.time()

    def start_span(
        self,
        operation: str,
        trace_id: str | None = None,
        parent_id: str | None = None,
        **tags: str,
    ) -> Span:
        """Start a new tracing span."""
        span = Span(
            trace_id=trace_id or str(uuid.uuid4())[:8],
            parent_id=parent_id,
            operation=operation,
            tags=tags,
        )
        self._active_spans[span.span_id] = span
        self.metrics.increment("spans.started")
        return span

    def end_span(self, span: Span, status: str = "ok") -> None:
        """End a tracing span and record its duration."""
        span.end(status)
        self._active_spans.pop(span.span_id, None)
        self._completed_spans.append(span)
        if len(self._completed_spans) > 500:
            self._completed_spans = self._completed_spans[-500:]

        self.metrics.record(f"span.{span.operation}.duration_ms", span.duration_ms)
        self.metrics.increment("spans.completed")

        if status == "error":
            self.metrics.increment("spans.errors")

    def record_request(self, endpoint: str, duration_ms: float, status: str = "ok") -> None:
        """Record an API request."""
        self.metrics.increment(f"requests.{endpoint}.total")
        self.metrics.record(f"requests.{endpoint}.duration_ms", duration_ms)
        if status == "error":
            self.metrics.increment(f"requests.{endpoint}.errors")

    def record_agent_execution(
        self, agent_name: str, duration_ms: float, success: bool
    ) -> None:
        """Record agent task execution."""
        self.metrics.increment(f"agent.{agent_name}.total")
        self.metrics.record(f"agent.{agent_name}.duration_ms", duration_ms)
        if success:
            self.metrics.increment(f"agent.{agent_name}.success")
        else:
            self.metrics.increment(f"agent.{agent_name}.failure")

    def get_system_info(self) -> dict[str, Any]:
        """Get system resource information."""
        import platform
        try:
            import psutil
            mem = psutil.virtual_memory()
            cpu_pct = psutil.cpu_percent(interval=0.1)
            memory_info = {
                "total_gb": round(mem.total / 1e9, 1),
                "available_gb": round(mem.available / 1e9, 1),
                "used_pct": mem.percent,
            }
        except ImportError:
            memory_info = {"note": "Install psutil for memory monitoring"}
            cpu_pct = -1

        return {
            "platform": platform.system(),
            "processor": platform.processor()[:50],
            "python_version": platform.python_version(),
            "cpu_percent": cpu_pct,
            "memory": memory_info,
            "uptime_seconds": round(time.time() - self._start_time, 1),
        }

    def stats(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.snapshot(),
            "active_spans": len(self._active_spans),
            "completed_spans": len(self._completed_spans),
            "system": self.get_system_info(),
        }
