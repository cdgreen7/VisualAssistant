"""Lightweight per-request stage timing for the research baseline.

Each request gets a RequestTiming; code wraps a stage in `with stage("yolo"):`
and the totals leave the server in a standard `Server-Timing` response header:

    Server-Timing: decode;dur=3.1, yolo;dur=24.6, tracking;dur=0.1, json;dur=0.2, total;dur=29.4

Response bodies are not changed. All clocks are time.perf_counter (monotonic),
all values are milliseconds, and nothing is logged per request.

These are SERVER-SIDE times only. `total` runs from when Flask starts handling
the request (the upload has already been received) to when the response object
is built; it excludes network transfer, TLS, and anything in the browser.
The browser measures those separately (web/static/timing.js).
"""
import contextvars
import json
import os
import re
import threading
import time
from contextlib import contextmanager

from flask import request
from flask.json.provider import DefaultJSONProvider

_current = contextvars.ContextVar("request_timing", default=None)
STAGE_NAME = re.compile(r"[a-z][a-z0-9_]{0,31}")


class RequestTiming:
    def __init__(self):
        self.started = time.perf_counter()
        self.stages = {}

    def add(self, name, milliseconds):
        self.stages[name] = self.stages.get(name, 0.0) + milliseconds

    def header(self):
        total = (time.perf_counter() - self.started) * 1000
        parts = [f"{name};dur={value:.1f}" for name, value in self.stages.items()]
        return ", ".join(parts + [f"total;dur={total:.1f}"])


def record(name, milliseconds):
    """Add an already-measured duration to the current request (no-op outside one)."""
    timing = _current.get()
    if timing is not None:
        timing.add(name, float(milliseconds))


@contextmanager
def stage(name):
    started = time.perf_counter()
    try:
        yield
    finally:
        record(name, (time.perf_counter() - started) * 1000)


class Timed:
    """Proxy that times one method of a component and passes everything else through."""

    def __init__(self, component, method, name):
        self._component, self._method, self._name = component, method, name

    def __getattr__(self, attribute):
        target = getattr(self._component, attribute)
        if attribute != self._method:
            return target

        def call(*args, **kwargs):
            with stage(self._name):
                return target(*args, **kwargs)
        return call


class _TimedJSON(DefaultJSONProvider):
    def dumps(self, obj, **kwargs):
        with stage("json"):
            return super().dumps(obj, **kwargs)


# ---------- optional collection of browser-side timings (research runs only)

MAX_LOG_BYTES = 20 * 1024 * 1024
MAX_BATCH_BYTES = 64 * 1024
MAX_RECORDS = 100


def client_label(user_agent):
    """Coarse client family from a User-Agent string, e.g. "iphone-safari"."""
    ua = user_agent or ""
    if "iPhone" in ua or "iPad" in ua or "iPod" in ua:
        device = "ipad" if "iPad" in ua else "iphone"
        browser = "chrome" if "CriOS" in ua else "firefox" if "FxiOS" in ua else "safari"
    elif "Android" in ua:
        device = "android"
        browser = "firefox" if "Firefox" in ua else "samsung" if "SamsungBrowser" in ua else "chrome" if "Chrome" in ua else "other"
    else:
        device = "desktop"
        browser = ("edge" if "Edg/" in ua else "chrome" if "Chrome" in ua else "firefox" if "Firefox" in ua
                   else "safari" if "Safari" in ua else "other")
    return f"{device}-{browser}"


def _clean(value, depth=0):
    """Keep only small, flat, JSON-safe timing data."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value if abs(value) < 1e12 else None
    if isinstance(value, str):
        return value[:120]
    if isinstance(value, dict) and depth < 2:
        return {str(k)[:40]: _clean(v, depth + 1) for k, v in list(value.items())[:40]}
    return None


class TimingLog:
    """Appends browser timing batches to a JSON-lines file. No images, no IP addresses."""

    def __init__(self, path):
        self.path, self._lock = path, threading.Lock()

    def append(self, payload, user_agent):
        records = payload.get("records") if isinstance(payload, dict) else None
        if not isinstance(records, list) or not 0 < len(records) <= MAX_RECORDS:
            return 0
        meta = _clean(payload.get("meta") if isinstance(payload.get("meta"), dict) else {})
        meta["client_from_server"] = client_label(user_agent)
        lines = [json.dumps({"received_unix": round(time.time(), 3), "meta": meta, "record": _clean(r)})
                 for r in records if isinstance(r, dict)]
        with self._lock:
            if os.path.exists(self.path) and os.path.getsize(self.path) > MAX_LOG_BYTES:
                return 0
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "a") as output:
                output.write("\n".join(lines) + "\n")
        return len(lines)


def install(app, timing_log_path=None):
    """Add Server-Timing to every response; optionally accept browser timings."""
    app.json = _TimedJSON(app)
    log = TimingLog(timing_log_path) if timing_log_path else None

    @app.before_request
    def start_timing():
        request.environ["va.timing_token"] = _current.set(RequestTiming())

    @app.after_request
    def add_server_timing(response):
        timing = _current.get()
        if timing is not None:
            response.headers["Server-Timing"] = timing.header()
        return response

    @app.teardown_request
    def stop_timing(_):
        token = request.environ.pop("va.timing_token", None)
        if token is not None:
            _current.reset(token)

    @app.post("/api/research/timings")
    def collect_timings():
        if log is None:
            return {"error": {"code": "timing_log_off", "message": "Start the server with --timing-log to collect."}}, 404
        if request.content_length is None or request.content_length > MAX_BATCH_BYTES:
            return {"error": {"code": "timings_too_large", "message": "Timing batch is too large."}}, 413
        return {"stored": log.append(request.get_json(silent=True), request.headers.get("User-Agent"))}
