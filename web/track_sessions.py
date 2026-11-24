"""Per-browser tracking state for /api/detect.

Each awareness run in a browser sends its own random session id and a frame
number that only goes up. One ObjectTracker is kept per session id, so two
phones never share track IDs, and a new session id (awareness restarted)
starts again from ID 1. Idle sessions are dropped.
"""
import re
import threading
import time

from vision.tracking import ObjectTracker

SESSION_ID = re.compile(r"[A-Za-z0-9_-]{8,64}")
IDLE_SECONDS = 60
MAX_SESSIONS = 32

OK, STALE, OFF = "ok", "stale", "off"


class TrackSessions:
    def __init__(self, idle_seconds=IDLE_SECONDS, max_sessions=MAX_SESSIONS, clock=time.monotonic):
        self._sessions = {}  # session id -> {"tracker", "frame", "used"}
        self._lock = threading.Lock()
        self._idle_seconds, self._max_sessions, self._clock = idle_seconds, max_sessions, clock

    def __len__(self):
        return len(self._sessions)

    def track(self, session_id, frame, detections):
        """(track ids, status) for one frame of a session.

        A frame numbered at or below the newest one already processed is stale
        (late or repeated): it gets no IDs and does not disturb the tracks.
        """
        with self._lock:
            now = self._clock()
            for key in [k for k, s in self._sessions.items() if now - s["used"] > self._idle_seconds]:
                del self._sessions[key]
            session = self._sessions.get(session_id)
            if session is None:
                if len(self._sessions) >= self._max_sessions:
                    del self._sessions[min(self._sessions, key=lambda k: self._sessions[k]["used"])]
                session = self._sessions[session_id] = {"tracker": ObjectTracker(), "frame": None, "used": now}
            session["used"] = now
            if session["frame"] is not None and frame <= session["frame"]:
                return [None] * len(detections), STALE
            session["frame"] = frame
            return session["tracker"].update(detections, now), OK

    def reset(self, session_id):
        """Forget a session (awareness stopped). True if it existed."""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None
