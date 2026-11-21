import platform
import queue
import subprocess
import threading


class Speaker:
    """Non-blocking text-to-speech. say() returns immediately; a background thread speaks."""

    def __init__(self, rate=200, max_pending=2):
        self.rate = rate
        self._queue = queue.Queue(maxsize=max_pending)
        self._speaking = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def say(self, text):
        # Drop the oldest pending message if full, so speech never lags far behind the camera
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        self._queue.put(text)

    def is_busy(self):
        return self._speaking or not self._queue.empty()

    def stop(self):
        self._queue.put(None)
        self._thread.join(timeout=2)

    def _run(self):
        speak = self._make_engine()
        while True:
            text = self._queue.get()
            if text is None:
                break
            self._speaking = True
            speak(text)
            self._speaking = False

    def _make_engine(self):
        if platform.system() == "Darwin":
            # macOS built-in TTS is more reliable from a thread than pyttsx3
            return lambda text: subprocess.run(["say", "-r", str(self.rate), text])

        # otherwise, use pyttsx3 for cross-platform TTS.
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)

        def speak(text):
            engine.say(text)
            engine.runAndWait()

        return speak


if __name__ == "__main__":
    import time

    s = Speaker()
    s.say("Visual assistant ready")
    s.say("person left")
    while s.is_busy():
        time.sleep(0.1)
    s.stop()