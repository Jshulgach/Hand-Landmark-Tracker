"""Shared background processing worker for multi-camera backends."""

import threading
import time


class ProcessingThread(threading.Thread):
    """Run capture, detection, and triangulation independently of rendering."""

    def __init__(self, tracker):
        super().__init__(daemon=True, name="ProcessingThread")
        self.tracker = tracker
        self._running = True
        self._paused = threading.Event()
        self._paused.set()
        self._lock = threading.Lock()
        self._latest_result = None
        self._proc_fps = 0.0
        self._t_prev = time.perf_counter()

    def run(self):
        while self._running:
            self._paused.wait()
            if not self._running:
                break
            try:
                result = self.tracker.process_frame()
                with self._lock:
                    self._latest_result = result
                now = time.perf_counter()
                elapsed = now - self._t_prev
                self._proc_fps = 1.0 / elapsed if elapsed > 0 else 0.0
                self._t_prev = now
            except Exception as exc:
                print(f"[ProcessingThread] Error: {exc}")
                self._running = False
                break

    def get_latest(self):
        with self._lock:
            result = self._latest_result
            self._latest_result = None
            return result

    def get_processing_fps(self):
        return self._proc_fps

    def pause(self):
        self._paused.clear()

    def resume(self):
        self._paused.set()

    def stop(self):
        self._running = False
        self._paused.set()
        self.join(timeout=5)
