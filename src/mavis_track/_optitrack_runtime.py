"""Process-wide SDK lifetime and exclusive ownership of OptiTrack capture."""
from __future__ import annotations

import atexit
import sys
import threading
from typing import Any


class SDKRuntime:
    """The native singleton is initialized once and shut down only at app exit."""

    def __init__(self, sdk: Any):
        self.sdk = sdk
        self.lock = threading.RLock()
        self.owner: Any = None
        self.initialized = False
        self.closing = False
        self.closed = False

    def acquire(self, owner: Any) -> bool:
        with self.lock:
            if self.closed or self.closing:
                raise RuntimeError("OptiTrack SDK has shut down; restart the application")
            if self.owner is not None:
                raise RuntimeError("OptiTrack cameras are still owned by another capture session; stop it first")
            fresh = not self.initialized
            if fresh:
                self.sdk.initialize_sdk()
                self.initialized = True
            self.owner = owner
            return fresh

    def release(self, owner: Any) -> None:
        with self.lock:
            if self.owner is owner:
                self.owner = None

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closing = True
            owner = self.owner
        # Never hold the runtime lock while stopping/joining workers:
        # stop_all releases its lease while holding the manager's stop lock.
        if owner is not None:
            owner.stop_all()
        with self.lock:
            if self.owner is not None:
                raise RuntimeError("Camera capture is still active; SDK shutdown deferred")
            self.closed = True
            if self.initialized:
                print("Shutting down OptiTrack SDK...")
                self.sdk.shutdown_sdk()


_registry_lock = threading.RLock()
_runtimes: dict[int, SDKRuntime] = {}


def capture_runtime(sdk: Any) -> SDKRuntime:
    with _registry_lock:
        key = id(sdk)
        if key not in _runtimes:
            _runtimes[key] = SDKRuntime(sdk)
        return _runtimes[key]


def shutdown_capture_sdks() -> None:
    """Final application cleanup; repeated calls are harmless."""
    with _registry_lock:
        runtimes = list(_runtimes.values())
    for runtime in runtimes:
        runtime.close()


def _shutdown_at_exit() -> None:
    # A best-effort fallback for library callers that do not run the GUI main().
    with _registry_lock:
        runtimes = list(_runtimes.values())
    for runtime in runtimes:
        try:
            runtime.close()
        except Exception as exc:  # noqa: BLE001 - Interpreter-exit native cleanup.
            print(f"OptiTrack final cleanup failed: {exc}", file=sys.stderr)


atexit.register(_shutdown_at_exit)
