"""Cleanup helpers shared by compatibility trackers."""

from functools import wraps


def close_after(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        finally:
            self.close()
    return wrapped


def cleanup_failed_start(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception:
            self.cleanup()
            raise
    return wrapped
