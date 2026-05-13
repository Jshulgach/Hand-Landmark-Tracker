"""Compatibility shim for webcam demos.

This module intentionally re-exports the shared broadcasters from
`handtrack.io.broadcast` so existing webcam scripts can keep importing
`broadcast` locally while we centralize implementation in the core package.
"""

import sys
from pathlib import Path

_src_root = Path(__file__).resolve().parents[2]
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

from handtrack.io.broadcast import DataBroadcaster, LSLBroadcaster, UDPBroadcaster

__all__ = ["DataBroadcaster", "UDPBroadcaster", "LSLBroadcaster"]