"""
Test script to verify LSL streams from mocap_handtrack_gui.py.

Usage:
  1. Start the GUI:   python mocap_handtrack_gui.py
  2. Click "Start Tracking" in the GUI
  3. Run this script:  python test_lsl_streaming.py

Modes:
  python test_lsl_streaming.py             # Monitor GUI streams
  python test_lsl_streaming.py --selftest  # Self-test (create + receive)
"""

import argparse
import sys
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np

try:
    from pylsl import StreamInfo, StreamInlet, StreamOutlet, resolve_byprop
except ImportError as e:
    print(f"ERROR: Import failed - {e}")
    print("Make sure pylsl is installed: pip install pylsl")
    sys.exit(1)


def format_timestamp(ts):
    """Convert timestamp to readable format."""
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3]


def monitor_stream(stream_name, timeout=10):
    """Find an LSL stream by name. Returns StreamInlet or None."""
    print(f"\n{'=' * 60}")
    print(f"Looking for stream: {stream_name}  (timeout={timeout}s)")
    print(f"{'=' * 60}")

    try:
        streams = resolve_byprop("name", stream_name, timeout=timeout)
        if not streams:
            print(f"  [FAIL] Stream '{stream_name}' not found!")
            return None

        inlet = StreamInlet(streams[0])
        info = inlet.info()

        print(f"  [OK]  Found stream: {stream_name}")
        print(f"        Type: {info.type()}")
        print(f"        Channels: {info.channel_count()}")
        print(f"        Sample rate: {info.nominal_srate()} Hz")
        print(f"        Format: {info.channel_format()}")

        return inlet

    except Exception as e:
        print(f"  [FAIL] Error connecting to stream: {e}")
        return None


def test_stream_data(inlet, stream_name, duration=15, sample_interval=1.0):
    """Pull data from an inlet for `duration` seconds and report stats."""
    print(f"\n  Monitoring '{stream_name}' for {duration}s ...")
    print(f"  {'Elapsed':<10} {'Packets':<10} {'Rate (Hz)':<12} {'Latency (ms)':<15}")
    print("  " + "-" * 55)

    start_time = time.time()
    last_report = start_time
    packet_count = 0
    timestamps = deque(maxlen=200)

    while time.time() - start_time < duration:
        sample, ts = inlet.pull_sample(timeout=0.1)
        if sample is not None:
            packet_count += 1
            timestamps.append(ts)

            elapsed = time.time() - last_report
            if elapsed >= sample_interval:
                if len(timestamps) > 1:
                    span = timestamps[-1] - timestamps[0]
                    rate = (len(timestamps) - 1) / span if span > 0 else 0
                else:
                    rate = 0
                latency_ms = (time.time() - ts) * 1000
                elapsed_str = f"{time.time() - start_time:.1f}s"
                print(
                    f"  {elapsed_str:<10} {packet_count:<10} {rate:<12.2f} {latency_ms:<15.2f}"
                )
                last_report = time.time()

    print("  " + "-" * 55)
    elapsed = time.time() - start_time
    avg = packet_count / elapsed if elapsed > 0 else 0
    print(f"  Total: {packet_count} packets in {elapsed:.1f}s  ({avg:.1f} Hz)")
    status = "STREAMING" if packet_count > 0 else "NO DATA"
    print(f"  Result: {status}")
    return packet_count > 0


# ------------------------------------------------------------------ #
# Self-test: create a dummy stream, push data, receive it
# ------------------------------------------------------------------ #
def run_selftest():
    """Create a dummy LSL stream and verify we can receive from it."""
    print("\n" + "=" * 70)
    print("LSL SELF-TEST")
    print("=" * 70)
    print("Creating a test stream and verifying round-trip ...\n")

    name = "LSL_SelfTest_Stream"
    n_ch = 3
    info = StreamInfo(name, "Test", n_ch, 100, "float32", "selftest_001")
    outlet = StreamOutlet(info)
    print(f"  Created outlet: {name}  ({n_ch} channels @ 100 Hz)")

    # Give the outlet a moment to advertise
    time.sleep(1.0)

    # Push samples in a background thread
    stop_flag = threading.Event()
    push_count = [0]

    def _push():
        while not stop_flag.is_set():
            outlet.push_sample([np.random.randn() for _ in range(n_ch)])
            push_count[0] += 1
            time.sleep(0.01)  # ~100 Hz

    t = threading.Thread(target=_push, daemon=True)
    t.start()

    # Resolve and pull
    streams = resolve_byprop("name", name, timeout=5)
    if not streams:
        print("  [FAIL] Could not resolve self-test stream!")
        stop_flag.set()
        return False

    inlet = StreamInlet(streams[0])
    received = 0
    deadline = time.time() + 3.0
    while time.time() < deadline:
        s, ts = inlet.pull_sample(timeout=0.1)
        if s is not None:
            received += 1

    stop_flag.set()
    t.join(timeout=2)

    print(f"  Pushed:    {push_count[0]} samples")
    print(f"  Received:  {received} samples")
    ok = received > 0
    print(f"  Result:    {'PASS' if ok else 'FAIL'}")
    print("=" * 70 + "\n")
    return ok


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="Test LSL streaming")
    parser.add_argument(
        "--selftest", action="store_true", help="Run a self-contained LSL loopback test"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=15,
        help="How long to monitor each stream (seconds)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("LSL STREAMING TEST - Hand Tracking Monitor")
    print("=" * 70)

    # Always run self-test first if requested
    if args.selftest:
        ok = run_selftest()
        if not ok:
            print("Self-test failed. LSL may not be working on this machine.")
            sys.exit(1)
        print("Self-test passed! LSL is working correctly.\n")
        return

    # Monitor GUI streams
    stream_names = [
        "StereoHandTracker_Landmarks",
        "StereoHandTracker_Angles",
    ]

    print("\nSearching for hand-tracking LSL streams ...")
    print("(Make sure mocap_handtrack_gui.py is running and tracking is started)\n")

    results = {}
    for sname in stream_names:
        inlet = monitor_stream(sname, timeout=5)
        if inlet:
            has_data = test_stream_data(inlet, sname, duration=args.duration)
            results[sname] = has_data
        else:
            results[sname] = False

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for sname, ok in results.items():
        short = sname.replace("StereoHandTracker_", "")
        status = "[OK]  ACTIVE + DATA" if ok else "[FAIL] INACTIVE / NO DATA"
        print(f"  {short:<20} {status}")

    any_found = any(results.values())
    print()
    if all(results.values()):
        print("  All streams active and streaming data!")
    elif any_found:
        print("  Some streams missing data. Is a hand visible to the cameras?")
    else:
        print("  No streams found. Checklist:")
        print("    1. Is mocap_handtrack_gui.py running?")
        print("    2. Did you click 'Start Tracking'?")
        print("    3. Try: python test_lsl_streaming.py --selftest")
    print("=" * 70 + "\n")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
