"""Offline sync report and replay entry points; no live GUI dependencies."""
from pathlib import Path


def add_parsers(subparsers):
    report = subparsers.add_parser('sync-report', help='audit camera videos and align to Vicon C3D using explicit anchors')
    report.add_argument('session')
    report.add_argument('--vicon', help='Nexus C3D export')
    report.add_argument('--anchors', help='CSV: camera_index,sdk_frame_id,vicon_frame,role (fit/check)')
    report.add_argument('--expected-rate', type=float, default=50.)
    report.add_argument('--tolerance-ms', type=float, default=5.)
    report.add_argument('--exposure-channel', help='C3D analog channel containing OptiHub exposure pulses')
    report.add_argument('--threshold', type=float, default=2.)
    report.add_argument('--output')
    replay = subparsers.add_parser('sync-replay', help='review aligned camera videos alongside Vicon markers')
    replay.add_argument('session')
    replay.add_argument('--vicon', required=True, help='same C3D used by sync-report')
    replay.add_argument('--mapping', help='alignment/frame_mapping.csv by default')
    replay.add_argument('--projection', help='optional calibrated Vicon-to-camera transforms JSON')
    replay.add_argument('--no-display', action='store_true', help='validate inputs and decode mapped video frames without a window')


def run(args):
    try:
        if args.command == 'sync-report':
            from mavis_track._sync_analysis import analyze_session
            report = analyze_session(args.session, vicon_path=args.vicon, anchors_path=args.anchors,
                expected_rate=args.expected_rate, tolerance_ms=args.tolerance_ms,
                exposure_channel=args.exposure_channel, threshold=args.threshold, output=args.output)
            for camera, stats in report['cameras'].items():
                print(f"Camera {camera}: {stats['status']}")
            print('Report:', Path(args.output) if args.output else Path(args.session)/'alignment')
            return 0
        from ._sync_replay import ReplayData, show_replay
        with ReplayData(args.session, args.vicon, mapping=args.mapping, projection=args.projection) as data:
            if args.no_display:
                decoded = 0
                for camera, matches in data.mapping.items():
                    for frame in sorted(matches):
                        if data.image(camera, frame) is None:
                            raise ValueError(f'Cannot decode camera {camera} at Vicon frame {frame}')
                        decoded += 1
                print(f'Checked C3D/recording identity and decoded {decoded} mapped images. See report for timing confidence.')
                return 0
            return show_replay(data)
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
        print(f'Sync review failed: {exc}')
        return 1
