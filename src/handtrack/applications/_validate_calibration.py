"""Measure distance error and stationary repeatability using a rigid ChArUco target."""
import argparse
import csv
import hashlib
import importlib
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path


def add_validation_arguments(parser):
    parser.add_argument('--distance-mm', type=float, required=True,
                        help='independently measured distance BETWEEN the two selected corner centers')
    parser.add_argument('--corner-ids', type=int, nargs=2, default=[0, 19], metavar=('A', 'B'))
    parser.add_argument('--locations', nargs='+', default=['center', 'left', 'right', 'near', 'far', 'high', 'low'])
    parser.add_argument('--trials', type=int, default=2, help='visits per location; return to the same fixture pose')
    parser.add_argument('--samples', type=int, default=60, help='fresh endpoint pairs per stationary trial')
    parser.add_argument('--sample-hz', type=float, default=10.)
    parser.add_argument('--max-reprojection-px', type=float, default=1.5, help='sample filter, not a mocap accuracy specification')
    parser.add_argument('--min-ray-angle-deg', type=float, default=2.)
    parser.add_argument('--dictionary', help='ArUco dictionary name; omitted means auto-detect')
    parser.add_argument('--calibration', type=Path, help='defaults to the backend active calibration')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/calibration_validation'))
    parser.add_argument('--independent-target', action='store_true',
                        help='record that this physical target was NOT used to calibrate the cameras')


def validate_arguments(args):
    import math
    for name in ('distance_mm', 'sample_hz', 'max_reprojection_px', 'min_ray_angle_deg'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name.replace("_", "-")} must be finite and positive')
    if args.min_ray_angle_deg >= 90 or args.samples < 2 or args.trials < 1:
        raise ValueError('Need samples >= 2, trials >= 1, and a minimum ray angle below 90 degrees')
    if min(args.corner_ids) < 0 or args.corner_ids[0] == args.corner_ids[1]:
        raise ValueError('Choose two distinct non-negative corner IDs')
    if not args.locations or len(set(args.locations)) != len(args.locations):
        raise ValueError('Location names must be nonempty and unique')


def _write_report(directory, report, rows):
    from handtrack.processing._calibration_validation import summarize_locations
    report['locations'] = summarize_locations(report['trials'])
    (directory / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    fields = ['location', 'trial', 'time_s', 'cameras', 'distance_mm', 'signed_error_mm',
              'reprojection_rms_px', 'ray_angle_deg', 'ax_mm', 'ay_mm', 'az_mm', 'bx_mm', 'by_mm', 'bz_mm']
    with (directory / 'samples.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _snapshots(manager):
    import numpy as np
    values = [worker.get_snapshot() for worker in manager.workers]
    latest = max((stamp for frame, stamp in values if frame is not None and stamp is not None), default=None)
    frames, stamps = [], []
    for camera, (frame, stamp) in enumerate(values):
        valid = frame is not None and stamp is not None and latest-stamp <= .05
        width, height = manager.get_resolution(camera)
        frames.append(frame if valid else np.zeros((height, width, 3), np.uint8))
        stamps.append(stamp if valid else None)
    return frames, stamps


def run_validation(args, backend):
    import concurrent.futures
    import cv2
    import numpy as np
    from mavis_track.calibration import load_camera_calibration
    from handtrack.processing._calibration_validation import reconstruct_endpoints, summarize_trial
    from ._calibration_preview import create_calibration_preview

    validate_arguments(args)
    config = importlib.import_module(f'handtrack.cameras.{backend}.config')
    name = 'multi_mjpeg' if backend == 'optitrack' else 'multi_webcam'
    manager_class = importlib.import_module(f'handtrack.cameras.{backend}.{name}').CameraManager
    legacy = 'optitrack_cam_py' if backend == 'optitrack' else 'webcam'
    module = importlib.import_module(f'unity_hand_tracking.{legacy}.calibration')
    path = args.calibration or Path(config.CALIBRATION_FILE)
    # Verify argument and target configuration before acquiring SDK resources.
    if max(args.corner_ids) >= (config.CHARUCO_SQUARES_X-1)*(config.CHARUCO_SQUARES_Y-1):
        raise ValueError('Selected corner IDs are outside the configured ChArUco board')
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    manager = manager_class()
    preview = None
    report = None
    rows, pending_rows, pending = [], [], []
    try:
        if manager.num_cameras < 2:
            raise ValueError('Validation requires at least two calibrated cameras')
        # OptiTrack creates workers in start_all(); resolution and identity
        # checks must use those running workers after their first frames.
        manager.start_all(exposure=10)
        if hasattr(manager, 'wait_ready'):
            manager.wait_ready()
        calibration = load_camera_calibration(path, count=manager.num_cameras,
            image_sizes=np.array([manager.get_resolution(i) for i in range(manager.num_cameras)]),
            camera_ids=getattr(manager, 'camera_indices', list(range(manager.num_cameras))),
            diagnostics=manager.get_camera_diagnostics() if hasattr(manager, 'get_camera_diagnostics') else None)
        preview = create_calibration_preview(manager, 'Calibration physical validation', 10)
        controls = preview.controls
        controls.paused = True
        controls.instructions = (*controls.instructions[:2],
            'Space: start/restart stationary trial   Q: save and stop   Esc: cancel (completed trials retained)')
        directory = args.output_dir / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        directory.mkdir(parents=True, exist_ok=False)
        report = dict(schema_version=1, state='running', backend=backend,
            calibration_file=str(path.resolve()), calibration_sha256=fingerprint,
            camera_identity_verified=bool(calibration[4]),
            reference_distance_mm=args.distance_mm, corner_ids=args.corner_ids,
            independent_target_declared=bool(args.independent_target),
            requested_locations=args.locations, requested_trials=args.trials, samples_per_trial=args.samples,
            sample_hz=args.sample_hz, max_reprojection_px=args.max_reprojection_px,
            min_ray_angle_deg=args.min_ray_angle_deg, trials=[], rejected_samples={},
            rejected_sample_count_definition='Rejected detection attempts, not independent temporal samples',
            metric_definitions=dict(endpoint_jitter_rms_mm='RMS 3D distance from each endpoint trial mean',
                distance_rmse_mm='RMS reconstructed distance error against the entered physical measurement',
                return_displacement='Displacement of trial means from first visit; includes repositioning error'),
            notes=['No calibration parameters are modified.',
                   'Coordinates are in the calibration camera-0 frame; no floor alignment is assumed.',
                   'Reprojection and ray-angle filters are project settings, not accuracy specifications.',
                   'Same-board results are consistency checks, not independent accuracy certification.',
                   'Jitter includes target motion; use a rigid stationary support.',
                   'Return displacement includes positioning error; use the same physical fixture and orientation.',
                   'Host receipt timestamps reject stale/reused frames but do not establish exposure synchronization.'])
        _write_report(directory, report, rows)
        print(f'Validation report: {directory.resolve()}')
        with concurrent.futures.ThreadPoolExecutor(max_workers=manager.num_cameras) as executor:
            dictionary = args.dictionary
            while dictionary is None and not controls.quit_requested:
                frames, _ = _snapshots(manager)
                for frame in frames:
                    future = executor.submit(module.auto_detect_dictionary, frame)
                    while not future.done():
                        preview.show_frames(frames, 'Show the validation board to a camera. Detecting dictionary...')
                        if controls.quit_requested:
                            break
                        time.sleep(.01)
                    if controls.quit_requested:
                        break
                    name, corners = future.result()
                    if name is not None and corners >= 4:
                        dictionary = name
                        break
                preview.show_frames(frames, 'Show the validation board to a camera. Detecting dictionary...')
            if controls.quit_requested:
                report['state'] = 'cancelled' if controls.cancelled else 'stopped'
                return 1
            _, first = module.build_charuco_detector(dict_name=dictionary)
            detectors = [first] + [module.build_charuco_detector(dict_name=dictionary)[1]
                                   for _ in range(manager.num_cameras-1)]
            report['dictionary'] = dictionary
            schedule = [(location, trial) for trial in range(1, args.trials+1) for location in args.locations]
            position = 0
            last_sample = float('-inf')
            used_stamps = {}
            rejected = Counter()
            started = time.monotonic()
            last_summary = ''
            while position < len(schedule) and not controls.quit_requested:
                location, trial = schedule[position]
                batch_started = time.monotonic()
                frames, stamps = _snapshots(manager)
                futures = [executor.submit(module.detect_charuco, (frame, detector))
                           for frame, detector in zip(frames, detectors)]
                while not all(future.done() for future in futures):
                    preview.show_frames(frames, f'{location} — trial {trial}/{args.trials}; {len(pending)}/{args.samples} samples')
                    if controls.quit_requested:
                        break
                    time.sleep(.005)
                if controls.quit_requested:
                    break
                detections = [future.result() for future in futures]
                display = [module.annotate_frame(frame, c, found, data)
                           for c, (frame, (found, data)) in enumerate(zip(frames, detections))]
                for image, (found, data) in zip(display, detections):
                    if found:
                        for key, point in zip(data[1].ravel(), data[0].reshape(-1, 2)):
                            if int(key) in args.corner_ids:
                                xy = tuple(np.round(point).astype(int))
                                cv2.circle(image, xy, 9, (0, 255, 255), 2)
                                cv2.putText(image, f'ID {key}', (xy[0]+10, xy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
                valid = False
                reason = ''
                try:
                    points, cameras, reprojection, angle = reconstruct_endpoints(detections, calibration,
                        args.corner_ids, max_reprojection_px=args.max_reprojection_px,
                        min_ray_angle_deg=args.min_ray_angle_deg)
                    valid = True
                except ValueError as error:
                    reason = str(error)
                now = time.monotonic()
                ready = valid and controls.capture_ready and batch_started >= controls.last_change
                if controls.paused or batch_started < controls.last_change or not controls.capture_ready:
                    pending.clear()
                    pending_rows.clear()
                if not pending:
                    controls.countdown.update(now, ready)
                controls.countdown.waiting_message = reason or 'Press Space when target is stationary at the requested location'
                if ready and (pending or controls.countdown.due(batch_started)) and now-last_sample >= 1/args.sample_hz:
                    if all(stamps[c] is not None and stamps[c] != used_stamps.get(c) for c in cameras):
                        last_sample = now
                        used_stamps.update({c:stamps[c] for c in cameras})
                        pending.append(points)
                        distance = float(np.linalg.norm(points[1]-points[0])*1000.)
                        row = dict(location=location, trial=trial, time_s=now-started,
                            cameras=';'.join(map(str,cameras)), distance_mm=distance,
                            signed_error_mm=distance-args.distance_mm,
                            reprojection_rms_px=reprojection, ray_angle_deg=angle)
                        row.update(zip(('ax_mm','ay_mm','az_mm','bx_mm','by_mm','bz_mm'), (points*1000.).ravel()))
                        pending_rows.append(row)
                    else:
                        rejected['reused_or_missing_frame'] += 1
                elif not valid and not controls.paused:
                    rejected[reason] += 1
                message = (f'{location} — trial {trial}/{args.trials}; {len(pending)}/{args.samples} samples. '
                           f'Hold IDs {args.corner_ids[0]} and {args.corner_ids[1]} still in 2+ cameras.')
                if pending:
                    stats = summarize_trial(pending, args.distance_mm) if len(pending)>1 else None
                    message += (f' Distance {pending_rows[-1]["distance_mm"]:.2f} mm.' +
                                (f' Jitter A/B {stats["endpoint_jitter_rms_mm"][0]:.2f}/{stats["endpoint_jitter_rms_mm"][1]:.2f} mm.' if stats else ''))
                if reason:
                    message += ' ' + reason
                if len(pending) >= args.samples:
                    summary = dict(location=location, trial=trial, **summarize_trial(pending, args.distance_mm))
                    report['trials'].append(summary)
                    rows.extend(pending_rows)
                    pending.clear()
                    pending_rows.clear()
                    report['rejected_samples'] = dict(rejected)
                    _write_report(directory, report, rows)
                    last_summary = (f'{location} trial {trial}: mean error {summary["mean_signed_error_mm"]:.2f} mm; '
                                    f'RMSE {summary["distance_rmse_mm"]:.2f} mm.')
                    print(last_summary)
                    position += 1
                    controls.paused = True
                    controls.countdown.reset()
                    if position < len(schedule):
                        following, next_trial = schedule[position]
                        message = f'Next: {following}, trial {next_trial}. Place target on support; press Space. {last_summary}'
                controls.capture_message = (f'Recording stationary samples: {len(pending)}/{args.samples} — keep target still'
                                            if pending else '')
                preview.show_frames(display, message)
            report['state'] = ('completed' if position == len(schedule) else
                               'cancelled' if controls.cancelled else 'stopped')
            report['rejected_samples'] = dict(rejected)
            return 0 if report['state']=='completed' else 1
    except Exception as error:
        if report is not None:
            report['state'] = 'error'
            report['error'] = str(error)
        raise
    finally:
        try:
            if report is not None:
                report['partial_samples'] = len(pending_rows)
                _write_report(directory, report, [*rows, *pending_rows])
                for location, values in report['locations'].items():
                    shift = values['max_return_endpoint_rms_shift_mm']
                    returned = f'{shift:.3f} mm' if shift is not None else 'needs another visit'
                    print(f'{location}: mean distance {values["mean_trial_distance_mm"]:.3f} mm; '
                          f'between-trial distance range {values["trial_distance_range_mm"]:.3f} mm; '
                          f'return endpoint displacement {returned}')
                print(f'Report and raw samples saved: {directory.resolve()}')
        finally:
            try:
                if preview is not None:
                    preview.shutdown()
            finally:
                manager.stop_all()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('optitrack','webcam'), required=True)
    add_validation_arguments(parser)
    args = parser.parse_args(argv)
    try:
        validate_arguments(args)
    except ValueError as error:
        parser.error(str(error))
    return run_validation(args, args.backend)


if __name__ == '__main__':
    raise SystemExit(main())
