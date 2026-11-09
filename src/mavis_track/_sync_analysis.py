"""Offline Vicon alignment with explicit anchors, loss accounting and check events.

No alignment is inferred from matching file names, host clocks or frame zero.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from ._frame_diagnostics import FrameDiagnostics


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_vicon(path):
    try:
        import ezc3d
    except ImportError as exc:
        raise RuntimeError("Install C3D support: pip install 'mavis-track[sync]'") from exc
    path = Path(path).resolve()
    data = ezc3d.c3d(str(path))
    points = np.array(data['data']['points'][:3], dtype=float).transpose(2, 1, 0)
    residuals = np.asarray(data['data']['meta_points']['residuals'])[0].T
    points[residuals < 0] = np.nan
    rate = float(data['parameters']['POINT']['RATE']['value'][0])
    if not np.isfinite(rate) or rate <= 0 or not len(points):
        raise ValueError("C3D must contain point frames at a positive rate")
    units = str(data['parameters']['POINT']['UNITS']['value'][0]).lower()
    scale = {'m': 1., 'mm': .001, 'cm': .01}.get(units)
    if scale is None:
        raise ValueError(f"Unsupported C3D coordinate units: {units}")
    first = int(data['header']['points']['first_frame'])
    analog = data['parameters'].get('ANALOG', {})
    return {'path': str(path), 'sha256': file_hash(path),
                'rate': rate, 'first_frame': first, 'last_frame': first + len(points)-1,
                'markers': list(data['parameters']['POINT']['LABELS']['value']), 'points': points * scale,
                'analog_labels': list(analog.get('LABELS', {}).get('value', [])),
                'analog_rate': float(analog.get('RATE', {}).get('value', [0])[0]),
                'analog': np.asarray(data['data']['analogs'])[0]}


def load_session(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 'mavis-camera-video-v1':
        raise ValueError("Not a camera video session")
    grouped = {}
    with (directory / 'frames.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            index, video_index = int(row['camera_index']), int(row['video_frame_index'])
            selected = grouped.setdefault(index, [])
            if video_index != len(selected):
                raise ValueError(f"Non-contiguous video indices in camera {index}")
            selected.append(row)
    if not grouped:
        raise ValueError("Session has no recorded video frames")
    return manifest, grouped


def video_path(directory, manifest, index):
    root = Path(directory).resolve()
    relative = manifest.get('video_files', {}).get(str(index), f'camera_{index}.avi')
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError("Video file must remain inside the session directory")
    return result


def exposure_edges(vicon, channel, threshold):
    if not np.isfinite(threshold):
        raise ValueError("Exposure threshold must be finite")
    if channel not in vicon['analog_labels']:
        raise ValueError(f"C3D has no analog channel {channel!r}")
    if vicon['analog_rate'] <= 0:
        raise ValueError("C3D has no positive analog sampling rate")
    signal = vicon['analog'][vicon['analog_labels'].index(channel)]
    high = np.isfinite(signal) & (signal >= threshold)
    # A high first sample has no observed rising edge; do not invent one.
    samples = np.flatnonzero(high[1:] & ~high[:-1]) + 1
    falling = np.flatnonzero(~high[1:] & high[:-1]) + 1
    result = []
    for i, sample in enumerate(samples):
        endings = falling[falling > sample]
        end = int(endings[0]) if len(endings) else None
        result.append({'edge_index': i, 'analog_sample': int(sample),
            'vicon_frame': vicon['first_frame'] + sample/vicon['analog_rate'] * vicon['rate'],
            'vicon_time_seconds': sample/vicon['analog_rate'],
            'pulse_width_seconds': (end-sample)/vicon['analog_rate'] if end is not None else None,
            'midpoint_vicon_frame': vicon['first_frame']+(end+sample)/2/vicon['analog_rate']*vicon['rate'] if end is not None else None,
            'truncated': end is None})
    return result


def read_anchors(path):
    if path is None:
        return []
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    result = []
    for row in rows:
        role = (row.get('role') or 'fit').strip().lower()
        if role not in ('fit', 'check'):
            raise ValueError("Anchor role must be fit or check")
        frame = float(row['vicon_frame'])
        if not math.isfinite(frame):
            raise ValueError("Anchor Vicon frames must be finite")
        result.append({'camera_index': int(row['camera_index']), 'sdk_frame_id': int(row['sdk_frame_id']),
                           'vicon_frame': frame, 'role': role, 'source': row.get('source', 'user-supplied')})
    return result


def _camera_axis(rows, rate):
    ids = [r.get('sdk_frame_id') for r in rows]
    if any(i is None for i in ids) or any(b <= a for a, b in itertools.pairwise(ids)):
        return None, None, 'SDK IDs unavailable, repeated or reset; explicit segment alignment required'
    hardware = [r.get('hardware_timestamp_ticks') for r in rows]
    frequencies = [r.get('hardware_timestamp_frequency') for r in rows]
    if (all(r.get('hardware_timestamp_valid') for r in rows)
            and all(t is not None for t in hardware) and all(f is not None and f > 0 for f in frequencies)
            and len(set(frequencies)) == 1 and all(b > a for a, b in itertools.pairwise(hardware))):
        return np.asarray([t-hardware[0] for t in hardware], dtype=float)/frequencies[0], 'hardware_clock', None
    return (np.asarray(ids, dtype=float)-ids[0])/rate, 'sdk_id_at_nominal_rate', None


def analyze_session(directory, *, vicon_path=None, anchors_path=None, expected_rate=50.,
                    tolerance_ms=5., exposure_channel=None, threshold=2., output=None):
    if not np.isfinite(expected_rate) or expected_rate <= 0 or not np.isfinite(tolerance_ms) or tolerance_ms <= 0:
        raise ValueError("Expected rate and tolerance must be finite and positive")
    directory = Path(directory).resolve()
    manifest, cameras = load_session(directory)
    output = Path(output).resolve() if output else directory / 'alignment'
    output.mkdir(parents=True, exist_ok=True)
    vicon = load_vicon(vicon_path) if vicon_path else None
    anchors = read_anchors(anchors_path)
    if anchors and vicon is None:
        raise ValueError("Alignment anchors require a Vicon C3D file")
    unknown = {a['camera_index'] for a in anchors} - cameras.keys()
    if unknown:
        raise ValueError(f"Anchors refer to unrecorded cameras: {sorted(unknown)}")
    diagnostics = FrameDiagnostics()
    report = {'schema': 'mavis-vicon-alignment-v1', 'trial_name': manifest.get('trial_name'),
                  'session': str(directory), 'complete': manifest.get('complete', False),
                  'frames_sha256': file_hash(directory/'frames.jsonl'),
                  'manifest_sha256': file_hash(directory/'manifest.json'), 'calibration': manifest.get('calibration'),
                  'expected_camera_rate': expected_rate, 'tolerance_ms': tolerance_ms,
                  'queue_drops': manifest.get('queue_items_dropped', 0), 'cameras': {},
                  'anchors': anchors, 'anchors_sha256': hashlib.sha256(Path(anchors_path).read_bytes()).hexdigest() if anchors_path else None,
                  'warnings': ['File-name timestamps and host receipt times are not exposure synchronization.']}
    if vicon:
        report['vicon'] = {key: vicon[key] for key in ('path', 'sha256', 'rate', 'first_frame', 'last_frame')}
    mapping = []
    for camera, rows in sorted(cameras.items()):
        for row in rows:
            diagnostics.observe(camera, dict(row))
        stamps = np.asarray([r['host_received_monotonic'] for r in rows], float)
        intervals = np.diff(stamps)
        stats = {'status': 'unmapped', 'frames': len(rows),
                     'max_delivery_interval_ms': float(max(intervals, default=0)*1000),
                     'delivery_jitter_p95_ms': float(np.percentile(abs(intervals-1/expected_rate), 95)*1000) if len(intervals) else None}
        report['cameras'][str(camera)] = stats
        axis, basis, issue = _camera_axis(rows, expected_rate)
        selected = [a for a in anchors if a['camera_index'] == camera]
        by_id = {r.get('sdk_frame_id'): i for i, r in enumerate(rows)}
        fit = [a for a in selected if a['role'] == 'fit']
        for anchor in selected:
            if anchor['sdk_frame_id'] not in by_id:
                raise ValueError(f"Camera {camera}: anchor SDK ID was not recorded")
            if not vicon['first_frame'] <= anchor['vicon_frame'] <= vicon['last_frame']:
                raise ValueError("Anchor lies outside the exported C3D range")
        if issue or not fit or vicon is None:
            stats['reason'] = issue or 'Supply explicit fit anchors and a Vicon C3D file'
            for row in rows:
                mapping.append({'camera_index': camera, 'video_frame_index': row['video_frame_index'],
                                    'sdk_frame_id': row.get('sdk_frame_id'), 'vicon_frame': '', 'vicon_time_seconds': '',
                                    'estimated_frame': '', 'confidence': 'unmapped'})
            continue
        anchor_ids = [a['sdk_frame_id'] for a in selected]
        if len(set(anchor_ids)) != len(anchor_ids):
            raise ValueError("An SDK frame cannot be both a fit and independent check anchor or appear twice")
        x = np.asarray([axis[by_id[a['sdk_frame_id']]] for a in fit])
        y = np.asarray([(a['vicon_frame']-vicon['first_frame'])/vicon['rate'] for a in fit])
        slope, intercept = (1., float(y[0]-x[0])) if len(fit) == 1 else np.polyfit(x, y, 1)
        if not np.isfinite(slope) or slope <= 0:
            raise ValueError("Anchor times must increase with camera time")
        predicted = slope*axis + intercept
        checks = [a for a in selected if a['role'] == 'check']
        errors = [(predicted[by_id[a['sdk_frame_id']]] -
                   (a['vicon_frame']-vicon['first_frame'])/vicon['rate'])*1000 for a in checks]
        check_span = (max(axis[by_id[a['sdk_frame_id']]] for a in checks)-
                      min(axis[by_id[a['sdk_frame_id']]] for a in checks)) if len(checks) >= 2 else 0
        coverage = check_span/max(float(axis[-1]-axis[0]), 1e-12)
        validated = len(checks) >= 2 and coverage >= .8 and max(abs(e) for e in errors) <= tolerance_ms
        stats.update(status='checked' if validated else 'unvalidated', basis=basis,
                     offset_seconds=float(intercept), drift_ppm=float((slope-1)*1e6) if len(fit) >= 2 else None,
                     check_events=len(checks), check_coverage_fraction=float(coverage),
                     check_max_error_ms=float(max((abs(e) for e in errors), default=0)) if errors else None,
                     check_rms_error_ms=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                     drift_note='ID-based drift is frame-count consistency, not an independent oscillator measurement.'
                     if basis == 'sdk_id_at_nominal_rate' else 'Hardware clock domain fitted to Vicon anchors.')
        used_frames = {}
        for row, stamp, camera_time in zip(rows, predicted, axis):
            fractional = float(vicon['first_frame'] + stamp*vicon['rate'])
            nearest = round(fractional)
            error_ms = abs(nearest-fractional)/vicon['rate']*1000
            in_range = vicon['first_frame'] <= nearest <= vicon['last_frame']
            confidence = ('checked_estimate' if validated else 'unvalidated_estimate')
            if camera_time < min(x) or camera_time > max(x):
                confidence = 'extrapolated_estimate'
            if not in_range or error_ms > tolerance_ms:
                confidence = 'outside_export' if not in_range else 'between_mocap_samples'
            result = {'camera_index': camera, 'video_frame_index': row['video_frame_index'],
                          'sdk_frame_id': row.get('sdk_frame_id'), 'vicon_frame': nearest if in_range and error_ms <= tolerance_ms else '',
                          'vicon_time_seconds': float(stamp), 'estimated_frame': fractional, 'confidence': confidence}
            mapping.append(result)
            if result['vicon_frame'] != '':
                used_frames.setdefault(nearest, []).append(result)
        for same_frame in used_frames.values():
            if len(same_frame) > 1:
                for row in same_frame:
                    row.update(vicon_frame='', confidence='ambiguous_multiple_images')
        stats['unmapped_images'] = sum(r['vicon_frame'] == '' for r in mapping if r['camera_index'] == camera)
    report['diagnostics'] = diagnostics.summary()
    if exposure_channel:
        if vicon is None:
            raise ValueError("Exposure channel requires a Vicon C3D file")
        edges = exposure_edges(vicon, exposure_channel, threshold)
        _write_csv(output / 'exposure_edges.csv', edges,
                   ['edge_index', 'analog_sample', 'vicon_frame', 'vicon_time_seconds',
                    'pulse_width_seconds', 'midpoint_vicon_frame', 'truncated'])
        periods = np.diff([edge['vicon_time_seconds'] for edge in edges])
        phases = np.asarray([edge['vicon_frame']-vicon['first_frame'] for edge in edges]) % 1
        center = float(np.angle(np.mean(np.exp(2j*np.pi*phases)))/(2*np.pi) % 1) if len(phases) else 0
        deviations = ((phases-center+.5) % 1 - .5)/vicon['rate']*1000
        report['exposure_signal'] = {'channel': exposure_channel, 'threshold': threshold,
            'analog_rate': vicon['analog_rate'], 'edges': len(edges),
            'sample_resolution_ms': 1000/vicon['analog_rate'],
            'phase_within_vicon_frame_ms': center/vicon['rate']*1000 if len(phases) else None,
            'phase_deviation_p95_ms': float(np.percentile(abs(deviations), 95)) if len(phases) else None,
            'median_period_seconds': float(np.median(periods)) if len(periods) else None,
            'note': 'Sampled threshold crossings; narrow or clipped pulses may be missed. Not automatically paired to images.'}
    _write_csv(output / 'frame_mapping.csv', mapping,
               ['camera_index', 'video_frame_index', 'sdk_frame_id', 'vicon_frame',
                'vicon_time_seconds', 'estimated_frame', 'confidence'])
    report['mapping_sha256'] = file_hash(output/'frame_mapping.csv')
    (output / 'alignment.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines = [f"Trial: {report['trial_name']}", f"Recording complete: {report['complete']}",
             f"Queue drops: {report['queue_drops']}"]
    lines.extend(f"Camera {key}: {stats['status']}; {stats.get('reason', stats.get('basis'))}"
                 for key, stats in report['cameras'].items())
    lines.append('Checked means agreement with user-supplied independent events, not hardware certification.')
    (output / 'summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report


def _write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
