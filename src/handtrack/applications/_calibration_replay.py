"""Recompute a calibration from saved observations without opening cameras."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import _calibration_runtime as runtime


def unpack_observations(data, phase, camera):
    offsets = np.asarray(data[f'{phase}_offsets_{camera}'])
    corners = np.asarray(data[f'{phase}_corners_{camera}'])
    ids = np.asarray(data[f'{phase}_ids_{camera}'])
    if (offsets.ndim != 1 or not offsets.size or offsets.dtype.kind not in 'iu'
            or offsets[0] != 0 or np.any(np.diff(offsets) < 0)
            or offsets[-1] != len(ids) or len(corners) != len(ids)):
        raise ValueError(f'Invalid {phase} offsets for camera {camera}')
    return ([corners[a:b].reshape(-1, 1, 2).astype(np.float32) if b > a else None
             for a, b in zip(offsets[:-1], offsets[1:])],
            [ids[a:b].reshape(-1, 1).astype(np.int32) if b > a else None
             for a, b in zip(offsets[:-1], offsets[1:])])


def replay_calibration(observations, output):
    """Use all existing quality gates and atomic publication at output."""
    with np.load(observations, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if int(data['schema_version']) != 1:
        raise ValueError('Unsupported observation archive version')
    count = int(data['num_cameras'])
    sizes = data['image_sizes']
    if count < 1 or sizes.shape != (count, 2) or not np.all(sizes == sizes[0]):
        raise ValueError('Replay requires matching camera resolutions')
    board = SimpleNamespace(getChessboardCorners=lambda: data['board_points'])
    state = dict(_camera_ids=data['camera_ids'], _image_sizes=sizes,
                 _diagnostics=None, _intrinsic_results=[], _stereo_results=[],
                 _reserved_indices=[], _board_validation=None,
                 CALIBRATION_FILE=str(output),
                 CHARUCO_SQUARE_LENGTH=float(data['charuco_square_length']),
                 CHARUCO_MARKER_LENGTH=float(data['charuco_marker_length']))
    if 'camera_uids' in data:
        state['_diagnostics'] = [
            dict(uid=tuple(uid), serial=str(serial)) for uid, serial in
            zip(data['camera_uids'], data['camera_serials'])]
    previous = {key: getattr(runtime, key, None) for key in state}
    try:
        for key, value in state.items():
            setattr(runtime, key, value)
        intrinsic = [unpack_observations(data, 'intrinsic', camera) for camera in range(count)]
        matrices, dists = runtime.calibrate_intrinsics(
            [item[0] for item in intrinsic], [item[1] for item in intrinsic],
            board, tuple(sizes[0]), count, reserve_board_views=False)
        runtime.validate_intrinsic_quality()
        alignment = [unpack_observations(data, 'alignment', camera) for camera in range(count)]
        rotations, translations = runtime.calibrate_stereo_pairs(
            [item[0] for item in alignment], [item[1] for item in alignment],
            board, matrices, dists, tuple(sizes[0]), count,
            reserved_indices=data['reserved_view_indices'])
        for pair in runtime._stereo_results:
            print(f'Pair {pair.camera_a}-{pair.camera_b}: held-out max '
                  f'{np.max(pair.heldout_errors):.3f} px')
        if runtime._board_validation is not None:
            print(f'Board reconstruction: {runtime._board_validation}')
        captures = sum(len(item[0]) for item in intrinsic) + len(alignment[0][0])
        runtime.save_calibration(count, tuple(sizes[0]), captures, matrices, dists,
                                 rotations, translations)
    finally:
        for key, value in previous.items():
            setattr(runtime, key, value)
    return Path(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('observations', type=Path)
    parser.add_argument('--output', type=Path, required=True,
                        help='destination for the quality-checked calibration')
    args = parser.parse_args(argv)
    replay_calibration(args.observations, args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
