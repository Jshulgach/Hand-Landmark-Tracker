"""Capture bookkeeping for individual lenses followed by overlapping camera pairs."""
import itertools
import os
from datetime import datetime
from pathlib import Path

import numpy as np


MIN_STAGE_CAPTURES = 20


class CalibrationSession:
    def __init__(self, count, target):
        self.count = count
        self.target = max(MIN_STAGE_CAPTURES, target)
        self.camera = 0
        self.phase = 'intrinsics'
        self.intrinsic_corners = [[] for _ in range(count)]
        self.intrinsic_ids = [[] for _ in range(count)]
        self.alignment_corners = [[] for _ in range(count)]
        self.alignment_ids = [[] for _ in range(count)]
        self.pair_counts = {pair: 0 for pair in itertools.combinations(range(count), 2)}
        self.reserved_indices = []
        self.active_target = None

    def connected(self, minimum):
        reached = {0}
        while True:
            previous = len(reached)
            for (a, b), count in self.pair_counts.items():
                if count >= minimum and (a in reached or b in reached):
                    reached.update((a, b))
            if len(reached) == previous:
                return len(reached) == self.count

    def choose_target(self, results):
        if self.phase == 'intrinsics':
            return (self.camera,) if results[self.camera][0] else None
        eligible = []
        for pair, count in self.pair_counts.items():
            a, b = pair
            if count >= self.target or not results[a][0] or not results[b][0]:
                continue
            ids_a, ids_b = results[a][1][1], results[b][1][1]
            if len(np.intersect1d(ids_a, ids_b)) >= 6:
                eligible.append(pair)
        # Keep the same pair through its countdown; prefer finishing an
        # existing pair over starting another when multiple pairs are visible.
        if self.active_target in eligible:
            return self.active_target
        return max(eligible, key=lambda pair: self.pair_counts[pair], default=None)

    def capture(self, target, results):
        if self.phase == 'intrinsics':
            camera = target[0]
            corners, ids = results[camera][1]
            self.intrinsic_corners[camera].append(corners.copy())
            self.intrinsic_ids[camera].append(ids.copy())
            return
        view = len(self.alignment_corners[0])
        # Reserve every fifth observation of EACH pair for independent board
        # validation, even when captures switch between different pairs.
        if self.pair_counts[target] % 5 == 4:
            self.reserved_indices.append(view)
        for camera in range(self.count):
            corners, ids = results[camera][1] if camera in target else (None, None)
            self.alignment_corners[camera].append(None if corners is None else corners.copy())
            self.alignment_ids[camera].append(None if ids is None else ids.copy())
        self.pair_counts[target] += 1

    def progress(self):
        if self.phase == 'intrinsics':
            count = len(self.intrinsic_corners[self.camera])
            return (f'Stage 1/2 — Camera {self.camera} intrinsics: {count}/{self.target}. '
                    'Show this camera a close board; vary position, depth and tilt.')
        pairs = ', '.join(f'{a}-{b}: {count}/{self.target}'
                          for (a, b), count in self.pair_counts.items())
        target = (f'Capturing pair {self.active_target[0]}-{self.active_target[1]}. '
                  if self.active_target else '')
        return (f'Stage 2/2 — Alignment. {target}Show the board to any two cameras; '
                f'connect every camera through completed pairs. {pairs}')

    def write_observations(self, path, module, manager, board, dictionary,
                           matrices=None, dists=None):
        """Persist numeric arrays, including missing views, without pickle."""
        data = dict(schema_version=1, phase=self.phase, dictionary=dictionary,
                    num_cameras=self.count, target_captures=self.target,
                    image_sizes=[manager.get_resolution(i) for i in range(self.count)],
                    camera_ids=getattr(manager, 'camera_indices', list(range(self.count))),
                    board_points=board.getChessboardCorners(),
                    charuco_square_length=module.CHARUCO_SQUARE_LENGTH,
                    charuco_marker_length=module.CHARUCO_MARKER_LENGTH,
                    reserved_view_indices=np.asarray(self.reserved_indices, dtype=int))
        if hasattr(manager, 'get_camera_diagnostics'):
            diagnostics = manager.get_camera_diagnostics()
            data['camera_serials'] = np.asarray([str(d.get('serial', '')) for d in diagnostics])
            data['camera_uids'] = np.asarray([d.get('uid') or (0, 0) for d in diagnostics])
        for phase, corners, ids in (
            ('intrinsic', self.intrinsic_corners, self.intrinsic_ids),
            ('alignment', self.alignment_corners, self.alignment_ids),
        ):
            for camera in range(self.count):
                lengths = [0 if values is None else len(values) for values in ids[camera]]
                data[f'{phase}_offsets_{camera}'] = np.r_[0, np.cumsum(lengths)].astype(np.int64)
                present = [i for i, size in enumerate(lengths) if size]
                data[f'{phase}_corners_{camera}'] = (
                    np.concatenate([corners[camera][i].reshape(-1, 2) for i in present])
                    if present else np.empty((0, 2), dtype=np.float32))
                data[f'{phase}_ids_{camera}'] = (
                    np.concatenate([ids[camera][i].reshape(-1) for i in present])
                    if present else np.empty(0, dtype=np.int32))
        for pair, count in self.pair_counts.items():
            data[f'pair_count_{pair[0]}_{pair[1]}'] = count
        if matrices is not None:
            for camera in range(self.count):
                data[f'intrinsic_camera_matrix_{camera}'] = matrices[camera]
                data[f'intrinsic_dist_coeffs_{camera}'] = dists[camera]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, **data)
        os.replace(temporary, path)


def observation_path(module):
    directory = Path(module.CALIBRATION_FILE).parent
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    return directory / f'calibration_observations_{stamp}.npz'
