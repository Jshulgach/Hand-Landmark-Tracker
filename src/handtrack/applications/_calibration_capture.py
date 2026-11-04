"""OptiTrack capture UI; solver and atomic quality checks remain in calibration runtime."""
import concurrent.futures
import time

import numpy as np

from ._calibration_preview import create_calibration_preview


class BoardCaptureGate:
    """Require several observations of the same stationary board pose.

    Host receipt timestamps reject stale/repeated/skewed batches; they do not
    establish hardware exposure synchronization.
    """

    def __init__(self, stable_seconds=.5, max_motion_px=1.0, max_skew=.05):
        self.stable_seconds = stable_seconds
        self.max_motion_px = max_motion_px
        self.max_skew = max_skew
        self.reset()

    def reset(self):
        self.anchor = None
        self.since = None
        self.last_stamps = None

    def check(self, results, now, stamps=None):
        if not all(found for found, _ in results):
            self.reset()
            return False, "Board must be visible in every camera"
        if stamps is not None:
            times = np.asarray(stamps, dtype=float)
            if (not np.isfinite(times).all() or np.any(now - times < 0)
                    or np.any(now - times > .25) or np.ptp(times) > self.max_skew):
                self.reset()
                return False, "Waiting for fresh camera frames with close receipt times"
            if self.last_stamps is not None and any(
                    stamp <= old for stamp, old in zip(stamps, self.last_stamps)):
                return False, "Waiting for a new frame from every camera"
            self.last_stamps = tuple(stamps)
        current = [(np.asarray(corners).reshape(-1, 2).copy(),
                    np.asarray(ids).reshape(-1).copy()) for _, (corners, ids) in results]
        still = self.anchor is not None and len(current) == len(self.anchor)
        if still:
            for (points, ids), (old_points, old_ids) in zip(current, self.anchor):
                common, ii, jj = np.intersect1d(ids, old_ids, return_indices=True)
                if (len(common) < 6 or
                        np.max(np.linalg.norm(points[ii] - old_points[jj], axis=1)) > self.max_motion_px):
                    still = False
                    break
        if not still:
            self.anchor, self.since = current, now
        ready = now - self.since >= self.stable_seconds
        return ready, ("Board steady" if ready else "Hold board still for 0.5 s")


def _capture_snapshots(manager):
    workers = getattr(manager, "workers", ())
    if workers and all(hasattr(worker, "get_snapshot") for worker in workers):
        snapshots = [worker.get_snapshot() for worker in workers]
        frames = [frame if frame is not None else np.zeros((worker.height, worker.width, 3), np.uint8)
                  for worker, (frame, _) in zip(workers, snapshots)]
        stamps = [stamp if frame is not None and stamp is not None else float("nan")
                  for frame, stamp in snapshots]
        return frames, stamps
    return manager.get_all_frames(), None


def run_capture(module):
    manager = module.CameraManager()
    preview = None
    try:
        if manager.num_cameras == 0:
            print('No cameras found. Exiting.')
            return
        manager.start_all(exposure=10)
        if hasattr(manager,'wait_ready'):
            manager.wait_ready()
        count = manager.num_cameras
        image_size = manager.get_resolution(0)
        preview = create_calibration_preview(manager,'OptiTrack ChArUco Calibration',10)
        controls = preview.controls
        print('Calibration: Up/Down exposure, Shift step 10, I/K IR, M max light, H high power, 0-9 camera, A all, Space pause, Q finish, Esc cancel.')
        print(f'Board: {module.CHARUCO_SQUARES_X}x{module.CHARUCO_SQUARES_Y}; target: {module.NUM_CALIBRATION_IMAGES} captures.')
        detected = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=count) as executor:
            while detected is None:
                frames = manager.get_all_frames()
                preview.show_frames(frames,'Show the ChArUco board to a camera.')
                if controls.quit_requested:
                    return
                # Poll the UI while dictionary detection runs off the Qt thread.
                for frame in frames:
                    future = executor.submit(module.auto_detect_dictionary,frame)
                    while not future.done():
                        preview.show_frames(manager.get_all_frames(),'Detecting the board dictionary...')
                        if controls.quit_requested:
                            return
                        time.sleep(.01)
                    name,corners = future.result()
                    if name is not None and corners >= 4:
                        detected = name
                        break
            board,detector = module.build_charuco_detector(dict_name=detected)
            # Each concurrent task owns a detector; native detector state is
            # never shared across cameras.
            detectors = [detector] + [module.build_charuco_detector(dict_name=detected)[1]
                                      for _ in range(count - 1)]
            gate = BoardCaptureGate()
            print(f'Dictionary: {detected}. Auto-capture when all cameras see the board.')
            all_corners = [[] for _ in range(count)]
            all_ids = [[] for _ in range(count)]
            captured = 0
            last_capture = time.monotonic()
            while captured < module.NUM_CALIBRATION_IMAGES:
                batch_started = time.monotonic()
                frames, stamps = _capture_snapshots(manager)
                snapshot_now = time.perf_counter()
                futures = [executor.submit(module.detect_charuco,(frame,detector))
                           for frame,detector in zip(frames,detectors)]
                while not all(future.done() for future in futures):
                    preview.show_frames(manager.get_all_frames(),f'Checking board visibility ({captured}/{module.NUM_CALIBRATION_IMAGES})')
                    if controls.quit_requested:
                        break
                    time.sleep(.005)
                if controls.quit_requested:
                    break
                results = [future.result() for future in futures]
                visible = [found for found,_ in results]
                display = [module.annotate_frame(frame,index,found,data)
                           for index,(frame,(found,data)) in enumerate(zip(frames,results))]
                now = time.monotonic()
                steady, guidance = gate.check(results, snapshot_now if stamps is not None else now, stamps)
                if not controls.capture_ready or batch_started < controls.last_change:
                    gate.reset()
                    steady = False
                if (all(visible) and steady and controls.capture_ready
                        and batch_started >= controls.last_change and now-last_capture >= 1.0):
                    for index,(_,data) in enumerate(results):
                        corners,ids = data
                        all_corners[index].append(corners)
                        all_ids[index].append(ids)
                    captured += 1
                    gate.reset()
                    last_capture = now
                    print(f'Captured {captured}/{module.NUM_CALIBRATION_IMAGES}')
                    status = f'Captured {captured}/{module.NUM_CALIBRATION_IMAGES}. Change board angle and position.'
                elif controls.paused:
                    status = f'Paused ({captured}/{module.NUM_CALIBRATION_IMAGES}). Space to resume.'
                elif not controls.capture_ready:
                    status = f'Exposure settling ({captured}/{module.NUM_CALIBRATION_IMAGES})'
                else:
                    status = (f'{guidance} | Board {sum(visible)}/{count} cameras | '
                              f'Captures {captured}/{module.NUM_CALIBRATION_IMAGES}')
                preview.show_frames(display,status)
                if controls.quit_requested:
                    break
        preview.shutdown()
        if controls.cancelled:
            print('Calibration cancelled; previous calibration preserved.')
            return
        if captured < 10:
            print(f'Not enough captures ({captured}); previous calibration preserved.')
            return
        print(f'Computing calibration from {captured} image sets...')
        matrices,dists = module.calibrate_intrinsics(all_corners,all_ids,board,image_size,count)
        rotations,translations = module.calibrate_stereo_pairs(all_corners,all_ids,board,matrices,dists,image_size,count)
        module.save_calibration(count,image_size,captured,matrices,dists,rotations,translations)
        print('Calibration complete.')
    finally:
        if preview is not None:
            preview.shutdown()
        manager.stop_all()
