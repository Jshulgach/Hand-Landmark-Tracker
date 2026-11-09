"""Capture individual camera intrinsics, then align overlapping camera pairs."""
import concurrent.futures
import time

from ._calibration_preview import create_calibration_preview
from ._calibration_session import CalibrationSession, MIN_STAGE_CAPTURES, observation_path
from ._calibration_runtime import validate_intrinsic_quality


def _collect(module, manager, preview, executor, detectors, session, checkpoint):
    """Collect one stage, keeping the preview responsive during detection."""
    controls = preview.controls
    countdown = controls.countdown
    while not controls.cancelled:
        countdown.waiting_message = (
            f'Waiting for board in camera {session.camera} — other cameras are not required'
            if session.phase == 'intrinsics' else
            'Waiting for an unfinished camera pair with at least 6 common corners')
        batch_started = time.monotonic()
        frames = manager.get_all_frames()
        futures = [executor.submit(module.detect_charuco, (frame, detector))
                   for frame, detector in zip(frames, detectors)]
        while not all(future.done() for future in futures):
            preview.show_frames(manager.get_all_frames(), session.progress())
            if controls.quit_requested:
                break
            time.sleep(.005)
        if controls.cancelled:
            return False
        results = [future.result() for future in futures]
        display = [module.annotate_frame(frame, index, found, data)
                   for index, (frame, (found, data)) in enumerate(zip(frames, results))]
        target = session.choose_target(results)
        if target != session.active_target:
            countdown.reset()
            session.active_target = target
        ready = (target is not None and controls.capture_ready
                 and batch_started >= controls.last_change)
        countdown.update(time.monotonic(), ready)
        captured = ready and countdown.due(batch_started) and not controls.finished
        if captured:
            session.capture(target, results)
            countdown.reset()
            checkpoint()
            print('Captured — ' + session.progress())

        advance = False
        message = session.progress()
        if session.phase == 'intrinsics':
            size = len(session.intrinsic_corners[session.camera])
            if size >= session.target or (controls.finished and size >= MIN_STAGE_CAPTURES):
                session.camera += 1
                advance = session.camera == session.count
                if not advance:
                    controls.paused = True
                    session.active_target = None
                    message = session.progress() + ' Press Space when ready for this camera.'
            elif controls.finished:
                message += f' Need at least {MIN_STAGE_CAPTURES} captures before Q can advance.'
        else:
            advance = session.connected(session.target)
            if controls.finished:
                advance = session.connected(MIN_STAGE_CAPTURES)
                if not advance:
                    message += f' Q needs a connected graph with {MIN_STAGE_CAPTURES}+ captures per linking pair.'
        controls.finished = False
        if advance:
            countdown.reset()
            return True
        if controls.paused:
            message += ' PAUSED — Space to resume.'
        elif captured:
            message = 'Captured! ' + message
        preview.show_frames(display, message)
    return False


def run_capture(module):
    manager = module.CameraManager()
    preview = None
    try:
        if manager.num_cameras == 0:
            print('No cameras found. Exiting.')
            return
        manager.start_all(exposure=10)
        if hasattr(manager, 'wait_ready'):
            manager.wait_ready()
        count = manager.num_cameras
        image_size = manager.get_resolution(0)
        preview = create_calibration_preview(manager, 'OptiTrack ChArUco Calibration', 10)
        controls = preview.controls
        controls.instructions = (*controls.instructions[:2],
            'Space: pause/resume   Q: next camera / finish alignment (20+ captures)   Esc: cancel')
        print('Calibration: Space pause/resume, Q advance when enough captures, Esc cancel.')
        print('Stage 1: individual camera intrinsics. Stage 2: overlapping camera alignment.')
        session = CalibrationSession(count, module.NUM_CALIBRATION_IMAGES)
        path = observation_path(module)
        matrices = dists = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=count) as executor:
            detected = None
            while detected is None:
                frames = manager.get_all_frames()
                preview.show_frames(frames, 'Show the ChArUco board to a camera to detect its dictionary.')
                if controls.quit_requested:
                    return
                for frame in frames:
                    future = executor.submit(module.auto_detect_dictionary, frame)
                    while not future.done():
                        preview.show_frames(manager.get_all_frames(), 'Detecting the board dictionary...')
                        if controls.quit_requested:
                            return
                        time.sleep(.01)
                    name, corners = future.result()
                    if name is not None and corners >= 4:
                        detected = name
                        break
            board, detector = module.build_charuco_detector(dict_name=detected)
            # Each concurrent camera task owns its detector's mutable state.
            detectors = [detector] + [module.build_charuco_detector(dict_name=detected)[1]
                                      for _ in range(count - 1)]
            def checkpoint():
                session.write_observations(path, module, manager, board, detected, matrices, dists)
            checkpoint()
            print(f'Dictionary: {detected}. Observations: {path}')
            if not _collect(module, manager, preview, executor, detectors, session, checkpoint):
                print('Calibration cancelled; previous calibration preserved.')
                return
            preview.show_frames(manager.get_all_frames(), 'Computing individual camera intrinsics...')
            if controls.cancelled:
                return
            matrices, dists = module.calibrate_intrinsics(
                session.intrinsic_corners, session.intrinsic_ids, board, image_size, count,
                reserve_board_views=False)
            checkpoint()
            validate_intrinsic_quality()
            print('Intrinsic quality checks passed. Lens parameters will remain fixed for alignment.')
            if count > 1:
                session.phase = 'alignment'
                session.active_target = None
                controls.paused = True
                controls.countdown.reset()
                checkpoint()
                if not _collect(module, manager, preview, executor, detectors, session, checkpoint):
                    print('Calibration cancelled; previous calibration preserved.')
                    return
                preview.show_frames(manager.get_all_frames(), 'Computing camera alignment with fixed intrinsics...')
                if controls.cancelled:
                    return
                rotations, translations = module.calibrate_stereo_pairs(
                    session.alignment_corners, session.alignment_ids, board,
                    matrices, dists, image_size, count, reserved_indices=session.reserved_indices)
            else:
                import numpy as np
                rotations, translations = [np.eye(3)], [np.zeros((3, 1))]
            captures = sum(map(len, session.intrinsic_corners)) + len(session.alignment_corners[0])
            module.save_calibration(count, image_size, captures, matrices, dists, rotations, translations)
            print(f'Calibration complete. Observations saved: {path}')
    finally:
        if preview is not None:
            preview.shutdown()
        manager.stop_all()
