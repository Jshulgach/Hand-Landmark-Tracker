"""OptiTrack capture UI; solver and atomic quality checks remain in calibration runtime."""
import concurrent.futures
import time
from ._calibration_preview import create_calibration_preview


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
            print(f'Dictionary: {detected}. Auto-capture when all cameras see the board.')
            all_corners = [[] for _ in range(count)]
            all_ids = [[] for _ in range(count)]
            captured = 0
            last_capture = time.monotonic()
            while captured < module.NUM_CALIBRATION_IMAGES:
                batch_started = time.monotonic()
                frames = manager.get_all_frames()
                futures = [executor.submit(module.detect_charuco,(frame,detector)) for frame in frames]
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
                if (all(visible) and controls.capture_ready
                        and batch_started >= controls.last_change and now-last_capture >= 1.0):
                    for index,(_,data) in enumerate(results):
                        corners,ids = data
                        all_corners[index].append(corners)
                        all_ids[index].append(ids)
                    captured += 1
                    last_capture = now
                    print(f'Captured {captured}/{module.NUM_CALIBRATION_IMAGES}')
                    status = f'Captured {captured}/{module.NUM_CALIBRATION_IMAGES}. Change board angle and position.'
                elif controls.paused:
                    status = f'Paused ({captured}/{module.NUM_CALIBRATION_IMAGES}). Space to resume.'
                elif not controls.capture_ready:
                    status = f'Exposure settling ({captured}/{module.NUM_CALIBRATION_IMAGES})'
                else:
                    status = f'Board visible in {sum(visible)}/{count} cameras | Captures {captured}/{module.NUM_CALIBRATION_IMAGES}'
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
