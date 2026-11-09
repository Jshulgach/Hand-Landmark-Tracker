import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from handtrack.applications import _cli, _validate_calibration as app
from handtrack.processing._calibration_validation import (
    reconstruct_endpoints, summarize_trial, summarize_locations)


def geometry(depth=2., baseline=.4):
    matrix=np.array([[900.,0,640],[0,900,512],[0,0,1]])
    calibration=([matrix]*2,[np.zeros(5)]*2,[np.eye(3)]*2,
                 [np.zeros((3,1)),np.array([[-baseline],[0.],[0.]])],True)
    points=np.zeros((20,3),np.float64)
    points[:,:2]=np.mgrid[1:6,1:5].T.reshape(-1,2)*.04
    points[:,2]=depth
    detections=[]
    for camera in range(2):
        pixels,_=cv2.projectPoints(points,np.zeros(3),calibration[3][camera],matrix,np.zeros(5))
        detections.append((True,(pixels,np.arange(20,dtype=np.int32).reshape(-1,1))))
    return calibration,detections,points


def test_free_triangulation_recovers_metric_endpoints_and_rejects_bad_views():
    calibration,detections,expected=geometry()
    points,cameras,error,angle=reconstruct_endpoints(detections,calibration,[0,19])
    np.testing.assert_allclose(points,expected[[0,19]],atol=1e-10)
    assert cameras==(0,1) and error<1e-8 and angle>2
    stats=summarize_trial([points,points],195.)
    assert stats['mean_signed_error_mm']==pytest.approx(5.)
    # An incorrect measured reference remains an error; it cannot pull the
    # reconstructed endpoints toward the requested distance.
    assert stats['mean_distance_mm']==pytest.approx(200.)
    detections[1][1][0][19,0,1]+=30
    with pytest.raises(ValueError,match='reprojection'):
        reconstruct_endpoints(detections,calibration,[0,19])


def test_requires_two_views_positive_depth_and_useful_ray_geometry():
    calibration,detections,_=geometry()
    with pytest.raises(ValueError,match='at least two'):
        reconstruct_endpoints([detections[0],(False,(None,None))],calibration,[0,19])
    for depth,baseline,message in [(-2.,.4,'behind'),(2.,1e-8,'parallel')]:
        calibration,detections,_=geometry(depth,baseline)
        with pytest.raises(ValueError,match=message):
            reconstruct_endpoints(detections,calibration,[0,19])


def test_jitter_and_return_displacement_have_distinct_meanings():
    points=np.array([[[0.,0,1.],[.2,0,1.]],[[.002,0,1.],[.202,0,1.]]])
    first=dict(location='center',trial=1,**summarize_trial(points,195.))
    second=dict(location='center',trial=2,**summarize_trial(points+[0,.01,0],195.))
    np.testing.assert_allclose(first['endpoint_jitter_rms_mm'],[1.,1.])
    assert first['distance_rmse_mm']==pytest.approx(5.)
    assert first['distance_std_mm']==pytest.approx(0.)
    assert summarize_locations([first])['center']['max_return_midpoint_shift_mm'] is None
    result=summarize_locations([first,second])['center']
    assert result['max_return_midpoint_shift_mm']==pytest.approx(10.)
    assert result['max_return_endpoint_rms_shift_mm']==pytest.approx(10.)
    assert result['trial_distance_range_mm']==pytest.approx(0.)


def test_snapshot_skew_excludes_old_camera_images():
    old=np.ones((8,10,3),np.uint8)
    current=np.full((8,10,3),2,np.uint8)
    manager=SimpleNamespace(workers=[
        SimpleNamespace(get_snapshot=lambda:(old,1.)),
        SimpleNamespace(get_snapshot=lambda:(current,1.1)),
        SimpleNamespace(get_snapshot=lambda:(None,None))],get_resolution=lambda camera:(10,8))
    frames,stamps=app._snapshots(manager)
    assert stamps==[None,1.1,None]
    assert not frames[0].any() and not frames[2].any()
    np.testing.assert_array_equal(frames[1],current)


@pytest.mark.parametrize('extra',[
    ['--distance-mm','nan'],['--distance-mm','0'],['--samples','1'],
    ['--corner-ids','0','0'],['--sample-hz','-1'],['--locations','center','center']])
def test_invalid_arguments_fail_before_camera_access(extra):
    with pytest.raises(SystemExit) as error:
        _cli.main(['validate-calibration','--backend','webcam','--distance-mm','200',*extra])
    assert error.value.code==2


@pytest.mark.parametrize('backend',['webcam','optitrack'])
@pytest.mark.parametrize('cancel,resolution_mismatch',[(False,False),(True,False),(False,True)])
def test_live_workflow_saves_reports_without_changing_calibration(
        monkeypatch,tmp_path,cancel,resolution_mismatch,backend):
    from handtrack.applications._calibration_preview import CalibrationControls
    calibration,detections,_=geometry()
    published=tmp_path/'calibration.npz'
    saved=dict(schema_version=2,num_cameras=2,coordinate_units='meters',
               image_sizes=np.array([[1280,1024]]*2),camera_ids=[0,1])
    for c in range(2):
        for prefix,values in zip(('camera_matrix','dist_coeffs','R','T'),calibration[:4]):
            saved[f'{prefix}_{c}']=values[c]
    np.savez(published,**saved)
    original=published.read_bytes()
    clock=[0.]
    def now():clock[0]+=.1;return clock[0]
    monkeypatch.setattr(app.time,'monotonic',now)
    sequence=[0]
    class Worker:
        def __init__(self,camera):self.camera=camera
        def get_snapshot(self):
            if self.camera==0:sequence[0]+=1
            return np.full((8,10,3),self.camera,np.uint8),float(sequence[0])
    class Manager:
        num_cameras=2
        camera_indices=[0,1]
        def __init__(self):self.workers=[];self.stopped=False;self.ready=False
        def start_all(self,**kwargs):self.workers=[Worker(0),Worker(1)]
        def wait_ready(self):
            assert self.workers
            self.ready=True
        def get_resolution(self,camera):
            assert self.workers and self.ready, 'Resolution requires running, ready camera workers'
            return (640,480) if resolution_mismatch else (1280,1024)
        def stop_all(self):self.stopped=True
    manager=Manager()
    config=SimpleNamespace(CALIBRATION_FILE=str(published),CHARUCO_SQUARES_X=6,CHARUCO_SQUARES_Y=5)
    module=SimpleNamespace(build_charuco_detector=lambda **kwargs:(object(),object()),
        detect_charuco=lambda args:detections[int(args[0][0,0,0])],
        annotate_frame=lambda frame,*args:frame.copy())
    original_import=app.importlib.import_module
    def fake_import(name,*args):
        if name==f'handtrack.cameras.{backend}.config':return config
        camera_module='multi_mjpeg' if backend=='optitrack' else 'multi_webcam'
        legacy='optitrack_cam_py' if backend=='optitrack' else 'webcam'
        if name==f'handtrack.cameras.{backend}.{camera_module}':return SimpleNamespace(CameraManager=lambda:manager)
        if name==f'unity_hand_tracking.{legacy}.calibration':return module
        return original_import(name,*args)
    monkeypatch.setattr(app.importlib,'import_module',fake_import)
    class Preview:
        def __init__(self):self.controls=CalibrationControls(manager);self.closed=False;self.shown=0
        def show_frames(self,frames,status):
            self.shown+=1
            if cancel and self.shown>2:self.controls.handle_key('escape')
            elif self.controls.paused:self.controls.handle_key('space')
            assert self.shown<1000
        def shutdown(self):self.closed=True
    preview=Preview()
    import handtrack.applications._calibration_preview as preview_module
    monkeypatch.setattr(preview_module,'create_calibration_preview',lambda *args:preview)
    args=_cli.build_parser().parse_args(['validate-calibration','--backend',backend,
        '--distance-mm','200','--locations','center','left','--trials','2','--samples','3',
        '--dictionary','DICT_5X5_50','--output-dir',str(tmp_path/'reports')])
    if resolution_mismatch:
        with pytest.raises(ValueError,match='resolution differs'):
            app.run_validation(args,backend)
        assert manager.stopped
        assert published.read_bytes()==original
        assert not (tmp_path/'reports').exists()
        return
    result=app.run_validation(args,backend)
    assert result==(1 if cancel else 0)
    assert manager.stopped and preview.closed
    assert published.read_bytes()==original
    report=json.loads(next((tmp_path/'reports').glob('*/report.json')).read_text())
    assert report['state']==('cancelled' if cancel else 'completed')
    assert report['independent_target_declared'] is False
    if not cancel:
        assert len(report['trials'])==4
        for trial in report['trials']:
            assert trial['distance_rmse_mm']<1e-8
        assert report['locations']['center']['trials']==2
