from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from handtrack.applications._calibration_session import CalibrationSession
from handtrack.applications import _calibration_runtime as runtime


def _results(visible, count=4):
    corners=np.zeros((20,1,2),np.float32)
    ids=np.arange(20,dtype=np.int32).reshape(-1,1)
    return [(True,(corners,ids)) if camera in visible else (False,(None,None))
            for camera in range(count)]


def test_alignment_needs_a_connected_graph_but_never_all_cameras_visible():
    session=CalibrationSession(4,20)
    session.phase='alignment'
    for pair in [(0,1),(2,3)]:
        results=_results(pair)
        for _ in range(20):
            assert session.choose_target(results)==pair
            session.capture(pair,results)
    assert not session.connected(20)
    results=_results((1,2))
    for _ in range(20):
        session.capture((1,2),results)
    assert session.connected(20)
    assert len(session.reserved_indices)==12
    # Each synchronized observation belongs to exactly its chosen pair.
    assert all(sum(session.alignment_ids[camera][view] is not None for camera in range(4))==2
               for view in range(60))


def test_pair_selection_requires_shared_corner_ids_and_keeps_target():
    session=CalibrationSession(4,20)
    session.phase='alignment'
    results=_results((0,1,2))
    session.active_target=(1,2)
    assert session.choose_target(results)==(1,2)
    results[2]=(True,(np.zeros((6,1,2),np.float32),np.arange(30,36,dtype=np.int32)))
    assert session.choose_target(results)==(0,1)
    assert session.choose_target(_results((0,))) is None


def test_intrinsics_need_only_current_camera():
    session=CalibrationSession(4,20)
    assert session.choose_target(_results((0,)))==(0,)
    assert session.choose_target(_results((1,2,3))) is None
    session.camera=2
    assert session.choose_target(_results((2,)))==(2,)


def test_bad_intrinsics_stop_before_alignment(monkeypatch):
    monkeypatch.setattr(runtime,'_intrinsic_results',[
        SimpleNamespace(heldout_errors=np.array([.3,2.5]))])
    with pytest.raises(ValueError,match='alignment not started'):
        runtime.validate_intrinsic_quality()


def test_staged_solver_publishes_sparse_pair_chain_with_fixed_intrinsics(monkeypatch,tmp_path):
    """Exercise real lens fits, stereo fits, graph solve and publication gates."""
    count=3
    board_points=np.zeros((20,3),np.float32)
    board_points[:,:2]=np.mgrid[1:6,1:5].T.reshape(-1,2)*.04
    board=SimpleNamespace(getChessboardCorners=lambda:board_points)
    matrix=np.array([[900.,0,640],[0,900,512],[0,0,1]])
    shifts=[np.array([-.2*camera,0.,0.]) for camera in range(count)]
    ids=np.arange(20,dtype=np.int32).reshape(-1,1)
    session=CalibrationSession(count,36)
    for index in range(36):
        rotation=np.array([.3*np.sin(index*.7),.4*np.cos(index*.4),.1*np.sin(index*.2)])
        position=np.array([-.15+.2*np.sin(index*.3),-.1+.15*np.cos(index*.5),.8+.3*(index%5)/4])
        for camera in range(count):
            corners,_=cv2.projectPoints(board_points,rotation,position,matrix,np.zeros(5))
            session.intrinsic_corners[camera].append(corners.astype(np.float32))
            session.intrinsic_ids[camera].append(ids.copy())
    for name,value in {'_image_sizes':None,'_camera_ids':None,'_diagnostics':None,
                       '_intrinsic_results':[],'_stereo_results':[], '_reserved_indices':[],
                       '_board_validation':None,'CHARUCO_SQUARE_LENGTH':.04,
                       'CHARUCO_MARKER_LENGTH':.03,'CALIBRATION_FILE':str(tmp_path/'latest.npz')}.items():
        monkeypatch.setattr(runtime,name,value,raising=False)
    matrices,dists=runtime.calibrate_intrinsics(
        session.intrinsic_corners,session.intrinsic_ids,board,(1280,1024),count,
        reserve_board_views=False)
    runtime.validate_intrinsic_quality()
    original_matrices=[m.copy() for m in matrices]
    original_dists=[d.copy() for d in dists]
    session.phase='alignment'
    for pair in [(0,1),(1,2)]:
        for index in range(36):
            rotation=np.array([.25*np.sin(index*.7),.3*np.cos(index*.4),.05*np.sin(index*.2)])
            position=np.array([-.05+.15*np.sin(index*.3),-.1+.1*np.cos(index*.5),1.2+.3*(index%5)/4])
            results=[]
            for camera in range(count):
                corners,_=cv2.projectPoints(board_points,rotation,position+shifts[camera],matrix,np.zeros(5))
                results.append((True,(corners.astype(np.float32),ids))
                               if camera in pair else (False,(None,None)))
            session.capture(pair,results)
    rotations,translations=runtime.calibrate_stereo_pairs(
        session.alignment_corners,session.alignment_ids,board,matrices,dists,(1280,1024),count,
        reserved_indices=session.reserved_indices)
    assert [(p.camera_a,p.camera_b) for p in runtime._stereo_results]==[(0,1),(1,2)]
    for camera in range(count):
        np.testing.assert_array_equal(matrices[camera],original_matrices[camera])
        np.testing.assert_array_equal(dists[camera],original_dists[camera])
        np.testing.assert_allclose(translations[camera].reshape(3),shifts[camera],atol=.003)
    runtime.save_calibration(count,(1280,1024),180,matrices,dists,rotations,translations)
    with np.load(tmp_path/'latest.npz',allow_pickle=False) as published:
        assert published['quality_passed'].item()
        assert published['board_validation_views']>=3
    # Archived sparse observations can be replayed without a camera SDK or UI.
    from handtrack.applications._calibration_replay import replay_calibration
    archive=tmp_path/'observations.npz'
    session.write_observations(archive,
        SimpleNamespace(CHARUCO_SQUARE_LENGTH=.04,CHARUCO_MARKER_LENGTH=.03),
        SimpleNamespace(get_resolution=lambda camera:(1280,1024)),board,'DICT_5X5_50',matrices,dists)
    previous_results=runtime._intrinsic_results
    replay_calibration(archive,tmp_path/'replayed.npz')
    assert runtime._intrinsic_results is previous_results
    assert runtime.CALIBRATION_FILE==str(tmp_path/'latest.npz')
    with np.load(tmp_path/'replayed.npz',allow_pickle=False) as replayed:
        assert replayed['quality_passed'].item()
        assert replayed['stereo_error_metric'].item()=='joint_board_pose_point_rms'
        for camera in range(count):
            np.testing.assert_allclose(replayed[f'camera_matrix_{camera}'],matrices[camera])
