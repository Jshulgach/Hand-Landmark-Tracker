from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from handtrack.processing._wilor_geometry import reconstruct_wilor
from mavis_track.wilor import WiLoRHandResult, WiLoRFrameResult


def rig(label='Right'):
    rng=np.random.default_rng(64)
    joints=rng.normal(0,.035,(21,3))
    joints[0]=0
    joints+=np.array([.03,-.02,1.5])
    vertices=rng.normal(0,.035,(778,3))+joints[0]
    matrix=np.array([[950.,0,640],[0,950,512],[0,0,1.]])
    rotations=[np.eye(3),cv2.Rodrigues(np.array([0.,.3,0.]))[0],np.eye(3)]
    translations=[np.zeros((3,1)),-rotations[1]@np.array([[.4],[0.],[0.]]),np.array([[0.],[-.35],[0.]])]
    calibration=([matrix]*3,[np.zeros(5)]*3,rotations,translations,True)
    results=[]
    for c in range(3):
        rv,_=cv2.Rodrigues(rotations[c])
        pixels,_=cv2.projectPoints(joints,rv,translations[c],matrix,np.zeros(5))
        mesh_pixels,_=cv2.projectPoints(vertices,rv,translations[c],matrix,np.zeros(5))
        hand=WiLoRHandResult(100+c,label,.9,pixels.reshape(21,2),mesh_pixels.reshape(778,2),
            (joints-joints[0])@rotations[c].T/1.1,
            (vertices-joints[0])@rotations[c].T/1.1,
            np.array([999.,999.,999.]),np.array([0.,0.,100.,100.]),{})
        results.append(WiLoRFrameResult(0,1.,f'camera:{c}',(1280,1024),(hand,),5000.,'caller'))
    return results,calibration,joints,vertices


@pytest.mark.parametrize('label',['Left','Right'])
def test_calibrated_mesh_and_skeleton_share_world_transform(label):
    results,calibration,joints,vertices=rig(label)
    hands=reconstruct_wilor(results,[1.,1.01,1.02],calibration)
    assert len(hands)==1
    hand=hands[0]
    assert hand.handedness==label and hand.cameras==(0,1,2)
    assert hand.mesh_scale==pytest.approx(1.1)
    np.testing.assert_allclose(hand.triangulated_joints,joints,atol=1e-8)
    np.testing.assert_allclose(hand.joints,joints,atol=1e-8)
    np.testing.assert_allclose(hand.vertices,vertices,atol=1e-8)
    assert hand.mesh_fit_rms_mm<1e-5 and hand.joint_reprojection_rms_px<1e-5
    assert not hand.vertices.flags.writeable
    assert results[0].hands[0].wrist_camera_translation[0]==999.


def test_missing_disabled_stale_and_ambiguous_views_do_not_produce_geometry():
    results,calibration,_,_=rig()
    assert reconstruct_wilor(results,[1.,1.,1.],calibration,enabled={0})==()
    assert reconstruct_wilor(results,[1.,1.2,1.4],calibration)==()
    assert reconstruct_wilor([results[0],None,None],[1.,None,None],calibration)==()
    duplicate=replace(results[0],hands=results[0].hands*2)
    assert reconstruct_wilor([duplicate,*results[1:]],[1.,1.,1.],calibration)==()


def test_reprojection_consensus_rejects_a_bad_camera():
    results,calibration,joints,_=rig()
    bad=replace(results[2].hands[0],image_landmarks=results[2].hands[0].image_landmarks+[100.,100.])
    results[2]=replace(results[2],hands=(bad,))
    hands=reconstruct_wilor(results,[1.,1.,1.],calibration,max_reprojection_px=2.)
    assert len(hands)==1 and hands[0].cameras==(0,1)
    np.testing.assert_allclose(hands[0].triangulated_joints,joints,atol=1e-8)


def test_no_hand_or_geometry_is_fabricated_when_cross_view_agreement_fails():
    results,calibration,_,_=rig()
    bad=replace(results[1].hands[0],image_landmarks=results[1].hands[0].image_landmarks+[0.,300.])
    results=[results[0],replace(results[1],hands=(bad,)),None]
    assert reconstruct_wilor(results,[1.,1.,None],calibration,max_reprojection_px=1.)==()
