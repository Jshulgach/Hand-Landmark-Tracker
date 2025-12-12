import numpy as np


def angle_between_points(a, b, c, degrees=True):
    """
    Computes the angle at point `b` formed by segments `ab` and `bc`.

    Args:
        a, b, c (np.ndarray): 3D coordinates of the points.
        degrees (bool): Whether to return angle in degrees (True) or radians (False).

    Returns:
        float: Angle at point `b` in degrees or radians.
    """
    ab = a - b
    cb = c - b
    dot_product = np.dot(ab, cb)
    norm_product = np.linalg.norm(ab) * np.linalg.norm(cb)
    if norm_product == 0:
        cosine_angle = 1.0  # or some default
    else:
        cosine_angle = np.clip(dot_product / norm_product, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle) if degrees else angle


def compute_all_joint_angles(landmarks):
    """
    Computes angles at each landmark by treating each point as the center of a local triangle (i-1, i, i+1).

    Args:
        landmarks (np.ndarray): Array of shape (21, 3)

    Returns:
        dict: Map of index -> angle (in degrees) for valid joints.
    """
    angles = {}
    for i in range(1, len(landmarks)-1):
        try:
            angle = angle_between_points(landmarks[i - 1], landmarks[i], landmarks[i + 1])
            angles[i] = angle
        except:
            continue
    return angles


def compute_3point_finger_angles(landmarks):
    """
    Computes angle for each finger: fingertip -> MCP -> wrist.

    Args:
        landmarks (np.ndarray): Array of shape (21, 3)

    Returns:
        dict: Map of finger name -> angle at MCP.
    """
    wrist = landmarks[0]
    indices = {
        'thumb':  (4, 2),   # tip, mcp
        'index':  (8, 5),
        'middle': (12, 9),
        'ring':   (16, 13),
        'pinky':  (20, 17),
    }
    angles = {}
    for finger, (tip_idx, mcp_idx) in indices.items():
        tip = landmarks[tip_idx]
        mcp = landmarks[mcp_idx]
        angles[finger] = angle_between_points(tip, mcp, wrist)
    return angles

#
#
# def angle_between(v1, v2):
#     v1 = np.array(v1)
#     v2 = np.array(v2)
#     v1 /= np.linalg.norm(v1)
#     v2 /= np.linalg.norm(v2)
#     raw_angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
#     return raw_angle
#
#
# def compute_wrist_to_fingertip_joint_angles(hand_landmarks):
#
#     coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
#     angles = {}
#     angles["Thumb_MCP"] = angle_between(coords[2] - coords[1], coords[3] - coords[2])
#     angles["Thumb_IP"] = angle_between(coords[3] - coords[2], coords[4] - coords[3])
#
#     finger_names = ["Index", "Middle", "Ring", "Pinky"]
#     for i, base in enumerate([5, 9, 13, 17]):
#         mcp = coords[base] - coords[0]             # MCP to wrist
#         pip = coords[base + 1] - coords[base]      # PIP to MCP
#         dip = coords[base + 2] - coords[base + 1]  # DIP to PIP
#         tip = coords[base + 3] - coords[base + 2]  # Tip to DIP
#
#         angles[f"{finger_names[i]}_MCP"] = angle_between(mcp, pip)
#         angles[f"{finger_names[i]}_PIP"] = angle_between(pip, dip)
#         angles[f"{finger_names[i]}_DIP"] = angle_between(dip, tip)
#     return angles
#
#
# def compute_finger_angles(landmarks):
#     """
#     Computes the MCP joint angles (Palm, MCP, FingerTip) for each finger.
#     Returns a list of 5 angles (thumb, index, middle, ring, pinky).
#     """
#     # Define indices
#     idx_wrist = mp.solutions.hands.HandLandmark.WRIST
#     idx_thumb_tip = mp.solutions.hands.HandLandmark.THUMB_TIP
#     idx_thumb_mcp = mp.solutions.hands.HandLandmark.THUMB_MCP
#     idx_index_tip = mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP
#     idx_index_mcp = mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP
#     idx_middle_tip = mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP
#     idx_middle_mcp = mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP
#     idx_ring_tip = mp.solutions.hands.HandLandmark.RING_FINGER_TIP
#     idx_ring_mcp = mp.solutions.hands.HandLandmark.RING_FINGER_MCP
#     idx_pinky_tip = mp.solutions.hands.HandLandmark.PINKY_TIP
#     idx_pinky_mcp = mp.solutions.hands.HandLandmark.PINKY_MCP
#
#     def calc_angle(a, b, c):
#         """Returns angle (in degrees) at point b given 3D points a, b, c."""
#         ba = a - b
#         bc = c - b
#         cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#         angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
#         return np.degrees(angle)
#
#     angles = []
#     try:
#         # Palm landmark: wrist
#         palm = landmarks[idx_wrist]
#
#         # Thumb
#         angles.append(calc_angle(landmarks[idx_thumb_tip], landmarks[idx_thumb_mcp], palm))
#         # Index
#         angles.append(calc_angle(landmarks[idx_index_tip], landmarks[idx_index_mcp], palm))
#         # Middle
#         angles.append(calc_angle(landmarks[idx_middle_tip], landmarks[idx_middle_mcp], palm))
#         # Ring
#         angles.append(calc_angle(landmarks[idx_ring_tip], landmarks[idx_ring_mcp], palm))
#         # Pinky
#         angles.append(calc_angle(landmarks[idx_pinky_tip], landmarks[idx_pinky_mcp], palm))
#     except Exception as e:
#         print(f"Error computing angles: {e}")
#         angles = [0.0] * 5  # fallback
#
#     return angles
