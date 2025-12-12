import mediapipe as mp


def get_3point_finger_indices():
    # 3D angle computation based on 3-point vectors
    hand_lm = mp.solutions.hands.HandLandmark

    # Use the mediapipe landmark indices instead of hardcoded
    finger_indices_mp = [
        (hand_lm.WRIST, hand_lm.THUMB_CMC, hand_lm.THUMB_TIP),
        (hand_lm.WRIST, hand_lm.INDEX_FINGER_MCP, hand_lm.INDEX_FINGER_TIP),
        (hand_lm.WRIST, hand_lm.MIDDLE_FINGER_MCP, hand_lm.MIDDLE_FINGER_TIP),
        (hand_lm.WRIST, hand_lm.RING_FINGER_MCP, hand_lm.RING_FINGER_TIP),
        (hand_lm.WRIST, hand_lm.PINKY_MCP, hand_lm.PINKY_TIP),
    ]
    return finger_indices_mp

def get_hand_connections():
    # want to get the hand connections from mediapipe
    mp_hands = mp.solutions.hands
    return mp_hands.HAND_CONNECTIONS
