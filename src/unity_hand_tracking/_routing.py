"""Helpers for routing hand angle packets to Unity listeners."""

DEFAULT_UNITY_PORT_LEFT = 5015
DEFAULT_UNITY_PORT_RIGHT = 5017


def resolve_unity_target(
    hand, left_port=DEFAULT_UNITY_PORT_LEFT, right_port=DEFAULT_UNITY_PORT_RIGHT
):
    """Route packets by explicit label first, then by legacy hand_index."""
    label = str(hand.get("label", "")).strip().lower()
    if label == "right":
        return right_port, "Right"
    if label == "left":
        return left_port, "Left"

    hand_index = hand.get("hand_index")
    if hand_index == 1:
        return right_port, "Hand 1"
    if hand_index == 0:
        return left_port, "Hand 0"

    return left_port, "Unknown"
