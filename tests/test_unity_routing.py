from unity_hand_tracking._routing import (
    DEFAULT_UNITY_PORT_LEFT,
    DEFAULT_UNITY_PORT_RIGHT,
    resolve_unity_target,
)


def test_label_takes_priority_over_hand_index():
    port, hand_name = resolve_unity_target({"label": "Right", "hand_index": 0})
    assert port == DEFAULT_UNITY_PORT_RIGHT
    assert hand_name == "Right"


def test_hand_index_routes_legacy_unlabeled_packets():
    port, hand_name = resolve_unity_target({"hand_index": 1})
    assert port == DEFAULT_UNITY_PORT_RIGHT
    assert hand_name == "Hand 1"


def test_unknown_packets_fall_back_to_left_port():
    port, hand_name = resolve_unity_target({})
    assert port == DEFAULT_UNITY_PORT_LEFT
    assert hand_name == "Unknown"