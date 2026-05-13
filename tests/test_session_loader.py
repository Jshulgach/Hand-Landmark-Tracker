import numpy as np
from handtrack.io import SessionLoader


def test_load_landmarks_from_session_root(tmp_path):
    landmarks = np.zeros((3, 21, 3), dtype=np.float32)
    time_vector = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    np.savez(
        tmp_path / "landmarks.npz",
        landmarks=landmarks,
        sampling_rate=10,
        time_vector=time_vector,
    )

    loader = SessionLoader(str(tmp_path), label="")

    loaded_landmarks, sampling_rate, loaded_time_vector = loader.load_landmarks()

    assert loaded_landmarks.shape == (3, 21, 3)
    assert sampling_rate == 10
    assert np.array_equal(loaded_time_vector, time_vector)
