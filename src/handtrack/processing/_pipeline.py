# import os
# import numpy as np
# from ._preprocessing import EMGPreprocessor
# from ._joint_angles import compute_finger_angles  # if compute_finger_angles is defined elsewhere
# from ._session_loader import SessionLoader
# from ._file_utils import align_landmarks_to_emg)
# from handtrack.tracker import HandTracker
# from handtrack.ml import ModelManager
#
# def run_pipeline(root_dir, label, window_ms=250, step_ms=50, train_model=False, visualize=False, save_video=False, verbose=False):
#     # Load session data
#     loader = SessionLoader(root_dir, label, verbose)
#     emg, emg_fs, emg_t = loader.load_emg()
#     sync_offset = loader.load_sync_offset()
#     start_idx, end_idx = loader.get_event_indices()
#
#     # Check for landmark file
#     landmark_file = loader.get_landmark_path()
#     if not os.path.exists(landmark_file):
#         if verbose:
#             print(f"[INFO] Landmark file not found. Attempting to extract landmarks from video.")
#         video_path = loader.get_video_path()
#         tracker = HandTracker(source=video_path, apply_kalman=True, verbose=verbose)
#         landmarks, metadata = tracker.extract_landmarks(visualize=visualize, save_video=save_video)
#         lm_fs = metadata['sampling_rate']
#         lm_t = np.arange(landmarks.shape[0]) / lm_fs  # if not already computed
#     else:
#         landmarks, lm_fs, lm_t = loader.load_landmarks()
#
#     # Preprocess EMG
#     preprocessor = EMGPreprocessor(emg_fs, verbose=verbose)
#     emg_filtered = preprocessor.preprocess(emg)
#     emg_features = preprocessor.extract_features(emg_filtered, window_ms, step_ms)
#
#     # Align landmarks to EMG time
#     lm_interp = align_landmarks_to_emg(emg_t, lm_t, landmarks, sync_offset)
#
#     # Compute finger joint angles
#     if verbose:
#         print(f"Aligning landmarks to EMG time vector with sync offset {sync_offset}...")
#     start_idx = int(start_idx * emg_fs / 1000) if start_idx is not None else 0
#     end_idx = int(end_idx * emg_fs / 1000) if end_idx is not None else emg.shape[1]
#     lm_interp = lm_interp[start_idx:end_idx]
#     if verbose:
#         print(f"Landmark interpolation shape: {lm_interp.shape}")
#
#     # Compute angles for each window
#     if verbose:
#         print(f"Computing finger angles ...")
#     landmark_labels = []
#     window_size = int(window_ms * emg_fs / 1000)
#     step_size = int(step_ms * emg_fs / 1000)
#     for i in range(start_idx, emg.shape[1] - window_size + 1, step_size):
#         lm_avg = np.mean(lm_interp[i:i+window_size], axis=0)
#         landmark_labels.append(compute_finger_angles(lm_avg))
#     landmark_labels = np.array(landmark_labels)
#
#     # Truncate EMG features to match label count (if needed)
#     emg_features = emg_features[:len(landmark_labels)]
#
#     # Use model manager
#     manager = ModelManager(root_dir, label, input_dim=emg_features.shape[1], output_dim=landmark_labels.shape[1], verbose=verbose)
#     if train_model:
#         model, scaler = manager.train(emg_features, landmark_labels)
#     else:
#         model, scaler = manager.load()
#
#     # Predict
#     X_scaled = scaler.transform(emg_features)
#     X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
#     model.eval()
#     with torch.no_grad():
#         predictions = model(X_tensor).numpy()
#
#     return emg_features, landmark_labels, predictions, manager.eval_metrics
