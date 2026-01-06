import os
import pandas as pd


def get_sync_offset(root_dir, label=None, sync_label="Start", emg_fs=5000, video_fps=30, save_file=True):
    """
    Computes and optionally saves the sync offset between EMG and video systems based on a shared label.

    Parameters:
    - root_dir (str): Base directory containing `raw/`, `media/`, and `events/` folders.
    - label (str): Label to align on (default "Start").
    - emg_fs (int): EMG sampling rate (default 5000).
    - video_fps (int): Video frame rate (default 30).
    - save_file (bool): If True, saves `sync_offset.txt` under `offset/` in root_dir.
    - label_name (str): Optional name like "Dynamic5kHz" used to match the correct notes files.

    Returns:
    - offset (float): Time in seconds to add to the landmark/video timeline to align with EMG.

    """
    # Resolve file paths (Don't need to enforce this)
    #if label is None:
    #    raise ValueError("label_name must be provided to locate annotation files")

    emg_notes_path = os.path.join(root_dir, 'events', f'{label}_emg_events.txt' if label else 'emg_events.txt')
    video_notes_path = os.path.join(root_dir, 'events', f'{label}_video_events.txt' if label else 'video_events.txt')

    # Read both notes files
    video_notes = pd.read_csv(video_notes_path)
    emg_notes = pd.read_csv(emg_notes_path)

    # Find label rows
    if sync_label not in video_notes['Label'].values or sync_label not in emg_notes['Label'].values:
        raise ValueError(f"Label '{sync_label}' not found in both notes files.")

    video_row = video_notes[video_notes['Label'] == sync_label].iloc[0]
    emg_row = emg_notes[emg_notes['Label'] == sync_label].iloc[0]

    # Calculate times
    time_video = video_row['Sample Index'] / video_fps
    time_emg = emg_row['Sample Index'] / emg_fs
    offset = time_emg - time_video  # EMG is ahead if positive

    print(f"[INFO] Video label time: {time_video:.3f}s | EMG label time: {time_emg:.3f}s")
    print(f"[INFO] Computed sync offset: {offset:.3f} seconds")

    if save_file:
        offset_dir = os.path.join(root_dir, 'events')
        os.makedirs(offset_dir, exist_ok=True)
        offset_file = os.path.join(offset_dir, f'{label}_sync_offset.txt' if label else 'sync_offset.txt')
        with open(offset_file, 'w') as f:
            f.write(f"Offset to align EMG to video: {offset:.3f} s\n")
            f.write(f"Video Sync label time: {time_video:.3f} s\n")
            f.write(f"EMG Sync label time: {time_emg:.3f} s\n")
        print(f"[INFO] Offset saved to {offset_file}")

    return offset


# Alias for backward compatibility
calculate_sync_offset = get_sync_offset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Calculate synchronization offset between EMG and video timestamps."
    )
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root directory containing events/ folder")
    parser.add_argument("--label", type=str, default=None,
                        help="Session label (e.g., 'MySession')")
    parser.add_argument("--sync_label", type=str, default="Start",
                        help="Event label to sync on (default: 'Start')")
    parser.add_argument("--emg_fs", type=int, default=5000,
                        help="EMG sampling rate in Hz (default: 5000)")
    parser.add_argument("--video_fps", type=int, default=30,
                        help="Video frame rate (default: 30)")
    parser.add_argument("--no_save", action="store_true",
                        help="Don't save offset to file")
    
    args = parser.parse_args()
    
    offset = get_sync_offset(
        root_dir=args.root_dir,
        label=args.label,
        sync_label=args.sync_label,
        emg_fs=args.emg_fs,
        video_fps=args.video_fps,
        save_file=not args.no_save
    )
    
    print(f"\nSync offset: {offset:.4f} seconds")
