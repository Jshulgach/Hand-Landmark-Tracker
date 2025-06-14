import pandas as pd

VIDEO_NOTES_PATH = '../../data/video_notes.txt'
EMG_NOTES_PATH = '../../data/emg_notes.txt'
LABEL = 'ThumbsUp'
VIDEO_FPS = 30
EMG_FS = 5000

# Read annotation files
video_notes = pd.read_csv(VIDEO_NOTES_PATH)
emg_notes = pd.read_csv(EMG_NOTES_PATH)

# Find the label in both files
video = video_notes[video_notes['Label'] == LABEL].iloc[0]
emg = emg_notes[emg_notes['Label'] == LABEL].iloc[0]

# Time of label in video (assuming sample index is frame number)
time_video = video['Sample Index'] / VIDEO_FPS
# Time of label in EMG
time_emg = emg['Sample Index'] / EMG_FS

# Calculate offset (time difference)
offset = time_video - time_emg

print(f"Video Sync label time: {time_video:.3f} s")
print(f"EMG Sync label time: {time_emg:.3f} s")
print(f"Offset to align EMG to video: {offset:.3f} s")

# Save the offset to a file
with open('../../data/sync_offset.txt', 'w') as f:
    f.write(f"Offset to align EMG to video: {offset:.3f} s\n")
    f.write(f"Video Sync label time: {time_video:.3f} s\n")
    f.write(f"EMG Sync label time: {time_emg:.3f} s\n")