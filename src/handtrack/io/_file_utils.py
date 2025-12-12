import os
from tkinter import filedialog
import pandas as pd
import yaml


def load_txt_config(file_path=None, verbose=False):
    """
    Parse a simple key=value style configuration file (e.g. config.txt).

    Parameters:
        file_path (str): Path to the config file.
        verbose (bool): If True, print warnings and info messages.

    Returns:
        dict: Dictionary of key-value settings.

    """

    if file_path is None:
        file_path = filedialog.askopenfilename(title="Select Notes File", filetypes=[("Text files", "*.txt")])
        if not file_path:
            if verbose:
                print("Cancelled selection")
            return None

    # Dictionary to store the key-value pairs
    config_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace and ignore empty lines or comments
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Split the line into key and value at the first '='
            key, value = line.split('=', 1)
            config_data[key.strip()] = value.strip()
    return config_data


def load_yaml_config(file_path=None, verbose=False):
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML file.

    Returns:
        dict: Parsed config dictionary.
    """
    if file_path is None:
        file_path = filedialog.askopenfilename(title="Select Notes File", filetypes=[("Text files", "*.txt")])
        if not file_path:
            if verbose:
                print("Cancelled selection")
            return None

    if not os.path.exists(file_path):
        print(f"Config file not found: {file_path}")
        return None

    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def update_yaml_config(config_path, updates):
    """
    Update a YAML config file with new values.

    Args:
        config_path (str): Path to the config file.
        updates (dict): Dictionary of values to update.
    """
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}

    # Recursively update
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    config = deep_update(config, updates)

    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)


def get_sync_offset(root_dir, label=None, sync_label="start", emg_fs=5000, video_fps=30, save_file=True, verbose=False):
    """
    Computes and optionally saves the sync offset between EMG and video systems based on a shared label.

    Parameters:
    - root_dir (str): Base directory containing `raw/`, `media/`, and `events/` folders.
    - label (str): Label to align on (default "start").
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

    # Extract start and end indices from the event files
    indices = extract_event_indices(os.path.join(
        root_dir, 'events'),
        label=label,
        verbose=verbose
    )
    if not indices['start_idx'] or not indices['end_idx']:
        print(f"| [WARNING]  No valid start/end indices found for session: {label}")
        return None, None

    # The indices variable is a dictionary with
    #   'file_path': ['path\to\file.events',path\to\file.events'],
    #   'source_type': ['EMG', 'VIDEO'],
    #   'start_idx': [123, 456],
    #   'end_idx': [None, None]
    #
    if verbose:
        print(f"[INFO] Found {len(indices['file_path'])} event files matching label '{label}'")
        print(f"|  Source types: {indices['source_type']}")
        print(f"|  Start indices: {indices['start_idx']}")
        print(f"|  End indices: {indices['end_idx']}")
    if len(indices['start_idx']) < 2:
        print(f"[WARNING] Only one source type found for label '{label}'. Cannot compute sync offset.")
        return None
    # Find the video and EMG rows
    video_row = None
    emg_row = None
    for i, source_type in enumerate(indices['source_type']):
        if source_type.lower() == 'video':
            video_row = {'Sync Index': indices['start_idx'][i], 'File Path': indices['file_path'][i]}
        elif source_type.lower() == 'emg':
            emg_row = {'Sync Index': indices['start_idx'][i], 'File Path': indices['file_path'][i]}

    # Calculate times
    time_video = video_row['Sync Index'] / video_fps
    time_emg = emg_row['Sync Index'] / emg_fs
    offset = time_emg - time_video  # EMG is ahead if positive

    print(f"|  |__ Video label time: {time_video:.3f}s | EMG label time: {time_emg:.3f}s")
    print(f"|  |__ Computed sync offset: {offset:.3f} seconds")

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


def extract_event_indices(root_dir, label=None, verbose=False):
    """
    Check and parse events files for a specific label to determine start and end indices.

    Parameters:
        root_dir (str): Directory containing the event files (e.g. root/events).
        label (str): Required label prefix (e.g. "Dynamic1kHz").
        verbose (bool): If True, print debug info.

    Returns:
        dict: Dictionary with fields: file_path, source_type, start_idx, end_idx, and lists for each

    """
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Directory {root_dir} does not exist.")

    event_dict = {
        'file_path': [],
        'source_type': [],
        'start_idx': [],
        'end_idx': []
    }

    source_types = ['emg', 'video', 'landmark']
    all_files = [f for f in os.listdir(root_dir) if f.endswith('.events')]

    # Filter by label if provided
    if label:
        filtered_files = [f for f in all_files if label.lower() in f.lower()]
    else:
        filtered_files = all_files

    if verbose:
        print(f"|  Total event files found: {len(all_files)}")
        print(f"|  Event files matching label '{label}': {len(filtered_files)}")

    if not filtered_files:
        if verbose:
            print("|  [WARNING] No matching events files found.")
        return event_dict

    for file in filtered_files:
        full_path = os.path.join(root_dir, file)
        event_dict['file_path'].append(full_path)

        # Determine source type from filename
        source_type = next((st.upper() for st in source_types if st in file.lower()), None)
        if source_type is None and verbose:
            print(f"|  [WARNING] Unknown source type in filename '{file}'")
        event_dict['source_type'].append(source_type)

        if verbose:
            print(f"|  Parsing {source_type or 'UNKNOWN'} events file: {file}")

        s_idx, e_idx = parse_events_file(full_path)
        print(f"|  |__ Received 'Start' index: {s_idx}, 'End' index: {e_idx}")
        event_dict['start_idx'].append(s_idx)
        event_dict['end_idx'].append(e_idx)

    # Fallback in case no valid start/end was found
    if all(v is None for v in event_dict['start_idx']):
        if verbose:
            print("|  [WARNING] No valid event indices found. Using fallback [0, -1].")
        event_dict['start_idx'].append(0)
        event_dict['end_idx'].append(-1)

    return event_dict


def contains_header(file):
    """ Helper function to see if the first line in teh file contains labels or strings instead of numbers
    """
    try:
        with open(file, 'r') as f:
            first_line = f.readline().strip()
            # If the first line is empty or starts with a non-digit character, assume it's a header
            return not first_line or not first_line[0].isdigit()
    except Exception as e:
        print(f"[ERROR] Failed to read file {file}: {e}")
        return False


def parse_events_file(file_path):
    """
    Extract Start and End indices from a single event file, assumes .txt ending
    and a specific format with 'Start' and 'End' labels.

    Parameters:
        file_path (str): Path to the event file.

    Returns:
        start_idx (int): Start index for the EMG data.
        end_idx (int): End index for the EMG data.

    """
    try:
        with open(file_path, 'r') as f:
            start_idx, end_idx = None, None
            # Read each line and extract indices
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 3:
                    continue  # Skip malformed lines
                sample_index_str, _, label = parts
                label = label.strip().lower()
                if label == 'start' and start_idx is None:
                    start_idx = int(sample_index_str)
                elif label == 'end' and end_idx is None:
                    end_idx = int(sample_index_str)

            return start_idx, end_idx

    except Exception as e:
        print(f"[ERROR] Failed to parse events file {file_path}: {e}")
        return None, None


def load_feature_dataset(root_dir, label=None, config_file=None, verbose=False):
    """
    Load the feature dataset for EMG and joint angles.

    Parameters:
        root_dir (str): Root directory containing the dataset.
        label (str): Optional label to filter the dataset.
        config_file (dict): Optional configuration dictionary.
        verbose (bool): If True, print additional information.

    Returns:
        dict: Dictionary containing the loaded dataset.
    """

    if config_file:
        data_path = config_file.get('FEATURE_DATASET_PATH', None)
        if not data_path:
            data_path = os.path.join(root_dir, f"{label}_feature_dataset.npz" if label else "feature_dataset.npz")
    else:
        data_path = os.path.join(root_dir, f"{label}_feature_dataset.npz" if label else "feature_dataset.npz")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at {data_path}")

    if verbose:
        print(f"[INFO] Loading feature dataset from {data_path}")

    return np.load(data_path)
