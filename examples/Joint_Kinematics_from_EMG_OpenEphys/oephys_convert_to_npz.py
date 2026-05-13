"""
Convert Open Ephys .oebin files to NPZ format for faster loading.

This utility script converts Open Ephys Binary Format recordings to a simplified
NPZ format that can be loaded much faster in subsequent processing steps.

The output NPZ file contains:
    - amplifier_data: EMG data array (channels, samples)
    - sample_rate: Sampling frequency (Hz)
    - t_amplifier: Time vector (seconds)
    - channel_names: List of channel labels

Usage:
    python oephys_convert_to_npz.py --oebin_path /path/to/structure.oebin --output_path /path/to/output.npz

Author: NML (Neuro-Mechatronics Lab)
Created: 2026-02-16
"""

import argparse
from pathlib import Path

import numpy as np

try:
    from pyoephys.io import load_open_ephys_session
except ImportError:
    raise ImportError(
        "python-open-ephys is required. Install with:\n"
        "pip install --index-url https://test.pypi.org/simple/ --no-deps python-oephys"
    )


def convert_oebin_to_npz(
    oebin_path: str, output_path: str = None, verbose: bool = True
):
    """
    Convert .oebin file to NPZ format.

    Parameters
    ----------
    oebin_path : str
        Path to structure.oebin file or folder containing it
    output_path : str, optional
        Output NPZ file path. If None, creates {oebin_folder}_emg_data.npz
    verbose : bool
        Enable verbose output

    Returns
    -------
    output_path : str
        Path to created NPZ file
    """
    oebin_path = Path(oebin_path)

    if not oebin_path.exists():
        raise FileNotFoundError(f"Path does not exist: {oebin_path}")

    # Determine output path
    if output_path is None:
        if oebin_path.is_file():
            # Use parent folder name
            folder_name = oebin_path.parent.parent.name  # Go up to recording folder
            output_path = oebin_path.parent.parent / f"{folder_name}_emg_data.npz"
        else:
            # Use folder name
            folder_name = oebin_path.name
            output_path = oebin_path / f"{folder_name}_emg_data.npz"
    else:
        output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load .oebin data
    if verbose:
        print(f"Loading Open Ephys data from: {oebin_path}")

    try:
        session = load_open_ephys_session(str(oebin_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load Open Ephys data: {e}")

    # Extract data
    emg_data = session["amplifier_data"]
    sample_rate = session["sample_rate"]
    t_amplifier = session["t_amplifier"]
    channel_names = session.get(
        "channel_names", [f"ch{i}" for i in range(emg_data.shape[0])]
    )

    if verbose:
        print("\nData summary:")
        print(f"  Channels: {emg_data.shape[0]}")
        print(f"  Samples: {emg_data.shape[1]}")
        print(f"  Sampling rate: {sample_rate} Hz")
        print(f"  Duration: {t_amplifier[-1] - t_amplifier[0]:.2f} seconds")
        print(f"  Data type: {emg_data.dtype}")
        print(f"  Memory: {emg_data.nbytes / 1024**2:.1f} MB")

    # Save to NPZ
    if verbose:
        print(f"\nSaving to: {output_path}")

    np.savez(
        output_path,
        amplifier_data=emg_data,
        sample_rate=sample_rate,
        t_amplifier=t_amplifier,
        channel_names=channel_names,
        emg_data=emg_data,  # Alias for compatibility
        time_vector=t_amplifier,  # Alias for compatibility
        sampling_rate=sample_rate,  # Alias for compatibility
    )

    # Verify file size
    file_size = output_path.stat().st_size / 1024**2

    if verbose:
        print("\n✓ Conversion complete!")
        print(f"  Output file: {output_path}")
        print(f"  File size: {file_size:.1f} MB")
        print("\nThis file can now be used with OEphysSessionLoader by setting:")
        print(f"  oebin_path='{output_path}'")

    return str(output_path)


def batch_convert(
    input_dir: str, pattern: str = "**/structure.oebin", verbose: bool = True
):
    """
    Batch convert all .oebin files in a directory tree.

    Parameters
    ----------
    input_dir : str
        Root directory to search
    pattern : str
        Glob pattern to match .oebin files
    verbose : bool
        Enable verbose output

    Returns
    -------
    converted_files : list of str
        List of created NPZ file paths
    """
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {input_dir}")

    # Find all .oebin files
    oebin_files = list(input_dir.glob(pattern))

    if not oebin_files:
        print(f"No .oebin files found matching pattern: {pattern}")
        return []

    print(f"Found {len(oebin_files)} .oebin files")
    print("=" * 60)

    converted = []
    for i, oebin_path in enumerate(oebin_files, 1):
        print(f"\n[{i}/{len(oebin_files)}] Converting: {oebin_path.parent.name}")
        try:
            output_path = convert_oebin_to_npz(oebin_path, verbose=verbose)
            converted.append(output_path)
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print("\n" + "=" * 60)
    print(f"Batch conversion complete: {len(converted)}/{len(oebin_files)} successful")

    return converted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Open Ephys .oebin files to NPZ format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--oebin_path",
        type=str,
        default=None,
        help="Path to structure.oebin file or folder containing it",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output NPZ file path (default: auto-generate from oebin path)",
    )

    # Batch mode
    parser.add_argument(
        "--batch_dir",
        type=str,
        default=None,
        help="Directory for batch conversion (searches recursively for .oebin files)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/structure.oebin",
        help="Glob pattern for batch mode",
    )

    parser.add_argument(
        "--verbose", action="store_true", default=True, help="Enable verbose output"
    )
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output")

    args = parser.parse_args()

    verbose = args.verbose and not args.quiet

    # Check mode
    if args.batch_dir:
        # Batch mode
        batch_convert(args.batch_dir, pattern=args.pattern, verbose=verbose)
    elif args.oebin_path:
        # Single file mode
        convert_oebin_to_npz(args.oebin_path, args.output_path, verbose=verbose)
    else:
        parser.error("Either --oebin_path or --batch_dir must be specified")
