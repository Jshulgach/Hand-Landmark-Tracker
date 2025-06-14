#import os
#import PyQt5
#from PyQt5.QtWidgets import QApplication, QFileDialog
import tkinter as tk
from tkinter import filedialog


def check_video_path(video_path=None):
    """
    Get video file path from user input or file dialog. Checks if the path is valid.

    Parameters
    ----------
    video_path : str, optional
        Path to the video file. If not provided, a file dialog will be opened.

    Returns
    -------
    str
        Valid path to the video file.

    """
    # If no path was given, show file dialog
    if not video_path:
        #root = tk.Tk()
        #root.withdraw()  # Hide the main window
        video_path = filedialog.askopenfilename()
        if not video_path:
            print("No file selected, returning.")
            return None
    return video_path


def load_video(video_path=None):
    pass
