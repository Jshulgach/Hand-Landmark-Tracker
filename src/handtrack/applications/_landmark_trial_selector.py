"""
handtrack.applications._landmark_trial_selector

Graphical tool for interactively labeling trial events on landmark position data.

This GUI allows researchers to:
- Load and visualize landmark positions from .npy or .npz files
- Select a landmark index from a dropdown
- Click to mark trial onset points
- Save marked events to a text file for synchronizing EMG or other data
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, Scrollbar, VERTICAL
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class LandmarkSelector:
    """
    Tkinter-based application for manual EMG trial indexing.

    Attributes:
        emg_data (np.ndarray): EMG signal matrix (channels × samples)
        time_vector (np.ndarray): Time vector aligned with EMG samples
        sampling_rate (float): Sampling rate of amplifier
        current_channel (int): Channel index currently displayed
        indexing_enabled (bool): If True, allows user to click to insert marker
    """
    def __init__(self, root, sample_rate=1000):
        self.root = root
        self.root.title("EMG Trial Selector")

        self.landmark_data = None
        self.landmark_labels = None
        self.time_vector = None
        self.sampling_rate = sample_rate
        self.current_landmark = 0
        self.indexing_enabled = False
        self.trial_indices = []

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- Top Controls Frame ---
        control_frame = tk.Frame(root)
        control_frame.pack(side="top", fill="x", pady=5)

        tk.Button(control_frame, text="Load Landmark File", command=self.load_file).pack(side="left", padx=5)
        tk.Button(control_frame, text="Set Trial Index", command=self.enable_indexing).pack(side="left", padx=5)

        # --- Main Frame (Canvas + Sidebar) ---
        main_frame = tk.Frame(root)
        main_frame.pack(side="top", fill="both", expand=True)

        # === Plot Area ===
        self.figure, self.ax = plt.subplots(figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=main_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side="left", fill="both", expand=True)

        # === Sidebar Frame ===
        sidebar_frame = tk.Frame(main_frame)
        sidebar_frame.pack(side="right", fill="y", padx=10)

        # --- Channel Selector ---
        tk.Label(sidebar_frame, text="Landmark Index:").pack(anchor="w")
        self.landmark_selector = ttk.Combobox(sidebar_frame, state="readonly")
        self.landmark_selector.bind("<<ComboboxSelected>>", self.update_landmark)
        self.landmark_selector.pack(fill="x", pady=5)

        # --- Label Entry Field ---
        tk.Label(sidebar_frame, text="Custom Label:").pack(anchor="w")
        self.label_entry = tk.Entry(sidebar_frame)
        self.label_entry.pack(fill="x", pady=5)
        self.label_entry.insert(0, "Label")  # Default text

        # --- Table ---
        self.table = ttk.Treeview(sidebar_frame, columns=("Sample Index", "Label"), show="headings", height=20)
        self.table.heading("Sample Index", text="Sample Index")
        self.table.heading("Label", text="Label")
        self.table.column("Sample Index", width=100)
        self.table.column("Label", width=100)
        self.table.pack(side="top", fill="y")

        scrollbar = Scrollbar(sidebar_frame, orient=VERTICAL, command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        # --- Save & Delete Buttons ---
        button_frame = tk.Frame(sidebar_frame)
        button_frame.pack(side="bottom", pady=10)
        tk.Button(button_frame, text="Save", command=self.save_table).pack(side="left", padx=5)
        tk.Button(button_frame, text="Delete", command=self.delete_selected).pack(side="left", padx=5)
        tk.Button(button_frame, text="Clear All", command=self.clear_all_labels).pack(side="left", padx=5)

        self.canvas.mpl_connect("button_press_event", self.on_click)

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("NumPy files", "*.npy *.npz"), ("All files", "*.*")])
        if not path:
            return

        if path.endswith('.npy'):
            self.landmark_data = np.load(path)
            #self.time_vector = np.arange(self.landmark_data.shape[0])
            self.landmark_labels = [f"Landmark {i}" for i in range(self.landmark_data.shape[1])]
        else:
            data = np.load(path)
            self.landmark_data = data['landmarks']  # Key might vary depending on save format
            self.sampling_rate = data.get('sampling_rate', 1000)  # Default to 1000 if not provided
            #self.time_vector = data['time_vector'] if 'time_vector' in data else np.arange(self.landmark_data.shape[0]) / self.sampling_rate
            self.landmark_labels = data.get('landmark_labels', [f"Landmark {i}" for i in range(self.landmark_data.shape[1])])

        if self.landmark_data.ndim != 3 or self.landmark_data.shape[2] != 3:
            messagebox.showerror("Error", "Invalid landmark data shape.")
            return

        n_landmarks = self.landmark_data.shape[1]

        # Get the time vector
        self.time_vector = np.arange(self.landmark_data.shape[0])
        #self.time_vector = np.arange(self.landmark_data.shape[0]) / self.sampling_rate  # Convert to seconds

        #self.landmark_selector['values'] = [f"Landmark {i}" for i in range(n_landmarks)]
        self.landmark_selector['values'] = [
            f"{i}: {self.landmark_labels[i]}" for i in range(n_landmarks)
        ]
        self.landmark_selector.current(0)
        self.current_landmark = 0
        self.plot_landmark()

    def sample_index_to_timestamp(self, index):
        """
        Convert a sample index to a timestamp string.

        Parameters:
            index (int): Sample index to convert.

        Returns:
            str: Formatted timestamp string (HH:MM:SS).
        """
        seconds = index / self.sampling_rate
        return str(datetime.timedelta(seconds=int(seconds)))

    def save_table(self):
        """
        Save the trial markers to a text file with sample index and timestamp.
        """
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")],
            title="Save Trial Markers"
        )
        if not path:
            return

        # Collect and sort table data by sample index
        rows = []
        for row in self.table.get_children():
            sample_index, label = self.table.item(row)["values"]
            sample_index = int(sample_index)
            timestamp = self.sample_index_to_timestamp(sample_index)
            rows.append((sample_index, timestamp, label))

        rows.sort(key=lambda x: x[0])  # Sort by sample index

        # Write to text file
        with open(path, "w") as f:
            f.write("Sample Index,Timestamp,Label\n")
            for sample_index, timestamp, label in rows:
                f.write(f"{sample_index},{timestamp},{label}\n")

        messagebox.showinfo("Saved", f"Trial markers saved to:\n{path}")

    def delete_selected(self):
        """
        Delete selected rows from the table.
        """
        selected = self.table.selection()
        for item in selected:
            self.table.delete(item)

        # Remove the vertical line associated with the deleted indices and update the plot
        self.trial_indices = [idx for idx in self.trial_indices if idx not in [int(self.table.item(item)["values"][0]) for item in selected]]
        self.plot_landmark()

    def clear_all_labels(self):
        """
        Clear all labels from the table and reset trial indices.
        """
        self.table.delete(*self.table.get_children())
        self.trial_indices.clear()
        self.ax.clear()
        self.canvas.draw()
        messagebox.showinfo("Cleared", "All labels cleared.")

    def update_landmark(self, event=None):
        if self.landmark_data is None:
            return
        self.current_landmark = self.landmark_selector.current()
        self.plot_landmark()

    def enable_indexing(self):
        self.indexing_enabled = True

    def on_click(self, event):
        """
        Handle mouse click events on the plot to mark trial onset points.

        Parameters:
            event (matplotlib.backend_bases.Event): The mouse event.
        """
        if not self.indexing_enabled or event.inaxes != self.ax:
            return

        frame_clicked = int(event.xdata)
        self.ax.axvline(x=frame_clicked, color='blue', linestyle='--')
        self.trial_indices.append(frame_clicked)
        self.canvas.draw()

        # Insert the clicked frame and label into the table
        label = self.label_entry.get()
        self.table.insert("", "end", values=(frame_clicked,label))
        self.indexing_enabled = False

    def plot_landmark(self):
        self.ax.clear()
        x = self.landmark_data[:, self.current_landmark, 0]
        y = self.landmark_data[:, self.current_landmark, 1]
        z = self.landmark_data[:, self.current_landmark, 2]
        self.ax.plot(self.time_vector, x, label='X')
        self.ax.plot(self.time_vector, y, label='Y')
        self.ax.plot(self.time_vector, z, label='Z')
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Normalized Position")
        self.ax.set_title(f"Landmark {self.current_landmark} - {self.landmark_labels[self.current_landmark]} Position")
        self.ax.legend()

        for idx in self.trial_indices:
            self.ax.axvline(x=idx, color='blue', linestyle='--')

        self.canvas.draw()

    def on_closing(self):
        self.root.quit()
        self.root.destroy()


def launch_landmark_selector(sampling_rate=1000):
    """
    Launch the EMG trial selector GUI.
    """
    root = tk.Tk()
    app = LandmarkSelector(root, sampling_rate)
    root.mainloop()
