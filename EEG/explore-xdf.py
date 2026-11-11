#!/usr/bin/env python
"""
XDF Explorer and CSV Exporter
=============================

A simple tool to:
1. View the contents of XDF files
2. Export streams to CSV files

Requirements:
- Python 3.6+
- pyxdf (install with: pip install pyxdf)
- pandas (install with: pip install pandas)
- matplotlib (install with: pip install matplotlib)

Instructions:
1. Place this script in the same folder as your XDF file(s)
2. Double-click on the script to run it (or run from command line)
3. Follow the on-screen prompts
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    import pyxdf
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyxdf", "pandas", "matplotlib"])
    import pyxdf

class XDFExplorer:
    def __init__(self, root):
        self.root = root
        self.root.title("XDF Explorer")
        self.root.geometry("800x600")
        
        self.xdf_data = None
        self.file_path = None
        
        # Create the main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add buttons
        self.btn_frame = tk.Frame(self.main_frame)
        self.btn_frame.pack(fill=tk.X, pady=10)
        
        self.load_btn = tk.Button(self.btn_frame, text="Load XDF File", command=self.load_xdf)
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        self.view_btn = tk.Button(self.btn_frame, text="View Stream Info", command=self.view_stream_info, state=tk.DISABLED)
        self.view_btn.pack(side=tk.LEFT, padx=5)
        
        self.plot_btn = tk.Button(self.btn_frame, text="Plot Stream", command=self.plot_stream, state=tk.DISABLED)
        self.plot_btn.pack(side=tk.LEFT, padx=5)
        
        self.export_btn = tk.Button(self.btn_frame, text="Export to CSV", command=self.export_to_csv, state=tk.DISABLED)
        self.export_btn.pack(side=tk.LEFT, padx=5)
        
        # Add text area for displaying information
        self.info_frame = tk.Frame(self.main_frame)
        self.info_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.info_label = tk.Label(self.info_frame, text="Stream Information:")
        self.info_label.pack(anchor=tk.W)
        
        self.text_area = tk.Text(self.info_frame, wrap=tk.WORD)
        self.text_area.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbar to text area
        self.scrollbar = tk.Scrollbar(self.text_area)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_area.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.text_area.yview)
        
        # Add status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Please load an XDF file.")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Plot frame (hidden initially)
        self.plot_frame = None
        self.canvas = None

    def load_xdf(self):
        """Load an XDF file using a file dialog."""
        self.file_path = filedialog.askopenfilename(
            title="Select XDF File",
            filetypes=[("XDF files", "*.xdf"), ("All files", "*.*")]
        )
        
        if not self.file_path:
            return
        
        try:
            self.status_var.set(f"Loading {os.path.basename(self.file_path)}...")
            self.root.update()
            
            self.xdf_data = pyxdf.load_xdf(self.file_path)
            
            self.status_var.set(f"Loaded {os.path.basename(self.file_path)} successfully.")
            self.view_btn.config(state=tk.NORMAL)
            self.plot_btn.config(state=tk.NORMAL)
            self.export_btn.config(state=tk.NORMAL)
            
            # Automatically show stream info after loading
            self.view_stream_info()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load XDF file: {str(e)}")
            self.status_var.set("Failed to load XDF file.")

    def view_stream_info(self):
        """Display information about the streams in the XDF file."""
        if not self.xdf_data:
            return
        
        self.text_area.delete(1.0, tk.END)
        
        # Clear any existing plots
        if self.plot_frame:
            self.plot_frame.pack_forget()
            self.plot_frame = None
        
        self.text_area.insert(tk.END, f"File: {os.path.basename(self.file_path)}\n\n")
        
        for i, stream in enumerate(self.xdf_data[0]):
            self.text_area.insert(tk.END, f"Stream {i}: {stream['info']['name'][0]}\n")
            self.text_area.insert(tk.END, f"  Type: {stream['info']['type'][0]}\n")
            self.text_area.insert(tk.END, f"  Channel count: {stream['info']['channel_count'][0]}\n")
            self.text_area.insert(tk.END, f"  Sample rate: {stream['info']['nominal_srate'][0]} Hz\n")
            self.text_area.insert(tk.END, f"  Data points: {len(stream['time_series'])}\n")
            
            # If there are channels with names, show them
            if 'desc' in stream['info'] and stream['info']['desc'][0] is not None and 'channels' in stream['info']['desc'][0]:
                channels = stream['info']['desc'][0]['channels'][0]['channel']
                self.text_area.insert(tk.END, "  Channels:\n")
                for j, channel in enumerate(channels):
                    if 'label' in channel:
                        self.text_area.insert(tk.END, f"    {j}: {channel['label'][0]}\n")
            
            # Show the first few data points
            if len(stream['time_series']) > 0:
                self.text_area.insert(tk.END, "  First few data points:\n")
                max_show = min(5, len(stream['time_series']))
                for j in range(max_show):
                    data_point = stream['time_series'][j]
                    if len(data_point) > 10:  # If too many channels, show only first few
                        data_str = str(data_point[:5]) + " ... " + str(data_point[-5:])
                    else:
                        data_str = str(data_point)
                    self.text_area.insert(tk.END, f"    {j}: {data_str}\n")
                
            self.text_area.insert(tk.END, "\n")

    def plot_stream(self):
        """Plot data from a selected stream."""
        if not self.xdf_data:
            return
        
        stream_options = []
        for i, stream in enumerate(self.xdf_data[0]):
            name = stream['info']['name'][0]
            type_ = stream['info']['type'][0]
            stream_options.append(f"{i}: {name} ({type_})")
        
        if not stream_options:
            messagebox.showinfo("Info", "No streams available to plot.")
            return
        
        # Ask user which stream to plot
        selected = simpledialog.askstring(
            "Select Stream", 
            "Enter the stream number to plot:",
            initialvalue="0"
        )
        
        if selected is None:
            return
        
        try:
            selected_idx = int(selected)
            if selected_idx < 0 or selected_idx >= len(self.xdf_data[0]):
                raise ValueError("Stream index out of range")
                
            stream = self.xdf_data[0][selected_idx]
            name = stream['info']['name'][0]
            
            # Clear text area and any existing plots
            self.text_area.delete(1.0, tk.END)
            if self.plot_frame:
                self.plot_frame.pack_forget()
            
            # Create plot frame
            self.plot_frame = tk.Frame(self.info_frame)
            self.plot_frame.pack(fill=tk.BOTH, expand=True)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # Get data and timestamps
            data = stream['time_series']
            timestamps = stream['time_stamps']
            
            # Determine what to plot based on the data shape
            if len(data) == 0:
                messagebox.showinfo("Info", "No data points in this stream.")
                return
                
            # If there are too many channels, ask which ones to plot
            num_channels = data[0].size if hasattr(data[0], 'size') else 1
            
            if num_channels > 10:
                selected_channels = simpledialog.askstring(
                    "Select Channels", 
                    f"This stream has {num_channels} channels. Enter channel numbers to plot (comma-separated, max 5):",
                    initialvalue="0,1,2,3,4"
                )
                
                if selected_channels is None:
                    return
                    
                try:
                    channel_indices = [int(i.strip()) for i in selected_channels.split(',')]
                    channel_indices = [i for i in channel_indices if 0 <= i < num_channels][:5]  # Limit to 5 channels
                except:
                    channel_indices = list(range(min(5, num_channels)))
            else:
                channel_indices = list(range(min(5, num_channels)))
            
            # Prepare data for plotting
            if num_channels == 1:
                # Single channel data
                ax.plot(timestamps, data, label=name)
            else:
                # Multi-channel data
                y = []
                labels = []
                
                # Try to get channel names if available
                channel_names = []
                if (
                    'desc' in stream['info']
                    and stream['info']['desc']
                    and stream['info']['desc'][0] is not None
                    and 'channels' in stream['info']['desc'][0]
                ):
                    channels = stream['info']['desc'][0]['channels'][0]['channel']
                    for i in channel_indices:
                        if i < len(channels) and 'label' in channels[i]:
                            channel_names.append(channels[i]['label'][0])
                        else:
                            channel_names.append(f"Channel {i}")
                else:
                    channel_names = [f"Channel {i}" for i in channel_indices]
                
                # Extract data for each selected channel
                data_array = []
                for i in range(len(timestamps)):
                    if i < len(data):
                        if hasattr(data[i], '__len__') and len(data[i]) > 1:
                            row = []
                            for j in channel_indices:
                                if j < len(data[i]):
                                    row.append(data[i][j])
                            data_array.append(row)
                        else:
                            # Single value per timestamp
                            data_array.append([data[i]])
                
                # Convert to numpy array for easier slicing
                import numpy as np
                data_array = np.array(data_array)
                
                # Plot each channel
                for j, idx in enumerate(channel_indices):
                    if idx < data_array.shape[1]:
                        ax.plot(timestamps, data_array[:, j], label=channel_names[j])
            
            ax.set_title(f"Stream: {name}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True)
            
            # Add the plot to the GUI
            self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot stream: {str(e)}")
            self.status_var.set("Failed to plot stream.")

    def export_to_csv(self):
        """Export a selected stream to a CSV file."""
        if not self.xdf_data:
            return
            
        stream_options = []
        for i, stream in enumerate(self.xdf_data[0]):
            name = stream['info']['name'][0]
            type_ = stream['info']['type'][0]
            stream_options.append(f"{i}: {name} ({type_})")
        
        if not stream_options:
            messagebox.showinfo("Info", "No streams available to export.")
            return
        
        # Ask user which stream to export
        selected = simpledialog.askstring(
            "Select Stream", 
            "Enter the stream number to export to CSV:",
            initialvalue="0"
        )
        
        if selected is None:
            return
            
        try:
            selected_idx = int(selected)
            if selected_idx < 0 or selected_idx >= len(self.xdf_data[0]):
                raise ValueError("Stream index out of range")
                
            stream = self.xdf_data[0][selected_idx]
            name = stream['info']['name'][0]
            
            # Get data and timestamps
            data = stream['time_series']
            timestamps = stream['time_stamps']
            
            if len(data) == 0:
                messagebox.showinfo("Info", "No data points in this stream.")
                return
                
            # Ask for output file path
            default_filename = f"{os.path.splitext(os.path.basename(self.file_path))[0]}_{name}.csv"
            output_path = filedialog.asksaveasfilename(
                title="Save CSV File",
                defaultextension=".csv",
                initialfile=default_filename,
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if not output_path:
                return
                
            # Prepare data for CSV export
            self.status_var.set(f"Exporting {name} to CSV...")
            self.root.update()
            
            # Try to get channel names if available
            channel_names = []
            if 'desc' in stream['info'] and stream['info']['desc'][0] is not None and 'channels' in stream['info']['desc'][0]:
                channels = stream['info']['desc'][0]['channels'][0]['channel']
                for i in range(len(channels)):
                    if 'label' in channels[i]:
                        channel_names.append(channels[i]['label'][0])
                    else:
                        channel_names.append(f"Channel_{i}")
            
            # Determine number of channels
            num_channels = 1
            if len(data) > 0:
                if hasattr(data[0], 'size'):
                    num_channels = data[0].size
                elif hasattr(data[0], '__len__'):
                    num_channels = len(data[0])
            
            # If no channel names, create default ones
            if not channel_names:
                channel_names = [f"Channel_{i}" for i in range(num_channels)]
            
            # Create DataFrame
            df_data = {'Timestamp': timestamps}
            
            # Handle different data structures
            if num_channels == 1:
                if hasattr(data[0], '__len__'):
                    data = data[0]
                else:
                    df_data[channel_names[0]] = data
            else:
                for i in range(min(len(channel_names), num_channels)):
                    channel_values = []
                    for j in range(len(data)):
                        if j < len(data) and i < len(data[j]):
                            channel_values.append(data[j][i])
                        else:
                            channel_values.append(None)
                    df_data[channel_names[i]] = channel_values
            
            # Create and save DataFrame
            df = pd.DataFrame(df_data)
            df.to_csv(output_path, index=False)
            
            self.status_var.set(f"Exported {name} to {os.path.basename(output_path)} successfully.")
            messagebox.showinfo("Export Complete", f"Stream exported to {output_path} successfully.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export stream: {str(e)}")
            self.status_var.set("Failed to export stream.")

def main():
    """Main entry point for the application."""
    root = tk.Tk()
    app = XDFExplorer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
