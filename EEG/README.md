# CERELOG LSL

This repository includes `cerelog_lsl.py`, a Python script for publishing CERELOG data as a Lab Streaming Layer (LSL) stream. LSL-compatible applications can discover and record the stream over the network.

## Requirements

- Python 3
- The device, drivers, and Python dependencies required by the script

Check the imports in `cerelog_lsl.py` to identify required packages. Install them in the Python environment that will run the script.

## Usage

1. Connect to the CERELOG-EEG Board using a usb-c cable
2. Confirm Power(Green Light is ON)
3. From the directory containing the script, run:

	```bash
	python cerelog_lsl.py
	```
4. Open the OPENBCI GUI to stream data from the Board - 
4. Use LSL-compatible LabRecorder to record the published stream.

## Links

1. To stream from the cerelog board, download the OPENBCI GUI Dependency:


## Troubleshooting

- If no stream is detected, verify the script is running and check device connectivity and network/firewall settings.
- For an import error, install the missing dependency into the active Python environment.
- If channel data looks incorrect, verify the hardware setup and channel configuration in the script.

Validate recorded data before using it for analysis.
