# CERELOG LSL

This repository includes `cerelog_lsl.py`, a Python script for publishing CERELOG data as a Lab Streaming Layer (LSL) stream. LSL-compatible applications can discover and record the stream over the network.

## Requirements

- Python 3
- The device, drivers, and Python dependencies required by the script

Check the imports in `cerelog_lsl.py` to identify required packages. Install them in the Python environment that will run the script.

## Usage

1. Connect and configure the CERELOG/ADS1299 hardware.
2. From the directory containing the script, run:

	```bash
	python cerelog_lsl.py
	```

3. Use an LSL-compatible receiver, such as LabRecorder, to discover and record the published stream.

The available options, stream metadata, and device configuration are defined by `cerelog_lsl.py`; consult the script before changing hardware or acquisition settings.

## Troubleshooting

- If no stream is detected, verify the script is running and check device connectivity and network/firewall settings.
- For an import error, install the missing dependency into the active Python environment.
- If channel data looks incorrect, verify the hardware setup and channel configuration in the script.

Validate recorded data before using it for analysis.
