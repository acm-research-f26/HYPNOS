#!/bin/bash
# HYPNOS: PhysioNet 2018 Challenge training set (994 sleep studies), streamed:
# download one record -> convert to 30 s EEG windows -> delete the raw files.
# Output: data/processed/cinc2018/<record>.npz   Log: data/processed/cinc2018_log.txt
# Safe to close and re-run: finished records are skipped.
cd "$(dirname "$0")" || exit 1
mkdir -p data/processed
LOG="data/processed/cinc2018_log.txt"
exec > >(tee -a "$LOG") 2>&1
echo "=== CinC 2018 streaming started $(date) ==="
PY=".venv/bin/python"
"$PY" -c "import wfdb" 2>/dev/null || { echo "Installing wfdb into .venv..."; "$PY" -m pip install -q wfdb || { echo "!! pip install wfdb failed"; exit 1; }; }
caffeinate -i "$PY" -u preprocess_cinc2018.py "$@"   # caffeinate keeps the Mac from idle-sleeping while this runs
echo "=== Ended $(date) ==="
echo "You can close this window."
