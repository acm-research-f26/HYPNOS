#!/bin/bash
cd /Users/ani/Projects/hypnos-eeg-apnea
.venv/bin/python train_new.py --arch C --epochs 12 --tag full --patience 3
.venv/bin/python train_new.py --arch F --epochs 8 --tag full --patience 2
echo FULL_DONE
