#!/bin/bash
cd /Users/ani/Projects/hypnos-eeg-apnea
for a in A B C D E F; do
  echo "=== $a ===" 
  .venv/bin/python train_new.py --arch $a --patients 200 --epochs 3 --tag screen
done
echo SCREEN_DONE
