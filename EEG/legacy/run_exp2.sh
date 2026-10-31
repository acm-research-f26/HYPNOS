#!/bin/bash
cd /Users/ani/Projects/hypnos-eeg-apnea
P="--patients 200 --epochs 3 --K 1"
.venv/bin/python exp.py --name s_F3_seed1     --seed 1 $P
.venv/bin/python exp.py --name s_F3_aux_seed1 --seed 1 --aux $P
.venv/bin/python exp.py --name s_F3_aux_w1    --aux --aux_w 1.0 $P
echo SCREEN2_DONE
