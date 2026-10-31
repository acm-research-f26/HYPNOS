#!/bin/bash
cd /Users/ani/Projects/hypnos-eeg-apnea
P="--patients 200 --epochs 3"
.venv/bin/python exp.py --name s_F3        --K 1 $P
.venv/bin/python exp.py --name s_F3_aug    --K 1 --aug $P
.venv/bin/python exp.py --name s_F3_aux    --K 1 --aux $P
.venv/bin/python exp.py --name s_F5        --K 2 $P
.venv/bin/python exp.py --name s_F5_gru    --K 2 --comb gru $P
echo SCREEN_EXP_DONE
