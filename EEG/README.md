# EEG: sleep apnea/hypopnea detection from one channel

Per-30 s-window apnea/hypopnea detection from a single central EEG channel (C4-M1), 125 Hz.
Trained on PhysioNet/CinC 2018, tested on a held-out CinC split and three other hospitals (UCDDB, MIT-BIH slpdb, DREAMS).
Results, ablations and caveats are in [RESULTS.md](RESULTS.md); plots are in `plots/`.

## Files
- `apnea.py`: data loading, models, training, scoring and ensemble reports (one file).
- `preprocess_cinc2018.py`: downloads CinC 2018 record by record and writes the windows (C4-M1, C3-M2, labels, stages).
- `build_cache.py`: cleans CinC 2018 windows and writes the float16 cache and patient split.
- `build_cache_c3.py`: adds the second channel (C3-M2) for the channel-flexible model (`--mc`).
- `build_cache_hp.py`: writes the high-passed cache on disk for low-RAM training (`train --mmap`).
- `prep_external.py`: UCDDB / MIT-BIH / DREAMS to windows (`--lowpass HZ --out DIR` for other bands).
- `preprocess.py`: filter / resample / segment / label helpers used by the two scripts above.
- `requirements_apnea.txt`: Python packages.

## Commands
```
python preprocess_cinc2018.py       # PhysioNet -> data/processed/cinc2018 (re-runnable)
python prep_external.py             # UCDDB / MIT-BIH / DREAMS -> data/processed/external
python build_cache.py && python build_cache_c3.py
python build_cache_hp.py 5          # optional, for --hp 5 --mmap
python apnea.py train NAME [--K 1 --aux --hp 5 --causal --mc --patients 0 --epochs 14]
python apnea.py score NAME... [--hp HZ --causal]
python apnea.py report NAME... [--plots FILE]     # several names = ensemble
```
`--hp` and `--causal` add a Butterworth high-pass (zero-phase or forward-only) and must match between training and scoring.

Data, weights and raw scores are not in the repository.
