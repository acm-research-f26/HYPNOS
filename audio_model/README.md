
# HYPNOS Audio Model

## Overview
The HYPNOS audio module uses a convolutional neural network
(CNN) to identify 30-second audio windows that may contain
obstructive apnea or hypopnea events.

This is a research prototype, not a medical diagnostic tool.

## Model
- Architecture: Three Conv2D blocks followed by
  GlobalAveragePooling2D and a dense classifier
- Input: 16 kHz audio converted to Mel spectrograms
- Window size: 30 seconds
- Window stride: 10 seconds
- Output: Binary prediction and model score

## Performance
Evaluated on 13,191 windows from five held-out APSAA subjects.

Using a validation-selected threshold of approximately 0.199:
- Overall accuracy: 67%
- Apnea/hypopnea precision: 63%
- Apnea/hypopnea recall: 86%
- Apnea/hypopnea F1: 0.73

## Files
- predict_audio.py: Inference pipeline
- osa_audio_model_30s.py: Model and preprocessing code
- hypnos_audio_v2.keras: Trained model checkpoint
- best_threshold.txt: Validation-selected classification threshold

## Running Inference
From the HYPNOS project root:

    python3 predict_audio.py test_audio.wav

Results are saved to audio_predictions.csv.

## Integration
The prediction pipeline can be called from another Python
module using:

    from predict_audio import predict_recording

    results = predict_recording("recording.wav")

The returned DataFrame contains:
- start_sec
- end_sec
- probability
- prediction

## Limitations
The model has not been validated on real iPhone or Android
overnight recordings. Results may be affected by recording
conditions, microphone differences, background noise,
and annotation quality.
