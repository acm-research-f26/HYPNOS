# HYPNOS – Obstructive Sleep Apnea Audio Model

## Paper

**Acoustic Screening for Obstructive Sleep Apnea in Home Environments Based on Deep Neural Networks**

This project is a reproduction of the audio-processing and deep-learning pipeline described in the paper.

The original source code and full labeled dataset used by the authors were not publicly available, so this implementation was created from the methodology described in the paper.

---

## Project Summary

The goal of this project is to process overnight audio recordings and prepare them for classification of possible apnea or hypopnea events.

The pipeline implemented in Python is:

```text
Audio recording
      ↓
Resample to 16 kHz
      ↓
Split into overlapping time segments
      ↓
Create Mel spectrogram
      ↓
Convert to decibels
      ↓
Normalize features
      ↓
CNN model
      ↓
Segment-level prediction
      ↓
Event counting
      ↓
Estimated AHI
```

The current neural network architecture is implemented successfully, but it has not yet been trained on labeled clinical sleep-audio data.

---

## Audio Preprocessing

The implementation uses the following parameters based on the paper:

- Sample rate: **16,000 Hz**
- Mel filters: **64**
- Frequency range: **70 Hz – 7,500 Hz**
- Window length: **50 ms**
- Frame shift: **20 ms**
- Baseline segment length: **30 seconds**
- Segment shift: **10 seconds**

Each audio segment is converted into a Mel spectrogram and normalized before being provided to the CNN.

---

## CNN Architecture

The implemented CNN contains:

- Conv2D layer with 16 filters
- Batch normalization
- Max pooling
- Dropout
- Conv2D layer with 32 filters
- Batch normalization
- Max pooling
- Dropout
- Conv2D layer with 64 filters
- Batch normalization
- Max pooling
- Dropout
- Flatten layer
- Dense layer with 512 units
- Sigmoid output layer

The final sigmoid output is designed for binary classification:

```text
0 = no apnea/hypopnea event
1 = possible apnea/hypopnea event
```

The current model contains approximately **1.83 million parameters**.

---

## My Modification

The original baseline implementation uses **30-second audio segments**.

I created a second version using **40-second audio segments** while keeping the segment shift at 10 seconds.

The purpose of this modification is to investigate how increasing the amount of audio context in each input segment changes the preprocessing output and the number of model inputs generated from the same recording.

### Baseline

Using a 60-second test recording:

```text
Segment length: 30 seconds
Segment shift: 10 seconds
```

Generated:

```text
0–30 seconds
10–40 seconds
20–50 seconds
30–60 seconds
```

Result:

```text
4 segments
Mel shape per segment: (64, 1501)
```

### Modified Version

Using:

```text
Segment length: 40 seconds
Segment shift: 10 seconds
```

Generated:

```text
0–40 seconds
10–50 seconds
20–60 seconds
```

Result:

```text
3 segments
Mel shape per segment: (64, 2001)
```

---

## Comparison

| Version | Segment Length | Number of Segments | Mel Shape |
|---|---:|---:|---|
| Baseline | 30 seconds | 4 | `(64, 1501)` |
| Modified | 40 seconds | 3 | `(64, 2001)` |

Increasing the segment duration reduces the number of overlapping samples generated from the same recording while increasing the amount of temporal information contained in each model input.

This may allow the network to observe a longer breathing context, although determining whether this improves apnea detection accuracy would require training and evaluating both versions on the same labeled dataset.

---

## Files

### `osa_audio_model_30s.py`

Baseline implementation using 30-second audio segments.

### `osa_audio_model.py`

Modified implementation using 40-second audio segments.

### `test_audio.wav`

Synthetic audio used to verify preprocessing, segmentation, and CNN input dimensions.

---

## Current Status

Completed:

- Audio loading
- 16 kHz resampling
- Mel spectrogram extraction
- dB conversion
- Feature normalization
- Overlapping segmentation
- CNN architecture
- Training function
- Prediction function
- Event counting
- AHI calculation
- OSA severity classification
- 30-second baseline
- 40-second modification

Remaining work for a full clinical reproduction:

- Obtain labeled sleep audio
- Train the CNN
- Create validation and test sets
- Measure accuracy, sensitivity, specificity, and related metrics
- Compare the 30-second and 40-second models after training

---

## Important Limitation

The CNN currently has not been trained using labeled apnea and hypopnea recordings.

Therefore, predictions from the untrained model should **not** be interpreted as real medical or diagnostic results.

The current implementation demonstrates the preprocessing pipeline, neural-network architecture, and experimental modification described above.