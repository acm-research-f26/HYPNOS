# HYPNOS Single-Channel EEG Sleep Apnea Detection

**paper title:** Detection of Sleep Apnea from Single-Channel Electroencephalogram (EEG) Using an Explainable Convolutional Neural Network (CNN)

## ⚲ project summary

> This implementation recreates the single-channel EEG sleep apnea detection pipeline described by Barnes et al. The original study trained a convolutional neural network to classify 30-second C4-A1 EEG segments as apnea or non-apnea. This initial HYPNOS implementation uses a subset of the UCD Sleep Apnea Database to reproduce the preprocessing, labeling procedure, CNN architecture, and evaluation pipeline.

## ⌖ motivation

> Traditional sleep apnea diagnosis usually relies on overnight polysomnography containing several physiological signals. This research investigates whether apnea events can instead be detected using only one EEG channel. A successful single-channel approach could eventually contribute to simpler and more accessible sleep monitoring systems.

## ✎ᝰ novelty

> The model works directly with raw single-channel EEG instead of relying on manually selected frequency features. A relatively small 1D convolutional neural network learns useful features directly from the EEG. The original study also analyzed the learned filters and found that the model used EEG information related to known sleep apnea biomarkers, especially delta and beta activity.

## ✰ methodology

1. **dataset:** St. Vincent's University Hospital / University College Dublin Sleep Apnea Database  
   https://physionet.org/content/ucddb/1.0.0/

2. **architecture:** Three-layer one-dimensional convolutional neural network. The first convolution uses 8 filters with kernel length 35, the second uses 128 filters with kernel length 175, and the third uses 16 filters with kernel length 175. Each convolution is followed by batch normalization, ELU activation, max pooling, and dropout. The network then uses a 64-node dense layer and two-class output layer.

3. **evaluation:** Subjects are separated between training, validation, and testing so EEG windows from a test participant are never included in training. Training and validation data are undersampled to reduce class imbalance while the test subject is left with its original class distribution.

4. **metrics:** Accuracy, Matthews correlation coefficient, ROC AUC, confusion matrix, training loss, and validation loss.

## ⛰︎ impact

> The project tests whether a single EEG signal contains enough information to identify sleep apnea events. Reducing the number of signals needed for screening could eventually make sleep monitoring less complicated and easier to perform outside of a full sleep laboratory.

#### future work

> The current version is a small HYPNOS prototype trained on five UCD participants. The next step is to reproduce the original experiment using the much larger SHHS Visit 2 dataset and subjectwise 10-fold cross-validation. HYPNOS can then investigate sleep-stage prediction, full-night apnea event detection, AHI estimation, and explainability methods such as filter lesioning and critical-band masking.

## **additional sources:**

> **Original paper:** Barnes, Lachlan D., et al. “Detection of Sleep Apnea from Single-Channel Electroencephalogram (EEG) Using an Explainable Convolutional Neural Network (CNN).” *PLOS ONE*, 2022.  
> https://doi.org/10.1371/journal.pone.0272167
>
> **UCD Sleep Apnea Database:**  
> https://physionet.org/content/ucddb/1.0.0/
>
> **Sleep Heart Health Study:**  
> https://sleepdata.org/datasets/shhs
>
> **PyTorch:**  
> https://pytorch.org/
>
> **SciPy:**  
> https://scipy.org/
>
> **PyEDFlib:**  
> https://pyedflib.readthedocs.io/