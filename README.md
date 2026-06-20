# Atrial Fibrillation Detection Using Machine Learning on Single-Lead ECG Signals

## Overview

This project presents an exploratory machine-learning workflow for classifying short single-lead ECG recordings using the PhysioNet/Computing in Cardiology Challenge 2017 dataset. The analysis focuses on ECG signal preprocessing, feature extraction, class imbalance handling, and model comparison using interpretable engineered ECG features.

The final workflow compares a Linear Support Vector Machine baseline, a LightGBM classifier with moderated class weights, and a Hybrid SVM–LightGBM model. The goal is to evaluate how feature-based machine-learning models perform for multiclass ECG rhythm classification while keeping the workflow reproducible and interpretable.

## Dataset

**Source:** PhysioNet/Computing in Cardiology Challenge 2017
**Data type:** Short single-lead ECG recordings
**Sampling rate:** 300 Hz
**Task:** Multiclass rhythm classification

The rhythm labels include:

* Normal rhythm (`N`)
* Atrial fibrillation (`A`)
* Other rhythm (`O`)
* Noisy recording (`~`)

The raw ECG data are not included in this repository. Users should download the dataset separately from PhysioNet and configure the local dataset path before running the notebook.

## Configuration

The dataset location is read from an environment variable called `AF_DATA_DIR` instead of being hardcoded. This makes the notebook portable across machines.

Set the environment variable before launching Jupyter. For example:

```bash
export AF_DATA_DIR="/path/to/af-classification-from-a-short-single-lead-ecg-recording-the-physionet-computing-in-cardiology-challenge-2017-1.0.0"
```

Expected dataset structure:

```text
AF_DATA_DIR/
├── REFERENCE-v3.csv
└── extracted_training2017/
    └── training2017/
        ├── A00001.mat
        ├── A00002.mat
        └── ...
```

## Methodology

### ECG Preprocessing

The preprocessing workflow includes:

1. Loading ECG signals from `.mat` files
2. Applying a 0.5–50 Hz bandpass filter
3. Z-score normalization
4. Signal polarity correction
5. Segmentation using 10-second sliding windows
6. ECG feature extraction

### Feature Extraction

The extracted features include:

* RR-interval variability features
* SDNN and RMSSD
* Poincaré plot features
* Frequency-domain features
* Signal amplitude features
* Signal entropy and energy
* Basic statistical features such as skewness and kurtosis

### Validation Design

A record-level train-test split was used so that all segments from the same ECG recording remain within the same split. This reduces leakage risk from overlapping ECG segments. Feature scaling was fit only on the training data and then applied to the test data.

## Models Compared

Three models were evaluated:

1. **Linear SVM**

   * Used as a simple baseline model
   * Trained on scaled engineered ECG features

2. **LightGBM with Moderated Class Weights**

   * Nonlinear gradient-boosting model
   * Class weights used to account for label imbalance

3. **Hybrid SVM–LightGBM**

   * Combines normalized Linear SVM decision scores with LightGBM predicted probabilities
   * Evaluated as an exploratory hybrid modeling approach

## Results

| Model                                            | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
| ------------------------------------------------ | -------: | -----------------: | --------------: | ----------: |
| Hybrid SVM–LightGBM with Moderated Class Weights |   0.7203 |             0.7153 |          0.7203 |      0.7132 |
| LightGBM with Moderated Class Weights            |   0.7182 |             0.7127 |          0.7182 |      0.7115 |
| Linear SVM                                       |   0.6399 |             0.6631 |          0.6399 |      0.6210 |

## Key Findings

The Hybrid SVM–LightGBM model achieved the highest overall performance, with an accuracy of approximately 72% and a weighted F1 score of approximately 0.71 on the record-level test set. LightGBM performed very similarly, suggesting that most of the predictive signal was captured by the nonlinear gradient-boosting model.

The Linear SVM baseline had lower performance, indicating that a linear decision boundary was less effective for the engineered ECG features. Class-specific evaluation showed that normal rhythm was classified most strongly, while atrial fibrillation and other rhythm showed moderate performance. The noisy recording class remained the most difficult to classify because of class imbalance and overlap with other rhythm categories.

## Limitations

This project is an exploratory machine-learning analysis and should not be interpreted as clinical validation. Although the workflow uses record-level splitting, the modeling unit is still the ECG segment rather than the patient.

Generalizability may be limited because the analysis uses one public ECG dataset with a specific recording format, sampling rate, label structure, and preprocessing workflow. Model performance may differ across ECG devices, clinical settings, patient populations, recording durations, or multi-lead ECG systems.

The model also relies on engineered ECG features. While these features are interpretable, they may not capture all complex waveform morphology patterns present in raw ECG signals. Several features depend on accurate R-peak detection, so noisy or irregular recordings may produce less reliable feature estimates.

## Future Work

Future improvements could include:

* External validation on an independent ECG dataset
* Nested grouped cross-validation
* Additional ECG morphology and signal-quality features
* Model calibration and threshold-based evaluation
* More detailed class-specific error analysis
* Binary atrial fibrillation detection as a clinically focused follow-up task
* Deep learning models trained directly on raw ECG waveforms, if sufficient data and compute resources are available

## Requirements

Main Python libraries used:

```text
numpy
pandas
scipy
matplotlib
seaborn
neurokit2
scikit-learn
lightgbm
```

## Repository Contents

```text
.
├── Atrial_Fibrillation_Detection.ipynb
└── README.md
```

