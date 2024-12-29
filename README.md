# 🫀 Atrial Fibrillation Detection Using Multi-Model Approaches

## 📌 Overview
This project implements various machine learning models for detecting Atrial Fibrillation (AF) from single-lead ECG recordings using the PhysioNet/Computing in Cardiology Challenge 2017 dataset. The study compares different approaches including Dual SVM, LightGBM, CNN, and a novel Hybrid model combining SVM and LightGBM.

## 🎯 Key Features
- Multi-model comparison approach
- Comprehensive ECG signal preprocessing pipeline
- Advanced feature extraction techniques
- Handling of imbalanced datasets
- Performance optimization through model hybridization

## 📊 Dataset
- **Source**: PhysioNet/Computing in Cardiology Challenge 2017
- **Characteristics**:
  - Single short-lead ECG recordings (30-60 seconds)
  - 300 Hz sampling rate
  - Band-pass filtered
- **Classes**:
  - Normal Sinus Rhythm (5,154 recordings)
  - Atrial Fibrillation (771 recordings)
  - Other Rhythm (2,557 recordings)
  - Noisy Recordings (46 recordings)

## 🛠️ Methodology

### Preprocessing Pipeline
1. Signal Loading and Filtering (0.5-50 Hz bandpass)
2. Z-score Normalization
3. Signal Inversion Correction
4. Segmentation using 10-second sliding windows
5. Feature Extraction
6. Missing Value Imputation
7. Feature Scaling
8. Label Encoding

### Implemented Models
1. **Dual Support Vector Machine**
   - Linear and RBF kernels
   - Hyperparameter optimization
   - Accuracy: 72.36%

2. **LightGBM**
   - Gradient boosting framework
   - Optimized for multiclass classification
   - Accuracy: 75.95%

3. **Convolutional Neural Network**
   - Custom architecture for ECG processing
   - Multiple convolutional layers
   - Accuracy: 70.54%

4. **Hybrid Model (SVM + LightGBM)**
   - Combined approach
   - Weighted averaging
   - Best performing model
   - Accuracy: 76.15%

## 📈 Results

### Model Performance Comparison
| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|---------|-----------|
| Dual SVM | 72.36% | 71.13% | 72.36% | 69.86% |
| LightGBM | 75.95% | 74.78% | 75.95% | 74.48% |
| CNN | 70.54% | 68.01% | 70.54% | 66.78% |
| Hybrid | 76.15% | 74.94% | 76.15% | 74.70% |

## 🚀 Key Findings
- Hybrid model demonstrates superior performance across all metrics
- LightGBM shows robust individual performance
- CNN and Dual SVM face challenges with class imbalance
- Preprocessing steps significantly impact model performance

## 🔍 Limitations
- Signal length variations affect window-based analysis
- Fixed bandpass filter may not be optimal for all signals
- Potential feature selection bias
- Class imbalance challenges

## 🔮 Future Work
- Implementation of longitudinal studies
- Integration of multimodal data
- Exploration of advanced deep learning architectures
- Development of personalized models

## 🛠️ Requirements
- Python 3.x
- scikit-learn
- LightGBM
- TensorFlow/Keras
- numpy
- pandas
- scipy

## 📚 References
1. Clifford, G. D., et al. (2017). AF Classification from a Short Single Lead ECG Recording
2. Geweid, G. G. N., & Chen, J. D. Z. (2022). Automatic Classification of Atrial Fibrillation
3. Hindricks, G., et al. (2021). ESC Guidelines for AF Diagnosis and Management
4. Additional references available in the full paper

## 👥 Contributors
- Chandini Karrothu
- Likhitha Marrapu
- Shivani Battu
