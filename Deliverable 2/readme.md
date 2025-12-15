# Assignment 5.2 – Feature Engineering + Classical ML Approach

## Project Overview
This project implements **Approach-1** of the Stress Detection study using the WESAD dataset. The goal is to perform **feature engineering** on chest and wrist sensor data and train classical Machine Learning models for stress detection.

**Approach-1**: Feature Engineering + Machine Learning Classification  
Models used: Logistic Regression, SVM, Random Forest, XGBoost

---

## Dataset

**Name:** WESAD – Wearable Stress and Affect Detection  
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection)  

**Description:**  
Multimodal physiological data from wearable devices, recorded from:

- **Chest Sensors:** ACC (X, Y, Z), ECG, EDA, EMG, Respiration, Temperature  
- **Wrist Sensors:** ACC, EDA, Temperature, BVP  

**Problem Type:** Multiclass Classification  
- 0 → Baseline  
- 1 → Stress  
- 2 → Amusement  
- 3–7 → Others  

**Dataset Used:** `S2.pkl` (subject 2) containing synchronized sensor streams.

---

## Features Extracted

1. **Statistical Features** – mean, median, std, variance, min, max, skew, kurtosis, RMS, percentiles  
2. **Temporal Features** – Zero Crossing Rate (ZCR), Signal Magnitude Area (SMA), energy, peak-to-peak amplitude  
3. **Frequency-Domain Features** – dominant frequency, spectral energy, spectral centroid, spectral entropy  
4. **Physiological Features** – EDA peaks, tonic level, ACC SMA  

**Window Size:** 10 seconds per segment

---

## Models

- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest  
- XGBoost  

**Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC–AUC  

---

## Notebook Execution

1. Load dataset (`S2.pkl`) using Python pickle.
2. Segment signals into 10-second windows.
3. Extract statistical, temporal, frequency, and physiological features.
4. Generate labels using majority voting per window.
5. Split data into train/test sets (80/20) and scale features.
6. Train models and evaluate.
7. Analyze results and feature importance.

---

## Results

- XGBoost or Random Forest typically performs best due to non-linear pattern learning.  
- Most important features: Spectral entropy, dominant frequency, RMS, SMA, peak-to-peak amplitude.  
- Limitations: Only one subject, basic features, no HRV or deep learning models.  

---

## Requirements

- Python >=3.9  
- pandas, numpy, scipy  
- scikit-learn, xgboost  
- matplotlib, seaborn  
- Jupyter Notebook  

---

## Author

**Name:** Ayesha Javaid & Muhammad Hassan

**Course:** Machine Learning  
**Assignment:** 5.2 – Feature Engineering + Classical ML
