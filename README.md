# EMG-Based Multi-Hand Gesture Classification for Prosthetic Control

## Abstract

Many individuals suffer from the loss of limb function due to arm amputations or congenital musculoskeletal disorders. Myoelectric prostheses provide a promising avenue for restoring hand functionality, enabling more natural interaction through muscle signals. This project focuses on building a machine learning system to classify hand gestures from surface EMG (sEMG) data collected via the MYO Thalmic armband.

## Project Objectives

* Develop a real-time limb prosthesis control model using myoelectric signals.
* Extract robust time- and frequency-domain features (sFE1, sFE2).
* Train and compare classification algorithms: SVM, Decision Tree, KNN, Naive Bayes, Bagging, and Boosting.
* Evaluate using Leave-One-Subject-Out cross-validation.
* Lay the groundwork for future deployment on real robotic hands (Inmoov).

## Dataset Overview

* Source: Kaggle (CC BY 4.0 License)
* Subjects: 36 individuals
* Sensor: MYO Thalmic bracelet
* Channels: 8 EMG sensors evenly spaced around the forearm
* Gestures: 6 static gestures (Palm, Fist, Flexion, Extension, Radial/Ulnar Deviations)
* Each gesture lasted 3 seconds, followed by a 3-second pause

## Feature Extraction

* Raw data filtered with FIR bandpass filter (1–400 Hz, order=100)
* Extracted 18 features per channel (144 total) including:

  * Time: Mean, RMS, Std, SNR, THD, SINAD, Peak, Crest Factor, etc.
  * Frequency: MeanFreq, MedianFreq, Band Power, Peak Amp, Peak Loc


## Results & Discussion

This project evaluated multiple classifiers across **four different feature extraction/classification strategies**. The overall accuracies for KNN, SVM, and Random Forest (Bagging) are presented below:

| Method                 | KNN (%) | SVM (%) | Random Forest (Bagging) (%) |
| ---------------------- | ------- | ------- | --------------------------- |
| 1st method             | 82.40   | 81.02   | 88.89                       |
| 2nd method             | 76.85   | 77.31   | 82.87                       |
| 3rd method (Lasso)     | 80.09   | 82.87   | 84.72                       |
| 4th method (Two-stage) | 94.44   | 90.74   | 99.07                       |

### 🔎 Key Observations:

* **Two-stage classification system** significantly improved all models, especially **Random Forest**, reaching **99.07% accuracy**.
* **Lasso-based feature selection** boosted SVM performance (82.87%), slightly outperforming KNN in that setting.
* The **1st method** already yielded strong results with ensemble models (Bagging = 88.89%) without advanced selection methods.
* **SVM and KNN** consistently provided competitive performance but lagged behind ensemble techniques.

### Interpretation:

* **Ensemble techniques**, particularly Random Forest, consistently outperformed individual models due to improved generalization and robustness.
* **Feature selection and system design** (e.g., Lasso, two-stage architecture) played a critical role in boosting classification performance.
* **KNN** responded well to optimized configurations, especially in the final two methods, suggesting it is viable for simpler deployments.

## Methods Overview

1. **Filtering**: FIR bandpass filter on all 8 EMG channels
2. **Segmentation**: 100-sample sliding windows per gesture
3. **Feature Extraction**: 18 per channel (144 total)
4. **Normalization**: Z-score standardization
5. **Modeling**: Trained and tested 6 classifiers
6. **Evaluation**: Accuracy, confusion matrix per model

## Key Outcomes

* Achieved robust multi-class classification of EMG-based gestures
* Demonstrated that ensemble models improve generalization
* Created a structured pipeline that can be migrated to real-time control

## Future Work

* Implement advanced deep learning models (e.g., FTDLSTM, LSTM)
* Transition from MATLAB to Python-based deployment
* Integrate with real robotic hands (e.g., Inmoov)
* Optimize models for embedded, low-power use

## Tools and Libraries

* Data Collection: MYO Thalmic Armband
* Signal Processing: FIR Filter, FFT
* Feature Extraction: Custom + MATLAB toolboxes
* Machine Learning: scikit-learn, MATLAB (fitcensemble, fitcknn, fitctree, etc.)

## 📦 Directory Structure

```
classification-multi-hand-gesture-using-emg/

├── src/
│   ├── preprocessing.py      # Filtering and cleaning
│   ├── features.py           # Feature extraction
│   ├── train.py              # Classifier training and evaluation
│   └── load_emg_data.py      # Data utilities
├── main.py                  # Pipeline entry point
├── requirements.txt
├── README.md


## ⚖️ Citation

Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and based on MYO EMG dataset from Kaggle.

---

> "Empowering prosthetic control through EMG, machine learning, and real-time gesture recognition."
