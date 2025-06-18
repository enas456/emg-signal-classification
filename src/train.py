import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the feature + gesture Excel file
featGest = pd.read_excel("featuressndgestures.xlsx")

# Number of subjects and gestures per subject
num_subjects = 36
gestures_per_subject = 6

# Split data into subjects (each subject has 6 gesture rows)
subjects = []
for i in range(num_subjects):
    start = i * gestures_per_subject
    end = start + gestures_per_subject
    subject_data = featGest.iloc[start:end].to_numpy()
    subjects.append(subject_data)

# Feature and label separation
num_features_per_channel = 18
total_features = num_features_per_channel * 8  # = 144
label_column_index = 144  # Python index (MATLAB 145)

# Containers for overall predictions
all_true_labels = []
all_predicted_labels = []

# Leave-One-Subject-Out cross-validation
for test_subject in range(num_subjects):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for i, subject in enumerate(subjects):
        features = subject[:, :label_column_index]
        labels = subject[:, label_column_index]

        if i == test_subject:
            test_data.append(features)
            test_labels.append(labels)
        else:
            train_data.append(features)
            train_labels.append(labels)

    # Concatenate training and test data
    train_data = np.vstack(train_data)
    train_labels = np.hstack(train_labels)
    test_data = np.vstack(test_data)
    test_labels = np.hstack(test_labels)

    # Normalize (z-score)
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # Train ensemble model (bagging with decision trees)
    model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42)
    model.fit(train_data, train_labels)

    # Predict
    predictions = model.predict(test_data)

    all_true_labels.extend(test_labels)
    all_predicted_labels.extend(predictions)

# Accuracy
accuracy = accuracy_score(all_true_labels, all_predicted_labels)
print(f"✅ Overall Accuracy: {accuracy * 100:.2f}%")

# Confusion matrix
cm = confusion_matrix(all_true_labels, all_predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
