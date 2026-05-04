import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

#Load data from preprocessing file
X_train = np.load('X_train.npy')
X_val = np.load('X_val.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')
y_test = np.load('y_test.npy')
feature_names = list(np.load('feature_names.npy', allow_pickle=True))

#Binarize target variable using median (542) as threshold with 1 = high demand, 0 = low demand
median = 542
y_train_bin = (y_train > median).astype(int)
y_val_bin = (y_val > median).astype(int)
y_test_bin = (y_test > median).astype(int)

print(f"Training set - High demand: {y_train_bin.sum()}, Low demand: {(y_train_bin==0).sum()}")

#Fit logistic regression on training data
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train_bin)

#Evaluate on validation set
y_pred_val = model.predict(X_val)

print("\nValidation Results:")
print(f"Accuracy: {accuracy_score(y_val_bin, y_pred_val):}")
print(f"Precision: {precision_score(y_val_bin, y_pred_val):}")
print(f"Recall: {recall_score(y_val_bin, y_pred_val):}")
print(f"F1 Score: {f1_score(y_val_bin, y_pred_val):}")
print("\nClassification Summary:")
print(classification_report(y_val_bin, y_pred_val, target_names=['Low Demand', 'High Demand']))