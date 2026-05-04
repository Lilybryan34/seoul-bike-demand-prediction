import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import pandas as pd

#Load data from preprocessing file
X_train = np.load('X_train.npy')
X_val = np.load('X_val.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')
y_test = np.load('y_test.npy')

#Fit ordinary least squares linear regression on training data
model = LinearRegression()
model.fit(X_train, y_train)

#evaluate on validation set
y_pred_val = model.predict(X_val)
mse_val = mean_squared_error(y_val, y_pred_val)
r2_val = r2_score(y_val, y_pred_val)

print("Validation Results: ")
print(f"MSE: {mse_val:}")
print(f"R2: {r2_val:}")

#Time series cross validation (because k value did not respect temporal order)
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2')

print("\n5-Fold Time Series Cross Validation:")
print(f"R2 scores: {cv_scores.round(4)}")
print(f"Mean R2: {cv_scores.mean():.4f}")
print(f"Std R2: {cv_scores.std():.4f}")

#Coefficient analysis
feature_names = list(np.load('feature_names.npy', allow_pickle=True))
coefficients = pd.Series(model.coef_, index=feature_names)
print("\nFeature Coefficients:")
print(coefficients.sort_values(ascending=False).round(2))


