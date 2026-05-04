import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

#Load data from preprocessing file
X_train = np.load('X_train.npy')
X_val = np.load('X_val.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')
y_test = np.load('y_test.npy')
feature_names = list(np.load('feature_names.npy', allow_pickle=True))

#try a range of alpha (lambda) values and pick the best one using validation set
alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
best_alpha = None
best_mse = float('inf')

for alpha in alphas:
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred_val)
    print(f"Alpha: {alpha}, MSE: {mse:.2f}")
    if mse < best_mse:
        best_mse = mse
        best_alpha = alpha

print(f"\nBest alpha: {best_alpha}")

#after finding the best alpha is 0.1, fit the final lasso model
# Fit final Lasso model with best alpha
final_model = Lasso(alpha=best_alpha, max_iter=10000)
final_model.fit(X_train, y_train)

# Evaluate on validation set
y_pred_val = final_model.predict(X_val)
mse_val = mean_squared_error(y_val, y_pred_val)
r2_val = r2_score(y_val, y_pred_val)

#print r2 values for each alpha
for alpha in alphas:
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)
    print(f"Alpha: {alpha}, MSE: {mse:.2f}, R2: {r2:.4f}")

print(f"\nFinal Lasso Model (alpha={best_alpha})")
print(f"MSE: {mse_val:.2f}")
print(f"R2: {r2_val:.4f}")

# Coefficient analysis
coefficients = pd.Series(final_model.coef_, index=feature_names)
print("\nLasso Coefficients:")
print(coefficients.sort_values(ascending=False).round(4))
print(f"\nFeatures shrunk to zero: {sum(final_model.coef_ == 0)}")
print(f"Features retained: {sum(final_model.coef_ != 0)}")