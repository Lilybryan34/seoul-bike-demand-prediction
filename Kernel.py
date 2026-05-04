import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score

#Load preprocessed data
X_train = np.load('X_train.npy')
X_val = np.load('X_val.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')
y_test = np.load('y_test.npy')
feature_names = list(np.load('feature_names.npy', allow_pickle=True))

#Try a range of alpha and gamma values using validation set
alphas = [0.001,0.01, 0.1, 1, 10, 100]
gammas = [0.0001,0.001, 0.01, 0.1, 1]


best_alpha = None
best_gamma = None
best_mse = float('inf')

for alpha in alphas:
    for gamma in gammas:
        model = KernelRidge(kernel='rbf', alpha=alpha, gamma=gamma)
        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred_val)
        print(f"Alpha: {alpha}, Gamma: {gamma}, MSE: {mse:.2f}")
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
            best_gamma = gamma

print(f"\nBest alpha: {best_alpha}, Best gamma: {best_gamma}")

#fit the final model with best parameters
final_model = KernelRidge(kernel='rbf', alpha=best_alpha, gamma=best_gamma)
final_model.fit(X_train, y_train)

#Evaluate on validation set
y_pred_val = final_model.predict(X_val)
mse_val = mean_squared_error(y_val, y_pred_val)
r2_val = r2_score(y_val, y_pred_val)

print(f"\nFinal KRR Model (alpha={best_alpha}, gamma={best_gamma})")
print(f"MSE: {mse_val:}")
print(f"R2: {r2_val:}")

#Final eval
y_pred_test = final_model.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
print(f"\n=== Test Set Results ===")
print(f"MSE: {mse_test:.2f}")
print(f"R2: {r2_test:.4f}")
