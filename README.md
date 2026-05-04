## Predicting Hourly Bike Sharing Demand in Seoul
Predicting hourly bike sharing demand in Seoul using OLS, Lasso, Logistic Regression, and Kernel Ridge Regression with weather and time features.

## Overview
This project applies machine learning techniques to predict hourly bike rental demand in Seoul using weather and time features.

## Dataset
Seoul Bike Sharing Demand — UCI Machine Learning Repository
https://doi.org/10.24432/C5F62R

## Methods
- Linear Regression (OLS)
- Lasso Regularization
- Logistic Regression
- Kernel Ridge Regression (RBF kernel)

## Results
| Model | R² (Validation) |
|---|---|
| Linear Regression | 0.34 |
| Lasso | 0.37 |
| KRR | 0.64 |

## Requirements
- Python 3.11
- pandas
- numpy
- scikit-learn
- matplotlib

## File Structure
- Preprocessing.py — data cleaning, encoding, splitting, scaling
- Linear Regression.py — OLS baseline model
- Lasso Regularization.py — Lasso feature selection
- Logistic Regression.py — classification model
- Kernel.py — Kernel Ridge Regression
