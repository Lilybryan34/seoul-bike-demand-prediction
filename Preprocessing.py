import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

#Load in dataset
df = pd.read_csv('SeoulBikeData.csv', encoding='unicode_escape')
# print(df.head())

#Check for missing values
print(df.isna().sum())

#remove the nonfunctioning days
df = df[df['Functioning Day'] == 'Yes']
df = df.drop(columns=['Functioning Day'])

#one-hot encoding for season and holiday
df = pd.get_dummies(df, columns=['Seasons', 'Holiday'], drop_first=True, dtype=int)
print(df.head())


#non-target variable  statistics
nontargets = ['Rented Bike Count', 'Hour','Temperature(°C)','Humidity(%)',
            'Wind speed (m/s)', 'Visibility (10m)','Dew point temperature(°C)',
            'Solar Radiation (MJ/m2)','Rainfall(mm)','Snowfall (cm)']
for feature in nontargets:
    print(f"{feature}:")
    print(f"Mean: {df[feature].mean():}")
    print(f"Min: {df[feature].min():}")
    print(f"Max:{df[feature].max():}")

#preliminary analysis of statistics (correlation with target variable)
numeric = df.select_dtypes(include='number')
corr = numeric.corr()['Rented Bike Count'].sort_values(ascending=False)
print(corr.round(2))
print()

#target variable statistics
print("Mean:", round(df['Rented Bike Count'].mean(), 2))
print("Median:", round(df['Rented Bike Count'].median(), 2))
print("Min:", df['Rented Bike Count'].min())
print("Max:", df['Rented Bike Count'].max())

#define features and target (drop date because date should not be scaled)
X = df.drop(columns=['Rented Bike Count', 'Date'])
y = df['Rented Bike Count']

#Train/validation/test split (70/15/15) in temporal order
n = len(X)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X.iloc[:train_end]
X_val = X.iloc[train_end:val_end]
X_test = X.iloc[val_end:]

y_train = y.iloc[:train_end]
y_val = y.iloc[train_end:val_end]
y_test = y.iloc[val_end:]

#scale data after splitting; fit only on training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

#save splits & feature names
np.save('feature_names.npy', X.columns.values)
np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)
np.save('y_test.npy', y_test)

print(df[['Snowfall (cm)', 'Seasons_Winter', 'Temperature(°C)', 'Humidity(%)']].corr())