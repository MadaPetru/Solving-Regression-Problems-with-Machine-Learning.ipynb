import sys

import numpy as np
import pandas as pd
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


def find_columns_with_missing_or_question(df):
    # Check for "?" values
    question_mask = df.isin(["?"]).any()

    # Convert "?" to NaN temporarily to check for real NaNs too
    df_with_nans = df.replace("?", pd.NA)
    nan_mask = df_with_nans.isna().any()

    # Combine both masks
    combined_mask = question_mask | nan_mask

    # Return list of columns with missing data
    columns_with_missing = combined_mask[combined_mask].index.tolist()
    return columns_with_missing


categorical_columns = ['make', 'fuel-type', 'aspiration',
                       'num-of-doors', 'body-style',
                       'drive-wheels', 'engine-location',
                       'wheel-base', 'engine-type',
                       'num-of-cylinders', 'fuel-system']
file_path = 'Automobile_data.csv'

# step 1, read data
df = pd.read_csv(file_path)

# step 2, replace null values with nan
df = df.replace('?', np.nan)
# eliminate the values where we do not know the price, because is the label, the target value
df = df.dropna(subset=['price'])

# step 3, choose your alghortim to handle the missing data
# the types of alghoritm used to replace missing data:
# Mean, Median, and Mode Imputation
# Forward and Backward Fill
# Interpolation Techniques

# converting to numerical to be able to made mathematic operations
# df['normalized-losses'] = pd.to_numeric(df['normalized-losses'])
#
# mean_imputation = df['normalized-losses'].fillna(df['normalized-losses'].mean())
# median_imputation = df['normalized-losses'].fillna(df['normalized-losses'].median())
# mode_imputation = df['normalized-losses'].fillna(df['normalized-losses'].mode().iloc[0])
#
# print("\nImputation using Mean:")
# print(mean_imputation)
#
# print("\nImputation using Median:")
# print(median_imputation)
#
# print("\nImputation using Mode:")
# print(mode_imputation)
#
# linear_interpolation = df['normalized-losses'].interpolate(method='linear')
# quadratic_interpolation = df['normalized-losses'].interpolate(method='quadratic')
#
# forward_fill = df['normalized-losses'].fillna(method='ffill')
# backward_fill = df['normalized-losses'].fillna(method='bfill')
#
# print("\nForward Fill:")
# print(forward_fill)
#
# print("\nBackward Fill:")
# print(backward_fill)
#
# print("\nLinear Interpolation:")
# print(linear_interpolation)
#
# print("\nQuadratic Interpolation:")
# print(quadratic_interpolation)

columns_with_missing_values = find_columns_with_missing_or_question(df)
for column in columns_with_missing_values:
    if column not in categorical_columns:
        df[column] = pd.to_numeric(df[column])
        df[column] = df[column].fillna(df[column].median())
    df[column] = df[column].fillna(df[column].mode().iloc[0])

# print("Columns with NaNs (if any):")
# print(df.isnull().sum()[df.isnull().sum() > 0])

# step 4, choose the features and the labels

X = df.drop(['price'], axis=1)  # features
y = df["price"]  # label

# step 5, convert to numerical data

categorical = X.filter(categorical_columns)

numerical = X.drop(categorical_columns, axis=1)

cat_numerical = pd.get_dummies(categorical, drop_first=True)

X = pd.concat([numerical, cat_numerical], axis=1)

# step 5, divide the test and learn data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# step 6, test with diferent alghoritms the output

lin_reg = LinearRegression()
knn_reg = KNeighborsRegressor(n_neighbors=5)
rf_reg = RandomForestRegressor(random_state=42, n_estimators=500)
svm_reg = svm.SVR()

# with knn i fot erros
alghoritms = [lin_reg, rf_reg, svm_reg]

error_min = sys.maxsize
model_with_best_performance = lin_reg
for model in alghoritms:
    regressor = model.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    error = metrics.mean_absolute_error(y_test, y_pred)
    if error < error_min:
        error_min = error
        model_with_best_performance = model

print(model_with_best_performance)
print('Mean Absolute Error:', error_min)
