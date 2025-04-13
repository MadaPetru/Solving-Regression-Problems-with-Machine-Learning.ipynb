import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# step 1, read some data that you can fetch to exercise ai

# print(sns.get_dataset_names())
tips_df = sns.load_dataset("tips")

# step 2, read the first 5 row of data in the exercise that we will use

# print(tips_df.head())

# step 3, divide data into feature and labels
# in our case, the label is the is the set of values from column tip
# where the feature is the set of values for the other columns

X = tips_df.drop(['tip'], axis=1)
y = tips_df["tip"]
# print(X.head())
# print(y.head())

#step 4, converting categorical data to numbers
# numerical variable contains the columns with values that are numerical already

numerical = X.drop(['sex', 'smoker', 'day', 'time'], axis = 1)
# print(numerical.head())

# now we filter the categorical data, like for sex can be female or male
# a common pattern to encode the categorical values is one-hot encoding, for every possible value
# of these columns like for sex can be male or female, we add 2 extra columns sex_female,sex_male
# and we will add 1 to the column femal if is a femal or male if it is a male

categorical = X.filter(['sex', 'smoker', 'day', 'time'])
# print(categorical.head())

cat_numerical = pd.get_dummies(categorical,drop_first=True)
# print(cat_numerical.head())

#step 5, now we merge all data converted in numerical type

X = pd.concat([numerical, cat_numerical], axis = 1)
# print(X.head())

#step 6, now it is a good practice to have a look on the visualisaztion of the data, because
# you may find some pattern or trends, we can find easier corelationn using the corr() function

plt.rcParams["figure.figsize"] = [8,6]
df = pd.concat([X, y], axis = 1)
corr = df.corr()
# print(corr)

# the heat map for me is not displayed in pycharm
# sns.heatmap(corr)

#step 7, now we divide data into training and test sets, for training 80% and for tests 20% usually

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.20, random_state=0)

#step 8, after dividing the data, is good and optional to normalize the data, to uniform the sacle,
# because some values can be too large and the other too small

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform (X_test)

#step 9, use different algorithms to see which one gives the most accurate response

#linear regression

# lin_reg = LinearRegression()
# regressor = lin_reg.fit(X_train, y_train)
# y_pred = regressor.predict(X_test)
# print(y_pred)

# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#!!!!
 # explanation from tutorial
 # By looking at the mean absolute error, it can be concluded that on average,
 # there is an error of 0.70 for predictions, which means that on average,
 # the predicted tip values are 0.70$ more or less than the actual tip values.
#!!!!

#knn -> KNN stands for K-nearest neighbors.
# KNN is a lazy learning algorithm, which is based on finding Euclidean distance between different data points.

# knn_reg = KNeighborsRegressor(n_neighbors=5)
# regressor = knn_reg.fit(X_train, y_train)
#
# y_pred = regressor.predict(X_test)
#
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#Random Forest Regression
#Random forest is a tree-based algorithm that converts features into tree nodes and then uses entropy loss to make predictions.

# rf_reg = RandomForestRegressor(random_state=42, n_estimators=500)
# regressor = rf_reg.fit(X_train, y_train)
# y_pred = regressor.predict(X_test)
#
#
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#Support Vector Regression
#The support vector machine is classification as well as regression algorithms,
# which minimizes the error between the actual predictions and predicted predictions
# by maximizing the distance between hyperplanes that contain data for various records.

# svm_reg = svm.SVR()
#
# regressor = svm_reg.fit(X_train, y_train)
# y_pred = regressor.predict(X_test)
#
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#K Fold Cross Validation
#For more stable results,
# it is recommended that all the parts of the dataset are used at least once for training and once for testing.
# The K-Fold cross-validation technique can be used to do so. With K-fold cross-validation,
# the data is divided into K parts. The experiments are also performed for K parts. In each experiment,
# K-1 parts are used for training, and the Kth part is used for testing.
#For example, in 5-fold cross-validation,
# the data is divided into five equal parts,
# e.g., K1, K2, K3, K4, and K5. In the first iteration, K1–K4 are used for training,
# while K5 is used for testing. In the second test, K1, K2, K3, and K5 are used for training, and K4 is used for testing
# . In this way, each part is used at least once for testing and once for training.

#here the regressor variable i did not install any package or create the variable, so i did not run the demo for this
# print(cross_val_score(regressor, X, y, cv=5, scoring ="neg_mean_absolute_error"))

#Predict a single value

print(tips_df.loc[100])
print('Actual value: ')
print(tips_df.loc[100].tip)

rf_reg = RandomForestRegressor(random_state=42, n_estimators=500)
regressor = rf_reg.fit(X_train, y_train)

single_record = sc.transform (X.values[100].reshape(1, -1))
predicted_tip = regressor.predict(single_record)
print('Predicted value: ')
print(predicted_tip)
