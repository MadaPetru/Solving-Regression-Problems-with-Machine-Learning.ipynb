import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# step 1 grab data
churn_df = pd.read_csv("customer_churn.csv")
churn_df.head()

# step 2 visualize the data
plt.rcParams["figure.figsize"] = [8, 10]
churn_df.Exited.value_counts().plot(kind='pie', autopct='%1.0f%%')

# step 3, get rid of unnecesarly columns and data
churn_df = churn_df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# step 4, divede data into features and labels
X = churn_df.drop(['Exited'], axis=1)
y = churn_df['Exited']

# print(X.head())

# step 5, convert categorical data to numbers

numerical = X.drop(['Geography', 'Gender'], axis=1)
categorical = X.filter(['Geography', 'Gender'])
cat_numerical = pd.get_dummies(categorical, drop_first=True)
X = pd.concat([numerical, cat_numerical], axis=1)

# print(X.head())

# step 6, divede data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# step 7, normalize data, optional step

# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# step 8, use the logistic regression model

# log_clf = LogisticRegression()
# classifier = log_clf.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)

# step 9, test you model to see how precises the data predicted is

# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print(accuracy_score(y_test, y_pred))

# try with knn classification alghorithm

# knn_clf = KNeighborsClassifier(n_neighbors=5)
# classifier = knn_clf.fit(X_train, y_train)
#
# y_pred = classifier.predict(X_test)
#
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print(accuracy_score(y_test, y_pred))


#try with random forest classification
# rf_clf = RandomForestClassifier(random_state=42, n_estimators=500)
# classifier = rf_clf.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
#
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))
# print(accuracy_score(y_test, y_pred))

#try with support vector classification

# svm_clf = svm.SVC()
# classifier = svm_clf .fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
#
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))
# print(accuracy_score(y_test, y_pred))

#predictin a single value

rf_clf = RandomForestClassifier(random_state=42, n_estimators=500)

classifier = rf_clf.fit(X_train, y_train)

sc = StandardScaler()
single_record = sc.transform(X.values[100].reshape(1, -1))
predicted_churn = classifier.predict(single_record)
print(predicted_churn)