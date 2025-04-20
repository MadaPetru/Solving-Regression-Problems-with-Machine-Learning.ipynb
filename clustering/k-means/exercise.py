import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

penguins_df = pd.read_csv('penguins_lter.csv')

# replace the empty values from columns
all_columns = penguins_df.columns
for column in all_columns:
    penguins_df[column] = penguins_df[column].fillna(method='ffill').fillna(method='bfill')

#this is just to check if i handled the null values correctly
# print("Columns with NaNs (if any):")
# print(penguins_df.isnull().sum()[penguins_df.isnull().sum() > 0])

# step 1, dividing data into features and labels, for clustering we do not use data labes
features = penguins_df.drop(["Species"], axis=1)
labels = penguins_df.filter(["Species"], axis=1)

# step 2, convert to numberical data
categorical_columns = ['studyName', 'Region', 'Island', 'Stage', 'Individual ID', 'Clutch Completion', 'Date Egg',
                       'Sex', 'Comments']
categorical = features.filter(categorical_columns)

numerical = features.drop(categorical_columns, axis=1)

cat_numerical = pd.get_dummies(categorical, drop_first=True)

features = pd.concat([numerical, cat_numerical], axis=1)
features = features.values

# step 3, find the optimal number of clustes usign the elbow method !!!

# loss = []
# for i in range(1, 11):
#     km = KMeans(n_clusters=i).fit(features)
#     loss.append(km.inertia_)
# plt.plot(range(1, 11), loss)
# plt.title('Finding Optimal Clusters via Elbow Method')
# plt.xlabel('Number of Clusters')
# plt.ylabel('loss')
# plt.show()

# try now with 5 clusters, after we find the optimal number of clusters
# training KMeans with 5 clusters
km_model = KMeans(n_clusters=5)
km_model.fit(features)

# display the plot
# pring the data points with prediced labels
plt.scatter(features[:, 0], features[:, 1], c=km_model.labels_, cmap='rainbow')

# print the predicted centroids
plt.scatter(km_model.cluster_centers_[:, 0], km_model.cluster_centers_[:, 1], s=100, c='black')
plt.show()

# step 4
# check to see how close the alghorithm is,
# using the labels, the actual results that we are trying to classify

# le = preprocessing.LabelEncoder()
# labels = le.fit_transform(labels['Species'])
# # pring the data points with original labels
# plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='rainbow')
# plt.show()
