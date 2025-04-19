import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

# create dummy data
# generating dummy data of 500 records with 4 clusters
# features, labels = make_blobs(n_samples=500, centers=4, cluster_std=2.00)

# plotting the dummy data
# plt.scatter(features[:, 0], features[:, 1])

# performing kmeans clustering using KMeans class
# km_model = KMeans(n_clusters=4)
# km_model.fit(features)

# printing centroid values
# print(km_model.cluster_centers_)

# printing predicted label values
# print(km_model.labels_)

# pring the data points
# plt.scatter(features[:, 0], features[:, 1], c=km_model.labels_, cmap='rainbow')

# print the centroids
# plt.scatter(km_model.cluster_centers_[:, 0], km_model.cluster_centers_[:, 1], s=100, c='black')

# print actual datapoints
# plt.scatter(features[:,0], features[:,1], c= labels, cmap='rainbow' )

# plt.show()


# import the data set for the tutorial example

iris_df = sns.load_dataset("iris")
# print(iris_df.head())

# step 1, dividing data into features and labels, for clustering we do not use data labes
features = iris_df.drop(["species"], axis=1)
labels = iris_df.filter(["species"], axis=1)
print(features.head())

# step 2, choose the number of clusters, here we choose 4
# training KMeans model
features = features.values
km_model = KMeans(n_clusters=4)
km_model.fit(features)

# To print labels of the Iris dataset, execute the following script:
print(km_model.labels_)

# step 3, display the plot for the 4 clusters
# print the data points
plt.scatter(features[:, 0], features[:, 1], c=km_model.labels_, cmap='rainbow')

# print the centroids
plt.scatter(km_model.cluster_centers_[:, 0], km_model.cluster_centers_[:, 1], s=100, c='black')
plt.show()

# step 4, find the optimal number of clustes usign the elbow method !!!
# Till now, in this chapter,
# we have been randomly initializing the value of K or the number of clusters.
# However, there is a way to find the ideal number of clusters.
# The method is known as the elbow method.
# In the elbow method, the value of inertia obtained by training K Means clusters with different number of K is plotted.
# The inertia represents the total distance between the data points within a cluster.
# Smaller inertia means that the predicted clusters are robust and close to the actual clusters.
# To calculate the inertia value, you can use the inertia_ attribute of the KMeans class object.


# training KMeans on K values from 1 to 10
loss =[]
for i in range(1, 11):
    km = KMeans(n_clusters = i).fit(features)
    loss.append(km.inertia_)

#printing loss against number of clusters
plt.plot(range(1, 11), loss)
plt.title('Finding Optimal Clusters via Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('loss')
plt.show()