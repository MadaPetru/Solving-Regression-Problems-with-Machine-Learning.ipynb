import numpy as np
import seaborn as sns
import pandas as pd
from sklearn import preprocessing
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

# EXAMPLE 1

# generating dummy data of 10 records with 2 clusters
# features, labels = make_blobs(n_samples=10, centers=2, cluster_std = 2.00)

# plotting the dummy data
# plt.scatter(features[:,0], features[:,1], color ='r' )
#
# #adding numbers to data points
# annots = range(1, 11)
# for label, x, y  in zip(annots, features[:, 0], features[:, 1]):
#     plt.annotate(
#         label,
#         xy=(x, y), xytext=(-3, 3),
#         textcoords='offset points', ha='right', va='bottom')
# plt.show()
#
# dendos = linkage(features, 'single')
#
# annots = range(1, 11)
#
# dendrogram(dendos,
#             orientation='top',
#             labels=annots,
#             distance_sort='descending',
#             show_leaf_counts=True)
# plt.show()

# training agglomerative clustering model
# hc_model = AgglomerativeClustering(n_clusters=2, linkage='ward')
# hc_model.fit_predict(features)
#
# #pring the data points
# plt.scatter(features[:,0], features[:,1], c= hc_model.labels_, cmap='rainbow' )
#
# plt.show()

# EXAMPLE 2

# generating dummy data of 500 records with 4 clusters
# features, labels = make_blobs(n_samples=500, centers=4, cluster_std = 2.00)
#
# #plotting the dummy data
# # plt.scatter(features[:,0], features[:,1] )
#
# # performing kmeans clustering using AgglomerativeClustering class
# hc_model = AgglomerativeClustering(n_clusters=4, linkage='ward')
# hc_model.fit_predict(features)
#
# #pring the data points
# plt.scatter(features[:,0], features[:,1], c= hc_model.labels_, cmap='rainbow' )
# plt.show()

# EXAMPLE WITH IRIS PLANTS
iris_df = sns.load_dataset("iris")
iris_df.head()

# dividing data into features and labels
features = iris_df.drop(["species"], axis=1)
labels = iris_df.filter(["species"], axis=1)
features.head()

# training agglomerative clustering model
features = features.values
hc_model = AgglomerativeClustering(n_clusters=3, linkage='ward')
hc_model.fit_predict(features)

# #pring the data points
# plt.scatter(features[:,0], features[:,1], c= hc_model.labels_, cmap='rainbow' )
# plt.show()

# create the denogram
# plt.figure(figsize=(10, 7))
# plt.title("Iris Dendograms")
# dend = shc.dendrogram(shc.linkage(features, method='ward'))
# plt.show()


# Print the plot after the prediction
le = preprocessing.LabelEncoder()
labels = le.fit_transform(labels)

# pring the data points with original labels
plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='rainbow')
plt.show()
