import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Read data
X = pd.read_csv(
'https://raw.githubusercontent.com/ltdaovn/dataset/master/MeanShift-3D.csv',
sep='\t')
#Visualizing the data points:
data_fig = plt.figure(figsize=(12, 10))
ax = data_fig.add_subplot(111, projection ='3d')
ax.scatter( X.iloc[:, 0],
X.iloc[:, 1],
X.iloc[:, 2], marker ='o',color ='green')

plt.show()
#Importing libraries:
from sklearn.cluster import estimate_bandwidth
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
#Now we can define the mean shift cluster model and fit it into our data.
msc = MeanShift(bandwidth=bandwidth, bin_seeding=True)
msc.fit(X)
cluster_centers = msc.cluster_centers_
labels = msc.labels_
cluster_label = np.unique(labels)
n_clusters = len(cluster_label)
n_clusters
#Visualizing the clusters:
msc_fig = plt.figure(figsize=(12, 10))
ax = msc_fig.add_subplot(111, projection ='3d')
ax.scatter( X.iloc[:, 0], X.iloc[:, 1], X.iloc[:, 2], marker ='o',color ='red')

ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
cluster_centers[:, 2], marker ='o', color ='green',
s = 300, linewidth = 5, zorder = 10)
plt.title('Estimated number of clusters: %d' % n_clusters)
plt.show()