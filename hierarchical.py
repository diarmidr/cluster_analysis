# coding: utf-8
"""This script performs a cluster analysis on day ahead price profiles"""
########################################################################################################################
# Import required tools
########################################################################################################################

# import off-the-shelf scripting modules
from __future__ import division     # Without this, rounding errors occur in python 2.7, but apparently not in 3.4
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage    # For cluster analysis
from scipy.cluster.hierarchy import fcluster   # For cluster membership
from matplotlib import pyplot as plt

# import raw temporal data
raw_data = pd.read_csv('N2EX_clean_hourly_2018.csv')
price = [float(x) for x in raw_data.ix[:, 'price_sterling']]

# Get 24h profile in list format, then stack in array.
arrays = []
for i in range(int(len(price)/24)):
    price_profile = np.array(price[i*24:(i+1)*24])
    arrays = arrays + [price_profile]
input_array = np.stack(arrays, axis=0)
#print(input_array)

# generate linkage matrix (this does all work, after this it's just a case of defining cluster cut-off points)
Z = linkage(input_array, 'ward')

# Elbow plot, for guidance on where to truncate dendrogram

last = Z[-32:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)
plt.figure(figsize=(25, 10))
plt.title('Elbow plot')
plt.xlabel('Clusters identified')
plt.ylabel('Distance travelled to join clusters')
plt.plot(idxs, last_rev, label="Dist")

acceleration = np.diff(last, 2)  # 2nd derivative of the distances
acceleration_rev = acceleration[::-1]
plt.plot(idxs[:-2] + 1, acceleration_rev, label="2nd Deriv Dist")
plt.legend()
plt.show(block=False)


# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')

def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

fancy_dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
    annotate_above=600, # Prevent excessive annotation of merge distances
    max_d = 600
)
plt.show(block=False)

# retrieve cluster membership for each vector
max_d = 600
cluster_membership = fcluster(Z, max_d, criterion='distance')
print(cluster_membership)


# retrieve the number of clusters
cluster_list = []
for i in cluster_membership:
    if i not in cluster_list:
        cluster_list = cluster_list + [i]
n = len(cluster_list)
print ("number of clusters = ", n)

# Generate a CSV where each price datum is indexed by hour then day then cluster membership
hour_index =[]
day_index = []
cluster_index =[]
price_data = []
for i in range(int(len(price)/24)):
    hour_index = hour_index + [x+1 for x in range(len(input_array[i,:]))]
    day = [i for x in input_array[i,:]]
    day_index = day_index + day
    cluster= [cluster_membership[i] for x in input_array[i,:]]
    cluster_index = cluster_index + cluster
    price_profile = price[i*24:(i+1)*24]
    price_data = price_data + price_profile

dataframe_output = pd.DataFrame({"hour": hour_index,
                                "day": day_index,
                                "cluster":cluster_index,
                                "price": price_data
                                })

dataframe_output.to_csv("hierarchical_clustering_max_d=" + str(max_d) + "_n="+ str(n) + ".csv", sep=',')

# Plot clusters by making price v period set for each then doing x-y scatter
plt.figure(figsize=(25, 10))
plt.title('Clusters')
for i in [i+1 for i in range(n)]:
    n_list_period = []  # Rectacle for period data where cluster = n
    n_list_price = []   # Receptacle for price data where cluster = n
    for h in range(len(hour_index)):
        if cluster_index[h] == i:
            n_list_period = n_list_period + [hour_index[h]]
            n_list_price = n_list_price + [price_data[h]]
    plt.plot(n_list_period, n_list_price, label="cluster_" + str(i), marker='.', markersize=8, linestyle='None')

plt.xlabel('Hour of day')
plt.ylabel('Day-ahead electrical price (£.kWh-1)')
axes = plt.gca()
axes.set_xlim([0, 24])
axes.set_ylim([0, 250])
plt.legend()
plt.show()
