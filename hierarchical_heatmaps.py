# coding: utf-8
"""This script performs a cluster analysis on day ahead price profiles"""
########################################################################################################################
# Import required tools
########################################################################################################################

# import off-the-shelf scripting modules
from __future__ import division     # Without this, rounding errors occur in python 2.7, but apparently not in 3.4
import numpy as np
import pandas as pd
import tkinter as _tkinter
from scipy.cluster.hierarchy import dendrogram, linkage    # For cluster analysis
from scipy.cluster.hierarchy import fcluster   # For cluster membership
from matplotlib import pyplot as plt
import math

# import raw temporal data
raw_data = pd.read_csv('N2EX_clean_hourly_2018_cluster_8_outlier_removed.csv')
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

# define cutoff point for intercluster distance
cutoff = 200
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
    annotate_above=cutoff, # Prevent excessive annotation of merge distances
    max_d=cutoff
)
plt.show(block=False)

# Elbow plot, for guidance on where to truncate dendrogram

last = Z[-50:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)
plt.figure(figsize=(25, 10))
plt.title('Elbow plot')
plt.xlabel('Clusters identified')
plt.ylabel('Distance travelled to join clusters')
plt.plot(idxs, last_rev, label="Dist", marker='D')

acceleration = np.diff(last, 2)  # 2nd derivative of the distances
acceleration_rev = acceleration[::-1]
plt.plot(idxs[:-2] + 1, acceleration_rev, label="2nd Deriv Dist", marker='.')
plt.legend()
plt.show(block=False)



# retrieve cluster membership for each vector
cluster_membership = fcluster(Z, cutoff, criterion='distance')
print(cluster_membership)

# retrieve the number of clusters
cluster_list = []
for i in cluster_membership:
    if i not in cluster_list:
        cluster_list = cluster_list + [i]
n = len(cluster_list)
print ("number of clusters = ", n)

# Generate a CSV for Excel plotting, where each price datum is indexed by hour then day then cluster membership
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

dataframe_output.to_csv("hierarchical_clustering_cutoff=" + str(cutoff) + "_n="+ str(n) + ".csv", sep=',')

# Plot clusters by making price v period set for each then doing 2d histogram
plt.figure(figsize=(25, 10))
for i in [i+1 for i in range(n)]:
    n_list_period = []  # Receptacle for period data where cluster = n
    n_list_price = []   # Receptacle for price data where cluster = n
    for h in range(len(hour_index)):
        if cluster_index[h] == i:
            n_list_period = n_list_period + [hour_index[h]]
            n_list_price = n_list_price + [price_data[h]]
    # next two lines insert a day of price = zero, so that histograms all have the same bottom
    n_list_period_normalised = n_list_period + [i+1 for i in range(24)]
    n_list_price_normalised = n_list_price + [0 for i in range(24)]
    # and same so that the non-extreme days have the same top
    n_list_period_normalised = n_list_period_normalised + [i+1 for i in range(24)]
    n_list_price_normalised = n_list_price_normalised + [120 for i in range(24)]

    plt.subplot(math.ceil(math.sqrt(n)),math.ceil(n/2),i) # Define array for subplots.
    plt.title('Cluster '+str(i))
    plt.hist2d(n_list_period, n_list_price, bins=40)
    plt.xlabel('Hour of day')
    plt.ylabel('Day-ahead electrical price (£.kWh-1)')
plt.show()


# Plot clusters by making price v period set for each then doing 2d histogram
plt.figure(figsize=(25, 10))
for i in [i + 1 for i in range(n)]:
    n_list_period = []  # Receptacle for period data where cluster = n
    n_list_price = []  # Receptacle for price data where cluster = n
    for h in range(len(hour_index)):
        if cluster_index[h] == i:
            n_list_period = n_list_period + [hour_index[h]]
            n_list_price = n_list_price + [price_data[h]]
    # next two lines insert a day of price = zero, so that histograms all have the same bottom
    n_list_period = n_list_period + [i + 1 for i in range(24)]
    n_list_price = n_list_price + [0 for i in range(24)]
    # and same so that the non-extreme days have the same top
    n_list_period = n_list_period + [i + 1 for i in range(24)]
    n_list_price = n_list_price + [120 for i in range(24)]

    plt.subplot(math.ceil(math.sqrt(n)), math.ceil(n / 2), i)  # Define array for subplots.
    plt.title('Cluster ' + str(i))
    plt.hist2d(n_list_period, n_list_price, bins=40)
    plt.xlabel('Hour of day')
    plt.ylabel('Cluster average day ahead electrical price (£.kWh-1)')
plt.show()

# CSV for plotting average profile for each cluster
price_data = pd.DataFrame(input_array)
clusters = pd.DataFrame(cluster_membership)
clusters_for_means = pd.concat([clusters, price_data], axis=1)
print(clusters_for_means)

clusters_for_means.to_csv("clusters_for_means=" + str(cutoff) + "_n="+ str(n) + ".csv", sep=',')

# Generate a csv showing how clusters fit on a calendar
day=1
calendar_view = pd.DataFrame(columns=("mon", "tue", "wed", "thur", "fri", "sat", "sun"))
while day < len(cluster_membership):
    counter = 0
    remaining_data = cluster_membership[day-1:]
    week_list = []
    while counter <= 6:

        week_list= week_list + [remaining_data[counter]]
        counter = counter + 1
    calendar_view = calendar_view.append([week_list])
    day = day + counter

calendar_view.to_csv("calendar_view=" + str(cutoff) + "_n="+ str(n) + ".csv", sep=',')