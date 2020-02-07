# coding: utf-8
"""This script:
 1: Imports n lists of 24 x d hourly data points
 2: for each list, normalised based on maximum element
 3: For each d, take points 24d - to 24(d+1) for each n and form list of length 24 x n
 4: Make list of lists of length 24 x n to get 24 x n X d array
 5: perform cluster analysis on this array"""
#########################
# Import required tools #
#########################

# import off-the-shelf scripting modules
from __future__ import division     # Without this, rounding errors occur in python 2.7, but apparently not in 3.4
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, is_monotonic    # For cluster analysis
from scipy.cluster.hierarchy import fcluster   # For cluster membership
from matplotlib import pyplot as plt
import tkinter as _tkinter
import math

# import raw temporal data
raw_data = pd.read_csv('all_data_for_clustering.csv', encoding='ISO-8859-1')

# Clean it and put in a big list of lists (l_o_l)
by_input_l_o_l = []
for col in ['Demand_(MW)', 'CCGT_Gen', 'Coal_Gen', 'Wind_Gen', 'Solar_Gen']:
    raw_input = raw_data[col]
    # Tidy up price data (remove commas from as-downloaded CSV)
    input = []
    for raw in raw_input:
        stripped = ''
        for c in str(raw):
            if c == ',':
                continue
            else:
                stripped = stripped + c
        input = input + [float(stripped) / 100]

    normaliser = max(input)  # May not be the most appropriate approach, as outliers squasht he rest of the data.
    normalised_input = [x / normaliser for x in input]
    by_input_l_o_l = by_input_l_o_l + [normalised_input]

# Get data ready for clustering by making an array where each row contains concatenation of 24h of each input variable
by_day_l_o_l = []
# Loop takes a day of data at a time (i.e. 24h data-points)
for d in range(int(len(by_input_l_o_l[0])/24)):
    day_row_multi_input = []
    for n in range(len(by_input_l_o_l)):
        day_row_multi_input += by_input_l_o_l[n][d*24:(d+1)*24] # Take 24h slice for nth variable and append it to row
    by_day_l_o_l = by_day_l_o_l + [day_row_multi_input]

# Clustering takes a numpy array
input_array = np.array(by_day_l_o_l)
#input_array = np.stack(input_array, axis=0)
print(input_array)

# generate linkage matrix (this does all work, after this it's just a case of defining cluster cut-off points)
Z = linkage(input_array, 'ward')

# define cutoff point for intercluster distance
cutoff = 15
# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')

# This code defines the dendrogram plot
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample index')
        plt.ylabel('Distance £/MWh')
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
# Make dendrogram plot of output
fancy_dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
    no_labels=True, 	# Suppress singleton cluster labels that clog up x -axis
    annotate_above=cutoff, # Prevent excessive annotation of merge distances
    max_d=cutoff
)
plt.show(block=False)

# Elbow plot, for guidance on where to truncate dendrogram
last = Z[-100:, 2]
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

print("idxs", idxs)
print("last_rev", last_rev)
print("accel_rev", acceleration_rev)

# retrieve cluster membership for each vector (day in our example)
cluster_membership = fcluster(Z, cutoff, criterion='distance')

# retrieve the number of clusters
cluster_list = []
for i in cluster_membership:
    if i not in cluster_list:
        cluster_list = cluster_list + [i]
n = len(cluster_list)
print ("number of clusters = ", n)

# Generate a CSV for Excel plotting, where each datum is indexed by 'hour' then day then cluster membership
hour_index =[]
day_index = []
cluster_index =[]
y_data = []
len_x_axis = len(by_day_l_o_l[0])
for d in range(len(by_day_l_o_l)):
    hour_index = hour_index + [x+1 for x in range(len_x_axis)]
    day = [d for x in input_array[d, :]]
    day_index = day_index + day
    cluster = [cluster_membership[d] for x in input_array[d, :]]
    cluster_index = cluster_index + cluster
    y = by_day_l_o_l[d]
    y_data = y_data + y

# Plot clusters by making price v period set for each then doing 2d histogram
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)  # This empty subplot is just for the common axes.
ax.set_xlabel("Hour of Day", labelpad=10)
ax.set_ylabel("Electrical Price £/MWh", labelpad=10)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

for d in [i + 1 for i in range(n)]:
    n_list_x = []  # Receptacle for period data where cluster = n
    n_list_y = []  # Receptacle for price data where cluster = n
    for h in range(len(hour_index)):
        if cluster_index[h] == i:
            n_list_x = n_list_x + [hour_index[h]]
            n_list_y = n_list_y + [y_data[h]]

    # next two lines insert an  hour of price = zero, so that histograms all have the same bottom
    n_list_x_normalised = n_list_x + [1]
    n_list_y_normalised = n_list_y + [0]

    # Code that defines grid array for cluster histograms
    if n > 3:
        fig.add_subplot(math.floor(math.sqrt(n)), math.ceil(n / 2), i)  # Define array and index for subplots.
    else:
        fig.add_subplot(2, 2, i)  # Define array and index for subplots.

    plt.title('Cluster ' + str(i))
    plt.hist2d(n_list_x_normalised, n_list_y_normalised, bins=24)

plt.show()
plt.show(block=False)

# CSV for plotting average profile for each cluster
y_DF = pd.DataFrame(input_array)
clusters = pd.DataFrame(cluster_membership)
clusters_for_means = pd.concat([clusters, y_DF], axis=1)

clusters_for_means.to_csv("clusters_for_means_cutoff=" + str(cutoff) + "_n="+ str(n) + ".csv", sep=',')

# get list of cluster membership to CSV for Aaron

clusters.to_csv("multi_input_clustering_cutoff=" + str(cutoff) + "_n="+ str(n) + ".csv", sep=',')

# Elbow and acceleration plot to CSV
# Pad acceleration plot so it fits in CSV with others (use large number so it's
#obvious that they are not part of the real dataset
acceleration_rev_pad = [1000000]
for i in range(len(acceleration_rev)):
    acceleration_rev_pad = acceleration_rev_pad + [acceleration_rev[i]]
acceleration_rev_pad = acceleration_rev_pad + [1000000]

elbow_plot = pd.DataFrame({"Index": idxs, "Distance to Merge": last_rev, "Acceleration": acceleration_rev_pad})
elbow_plot.to_csv("elbow_plot_cutoff=" + str(cutoff) + "_n="+ str(n) + ".csv", sep=',')
