# coding: utf-8
"""This script performs a cluster analysis on day ahead price profiles"""
########################################################################################################################
# Import required tools
########################################################################################################################

# import off-the-shelf scripting modules
from __future__ import division     # Without this, rounding errors occur in python 2.7, but apparently not in 3.4
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans,vq,whiten                 # For cluster analysis

# import raw temporal data
raw_data = pd.read_csv('N2EX_clean_hourly_2018.csv')
price = [float(x) for x in raw_data.ix[:, 'price_sterling']]

# Get 24h profile in list format, then stack in array.
arrays = []
for i in range(int(len(price)/24)):
    price_in_period = np.array(price[i*24:(i+1)*24])
    arrays = arrays + [price_in_period]
input_array = np.stack(arrays, axis=0)
print(input_array)

# find n clusters in the data
n = 7
centroids, distortion = kmeans(input_array, n)

print('centroids  : ',centroids)
print('distortion :',distortion)

# Assign each sample to a cluster
membership,_ = vq(input_array,centroids)
print(membership)

# CSV output

# The average values for the cluster vectors
clusters_output = pd.DataFrame()
for i in range(n):
    cluster = pd.DataFrame([centroids[i]])
    clusters_output = clusters_output.append(cluster)
clusters_output.to_csv("k-means_clusters_n=" + str(n) + ".csv", sep=',')

# For each day of data, put alongside its cluster membership
membership_col = pd.DataFrame(membership)
membership_output = pd.concat([membership_col, pd.DataFrame(input_array)], axis=1)
print(membership_output)
membership_output.to_csv("k-means_cluster_membership_n=" + str(n) + ".csv", sep=',')

