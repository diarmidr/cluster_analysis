# coding: utf-8
"""This script performs a k-means cluster analysis on vectors presented in an arrray.
Code by Diarmid Roberts and Aaron Yeardley"""
########################################################################################################################
# Import required tools
########################################################################################################################

# import off-the-shelf scripting modules
from __future__ import division     # Without this, rounding errors occur in python 2.7, but apparently not in 3.4
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage    # For cluster analysis
from scipy.cluster.hierarchy import fcluster   # For cluster membership
from sklearn.cluster import KMeans
from sklearn import metrics
from matplotlib import pyplot as plt
import math

df = pd.read_csv("UnS_Train_Data.csv", header=[0, 1], index_col=0)  # read the source data file and state the header rows and the index col.
X = df.copy()  # copy the source file so that the tidying of data that needs clustering doesn't affect the original file.
X.drop(X.columns[[-1]], axis=1, inplace=True)  # drop the output as that doesn't need clustering
for i in range(5):
    X.drop(X.columns[[+1]], axis=1, inplace=True)  # drop the date cols

def standardise(dataset):
    dataNorm = ((dataset - dataset.mean()) / dataset.std())
    # dataNorm[5] = dataset[5] # this may eventually be implemented so that some columns are not standardised.
    return dataNorm
X = standardise(X)  # standardise the input values using the above function

# From here to line 49 performs clustering over a range of k to find knee point.
# index = np.arange(1,31)
# inertia_series = []
# for k in index:
#     # Run K-means cluster analysis ten times and return lowest residual SS version (SS of distances from point to cluster
#     # centroid). Output is the cluster membership for each point (row in X)
#     kmc = KMeans(n_clusters=k, init='random', n_init=5, max_iter=300, tol=1e-04, random_state=0)
#     kmc.fit_predict(X)
#     inertia_series += [kmc.inertia_]
#
#
# fig, ax = plt.subplots()
# ax.plot(index, inertia_series, label='Unexplained variance')
# accel = np.diff(inertia_series,2) # 2nd deriv of reversed inertia series
# #print(accel)
# ax.plot(index[:-2] + 1, accel * 10, label='Acceleration X 10') # Only n-2 acceleration points, so modify index accordingly.
# ax.set_xlabel('Number of clusters')
# ax.set_xticks(index)
# ax.legend()
# plt.show()

# This bit outputs cluster labels for each vector based on a manually chosen k
k = 3
kmc = KMeans(n_clusters=k, init='random', n_init=5, max_iter=300, tol=1e-04, random_state=0)
kmc.fit_predict(X)
membership = kmc.labels_

# This is an alternative elbow plot test DR was working on, where the acceleration in unexplained variance on going from
# k+1 to k clusters is compared to the population of accelerations in all the preceeding incremements to k, to see if
# the latest jump is an outlier
# Outlier test on most recent acceleration point compared to population so far
# This test is predicated on the assumption that there will be none of the clusters identified at n>15 will be genuine
# accel_rev = accel[::-1] # Reverse the series so we start from the 'linear' end.
# for i in range(len(accel_rev)):
#     if i >= 9:  # Get a population of 10 before starting (assumes no genuine clusters in n:n-10 range)
#
#         std_dev = np.std(accel_rev[i-9:i+1], ddof=1)  # Get std dev of population up to and inc current point
#         mean = np.mean(accel_rev[i-9:i+1])
#         if accel_rev[i] > mean + 3 * std_dev:
#             print(accel_rev[i-9:i+1])
#             print(i)
#             print ('Outlying increase to inertia acceleration detected at ' + str(len(accel_rev) - i + 1) + ' clusters')
#             break
#         elif accel_rev[i] < mean - 3 * std_dev:
#             print ('Outlying decrease to inertia acceleration detected at ' + str(len(accel_rev) - i + 1) + ' clusters')
#             break
# accel_rev = pd.DataFrame(accel_rev)
# accel_rev.to_csv('accel_rev.csv')


