# coding: utf-8
"""This script:
 1: Imports n lists of 24 x d hourly data points
 2: For each list, normalised based on maximum element
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

def pre_cluster_step(data, cutoff, d):
    df = pd.read_csv(data, encoding='ISO-8859-1')
    # Clean it and put in a big list of lists (l_o_l)
    by_variable_l_o_l = []
    for col in ['Demand_(MW)', 'CCGT_Gen', 'Coal_Gen', 'Wind_Gen', 'Solar_Gen']:
        raw = df[col]
        # Tidy up price data (remove commas from as-downloaded CSV)
        tidied = []
        for r in raw:
            stripped = ''
            for c in str(r):
                if c == ',':
                    continue
                else:
                    stripped = stripped + c
            tidied += [float(stripped) / 100]

        normaliser = max(tidied)  # May not be the most appropriate approach, as outliers squasht he rest of the data.
        normalised_tidied = [x / normaliser for x in tidied]
        by_variable_l_o_l += [normalised_tidied]

    # Get data ready for clustering by making an array where each row contains concatenation of 24h of each input variable
    by_day_l_o_l = []
    # Loop takes a day of data at a time (i.e. 24h data-points)
    for i in range(int(len(by_variable_l_o_l[0])/int(d))):
        day_row_multi_input = []
        for n in range(len(by_variable_l_o_l)):
            day_row_multi_input += by_variable_l_o_l[n][i*d:(i+1)*d] # Take d hour slice for nth variable and append it to row
        by_day_l_o_l = by_day_l_o_l + [day_row_multi_input]
    # Clustering takes a numpy array
    input_array = np.array(by_day_l_o_l)
    # Generate linkage matrix (this does all work, after this it's just a case of defining cluster cut-off points)
    Z = linkage(input_array, 'ward')
    '''Below we use the fcluster function to get the cluster membership. We must set a cutoff, i.e. a threshold for 
    judging a cluster to be genuinely separate. The fcluster function takes a merge distance as it it's cutoff, i.e. 
    merges with distance >= cutoff have joined two genuine clusters.'''
    last = Z[-100:, 2] # Sub array of last 100 merges
    last_rev = last[::-1] # Reverse the series
    acceleration = np.diff(last, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1] # Reverse the series
    if cutoff=='manual':
        print('manual')
        n=int(input('Select number of clusters (i.e. last n-1 merges)'))
        # Select merge distance corresponding to merge that takes us from n clusters to n-1
        cutoff = (last_rev[n-1] + last_rev[n-2])/2 # 0th entry in 'last_rev' is 1 cluster containing all vectors
    elif cutoff=='acceleration':
        # Pullout index of maximum acceleration
        max_acc = max(acceleration_rev)
        n = list(acceleration_rev).index(max_acc) + 2 # + 2 as 0th entry in acceleration_rev is merge 2 -> 1 clusters
        cutoff = (last_rev[n-1] + last_rev[n-2])/2 # 0th entry in 'last_rev' is 1 cluster containing all vectors
    # Retrieve cluster membership for each vector (day in our example)
    cluster_membership = fcluster(Z, cutoff, criterion='distance')
    # retrieve the number of clusters
    cluster_list = []
    for i in cluster_membership:
        if i not in cluster_list:
            cluster_list += [i]
    n = len(cluster_list)
    print ("number of clusters = ", n)

    return cluster_membership

# Demo call
clustering = pre_cluster_step('input_variable_data_for_clustering.csv', cutoff='acceleration', d=1)
print(clustering)
