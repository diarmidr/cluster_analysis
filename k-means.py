# coding: utf-8
"""This script performs a cluster analysis on day ahead price profiles"""
########################################################################################################################
# Import required tools
########################################################################################################################

# import off-the-shelf scripting modules
from __future__ import division     # Without this, rounding errors occur in python 2.7, but apparently not in 3.4
import pandas as pd                 # CSV handling
import numpy as np
#import SciPy                        # For cluster analysis
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
    
