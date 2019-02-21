"""This script is for data analysis, and imports daily average temperature data from CSV then returns a CSV where each
week is a row."""

# import off-the-shelf modules
from __future__ import division     # Without this, rounding errors occur in python 2.7, but apparently not in 3.4
import numpy as np
import pandas as pd

# import raw temporal data
raw_data = pd.read_csv('temp_daily_2018.csv')
temp = [float(x) for x in raw_data.ix[:, 'low']]

# Generate a csv showing how clusters fit on a calendar
day=1
calendar_view = pd.DataFrame(columns=("mon", "tue", "wed", "thur", "fri", "sat", "sun"))
while day < len(temp):
    counter = 0
    remaining_data = temp[day-1:]
    week_list = []
    while counter <= 6:
        week_list= week_list + [remaining_data[counter]]
        counter = counter + 1
    calendar_view = calendar_view.append([week_list])
    day = day + counter

calendar_view.to_csv("calendar_view_temp_2018.csv", sep=',')