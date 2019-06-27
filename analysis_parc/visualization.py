"""Provides visualization of trajectories in graphical form from multiple perspectives"""

import pandas as pd
import matplotlib.pyplot as plt
import plotting
import numpy as np

df = pd.read_pickle('BoxCanyon_D1910_H1010_100_df.df')

print("Columns")
print( df.columns.values )


# Some representative Episodes

DROP_OK_MOUTH_WEST_1 = 0
DROP_OK_MOUTH_WEST_2 =199
DROP_OK_MOUTH_WEST_WANDER_FIRST_1 = 130
DROP_OK_MOUTH_WEST_WANDER_FIRST_2 = 115
DROP_OK_AT_TREE_1 = 115
DROP_OK_AT_TREE_2 = 160

DROP_OK_INSIDE_WEST_1 = 9
DROP_OK_INSIDE_WEST_2 = 120

STUCK_IN_CANYON_1 = 1
STUCK_IN_CANYON_2 = 4
STUCK_IN_CANYON_3 = 110
STUCK_IN_CANYON_4 = 135

CRASH_AT_END_OF_CANYON_1 = 2
CRASH_AT_END_OF_CANYON_2 = 5
CRASH_AT_END_OF_CANYON_3 = 7
CRASH_AT_END_OF_CANYON_4 = 10

CRASH_AT_SIDE_OF_CANYON_1 = 6


# -------------------------------------------------
# Episode to analyze
# -------------------------------------------------

episode = CRASH_AT_END_OF_CANYON_1


plotting.plot_trajectory(df, episode, "Trajectory for Episode {}".format(episode))

