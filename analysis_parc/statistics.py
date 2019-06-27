"""Calculates a number of high level statistics describing trajectories"""

import pandas as pd
import matplotlib.pyplot as plt
import queries

import numpy as np

df = pd.read_pickle('BoxCanyon_D1910_H1010_100_df.df')

print("Columns")
print( df.columns.values )

crashes_with_lengths =         \
    df[ [ 'episode', 'crash' ] ] \
    .groupby( 'episode' ).filter( lambda rows: rows[['crash']].sum() > 0) \
    .groupby( 'episode' ).count() \
    .rename( columns={ 'crash':'length' } ) \
    .reset_index()

episode_lengths = df[ [ 'episode','timestep' ] ] \
                   .groupby(['episode']).count() \
                   .rename(columns={'timestep':'length'}) \
                   .reset_index()

stuck_episodes = episode_lengths.loc[ episode_lengths['length'] == 70 ]






N = len( episode_lengths.index )
num_stucks = len( stuck_episodes.index )
num_crashes = len( crashes_with_lengths.index )

print("Number of episodes {}".format(N))
print("Number of crashes {} or {}%".format(num_crashes, num_crashes/N))
print("Number of stuck episodes {} or {}%".format(num_stucks, num_stucks/N))

print("Crashes")
print(crashes_with_lengths)


plt.hist([episode_lengths['length'],
          crashes_with_lengths['length']],
         label=['Episode Length','Crash Length'])

plt.title('Trajectory Lengths (Total {}, Drop OK {} Crashes {}, Stuck {} )'.format(N,N-num_stucks-num_crashes,num_crashes,num_stucks))
plt.xlabel('Trajectory Length')
plt.ylabel("Count")
plt.legend(loc='upper right')
plt.show()

