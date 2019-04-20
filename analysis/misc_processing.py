from sklearn import manifold
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import os
import pandas as pd

# Load and create the data
pickle_in = open('/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/map_volume.feats','rb')
map_volume = pickle.load(pickle_in)
print('map_volume loaded')
value_feature_map = map_volume['value_feature_map']

pickle_in = open('/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/selected_drop_traj_everything.tj','rb')
obs = pickle.load(pickle_in)
# pickle_in = open('/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/gym_gridworld/features/values_to_features.dict','rb')
# value_feature_map = pickle.load(pickle_in)

data = []
fc = []
slice = []
for i in range(len(obs)):
    hiker=0
    epis = obs[i]['episode']
    tstep = obs[i]['timestep']
    target = obs[i]['target']
    actions = obs[i]['actions']
    values = obs[i]['values']
    action_label = obs[i]['action_label']
    fc_rep = obs[i]['fc']
    # For slice we want only the bottom of the 5x5 slice as that indicates the alt of the objects. Alt is projected upwards.
    slice_rep = obs[i]['ego_vec']
    slice_rep = slice_rep.astype(int)
    slice_rep_bottom = slice_rep[-1:][0]
    alts = [int(value_feature_map[i]['alt']) for i in slice_rep_bottom]
    # print(alts)
    # If hiker (feature value = 50) exists in the slice
    if 50 in slice_rep_bottom:
        hiker=1

    drone_alt = obs[i]['drone_pos'][0][0]
    crash = obs[i]['crash']

    fc.append(obs[i]['fc'])
    slice.append(obs[i]['ego_vec'])
    data.append([epis, tstep, target, actions, round(values,3), action_label,fc_rep, alts, hiker, drone_alt, crash]) # round doesnt work in writing to a file
data = np.array(data,dtype=object) # object helps to keep arbitary type of data
columns = ['episode','timestep','target','actions','values','action_label','fc', 'altitudes', 'hiker_in_ego', 'drone_alt', 'crash']
datab = np.reshape(data,[data.shape[0],data.shape[1]])
df = pd.DataFrame(data,columns=columns)
df.to_pickle('/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/df_dataframe.df')
print('...dataframe saved')
# To load
# df = pd.read_pickle('/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/df_dataframe.df')

