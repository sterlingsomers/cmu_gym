import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

mapw = maph = 20
''' Dropping/Crash locations heatmap '''
def event_heatmap(obs, img):
    counts = np.zeros([mapw,maph])
    for i in range(len(obs)):
        x = obs[i][0]
        y = obs[i][1]
        counts[x,y] = counts[x,y] + 1
    counts = counts / counts.max()

    # Plot the map
    plot = plt.subplot(111)
    plot.imshow(img) # 50x50
    extent = plot.get_xlim()+ plot.get_ylim()
    plot.imshow(counts, interpolation='catrom',cmap='jet', alpha= 0.5, extent=extent)
    plot.set_xticklabels([])
    plot.set_yticklabels([])
    plot.set_xticks([])
    plot.set_yticks([])
    plt.show()

''' Trajectories '''
def trajectories_heatmap(obs, img):
    counts = np.zeros([mapw,maph])
    for i in range(len(obs)):
        trace = np.array(obs[i]['drone_pos'][:-1]) # Take out the last one as you remain still when you drop
        for tr in trace:
            drop_pos = tr.ravel()
            x = drop_pos[1] # alt has been kept in the first element
            y = drop_pos[2]
            counts[x,y] = counts[x,y] + 1

    counts = counts / counts.max()

    # Plot the map
    plot = plt.subplot(111)
    plot.imshow(img) # 50x50
    extent = plot.get_xlim()+ plot.get_ylim()
    plot.imshow(counts, interpolation='catrom',cmap='jet', alpha= 0.5, extent=extent) # cmap='jet', 'magma'
    plot.set_xticklabels([])
    plot.set_yticklabels([])
    plot.set_xticks([])
    plot.set_yticks([])
    plt.show()


def act_label(action):
    if action == 15:
        action_label = 'drop'
    elif action == 14:
        action_label = 'up-right'
    elif action == 13:
        action_label = 'up-diag-right'
    elif action == 12:
        action_label = 'up-forward'
    elif action == 11:
        action_label = 'up-diag-left'
    elif action == 10:
        action_label = 'up-left'
    elif action == 9:
        action_label = 'right'
    elif action == 8:
        action_label = 'diag-right'
    elif action == 7:
        action_label = 'forward'
    elif action == 6:
        action_label = 'diag-left'
    elif action == 5:
        action_label = 'left'
    elif action == 4:
        action_label = 'down-right'
    elif action == 3:
        action_label = 'down-diag-right'
    elif action == 2:
        action_label = 'down-forward'
    elif action == 1:
        action_label = 'down-diag-left'
    elif action == 0:
        action_label = 'down-left'
    return action_label

# Load mapping value <--> feature
def load_feat2value():
    pickle_in = open('/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/map_volume.feats','rb')
    map_volume = pickle.load(pickle_in)
    print('map_volume loaded')
    value_feature_map = map_volume['value_feature_map']
    return value_feature_map

# Load and create the data
def load_data():
    pickle_in = open('/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/BoxCanyon_D1910_H1010_100.tj','rb')
    obs = pickle.load(pickle_in)
    return obs

def create_dataframe(obs, value_feature_map):
    data = []
    for epis in range(len(obs)):
        print('EPISODE:',epis)
        # Get position of the flag
        indx = np.nonzero(obs[epis]['flag'])[0][0] # find the timestep where you switch to DRPPING AGENT !!!
        print('Switch to drop_agent happened at timestep=',indx)
        epis_length = obs[epis]['flag'].__len__() - 1  # We take out the last obs as the drone has dropped
        flag2 = 0 # Indicate that drop agent is in charge (the flag=1 happens only once and then everything else is 0)
        for timestep in range(epis_length):
            print(' ---> timestep:', timestep)
            hiker=0
            ''' EPISODES '''
            episode = epis
            # Each agent has separate timestep counts
            if obs[epis]['flag'][timestep] == 1:
                flag2=1 # and stays 1
            ''' TIMESTEPS '''
            if flag2 == 1:
                tstep = timestep - indx
            else:
                tstep = timestep
            ''' FLAG (BOOLEAN) '''
            flag = obs[epis]['flag'][timestep]
            ''' AGENT TYPE (STRING) '''
            agent_type = 'nav_agent' if flag2==0 else 'drop_agent'
            ''' ACTIONS (NUMERIC) '''
            actions = obs[epis]['actions'][timestep]
            ''' ACTIONS NAME (STRING) '''
            action_label = act_label(actions)
            ''' VALUES '''
            values = round(obs[epis]['values'][timestep],2)
            ''' DRONE X,Y POSITION (NUMPY) '''
            drone_pos = np.array(obs[epis]['drone_pos'][timestep]).transpose()[0][-2:]
            ''' DRONE Z ALTITUDE (INT) '''
            drone_alt = np.array(obs[epis]['drone_pos'][timestep]).transpose()[0][0][0]
            ''' DRONE HEADING (INT) '''
            headings = obs[epis]['headings'][timestep]
            ''' DRONE CRASH (BOOLEAN) '''
            crash = obs[epis]['crash'][timestep]
            ''' FC (VECTOR) '''
            fc_rep = obs[epis]['fc'][timestep]
            ''' ALTS IN COLUMNS (VECTOR) '''
            # For slice we want only the bottom of the 5x5 slice as that indicates the alt of the objects. Alt is projected upwards.
            slice_rep = obs[epis]['ego'][timestep]
            slice_rep = slice_rep.astype(int)
            slice_rep_bottom = slice_rep[-1:][0]
            alts = [int(value_feature_map[j]['alt']) for j in slice_rep_bottom]

            # If hiker (feature value = 50) exists in the slice
            if 50 in slice_rep_bottom:
                hiker=1

            data.append([episode, tstep, flag, agent_type, actions, action_label, round(values,3),drone_pos, drone_alt, headings, crash, fc_rep, alts, hiker ])

    # Construct dataframe
    data = np.array(data, dtype=object)  # object helps to keep arbitary type of data

    columns = ['episode', 'timestep', 'nav_stops', 'agent_type', 'actions', 'action_label', 'values', 'drone_position',
                       'drone_alt','heading', 'crash', 'fc', 'altitudes_in_slice', 'hiker_in_ego']

    # datab = np.reshape(data, [data.shape[0], data.shape[1]])
    df = pd.DataFrame(data, columns=columns)
    df.to_pickle('/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/BoxCanyon_D1910_H1010_100_df.df')
    print('...dataframe saved')
    # To load
    # df = pd.read_pickle('/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/df_dataframe.df')

obs = load_data()
img = obs[0]['map_volume'][0]['img']
df = pd.read_pickle('/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/BoxCanyon_D1910_H1010_100_df.df')
''' DROP LOCATIONS '''
data = df['drone_position'].loc[(df['agent_type'] == 'drop_agent') & (df['action_label'] == 'drop')]
data = data.values
event_heatmap(data, img)
''' DROP AGENT CRASHES '''
data = df['drone_position'].loc[(df['agent_type'] == 'drop_agent') & (df['crash'] == 1)]
data = data.values
event_heatmap(data, img)
''' NAV AGENT CRASHES '''
data = df['drone_position'].loc[(df['agent_type'] == 'nav_agent') & (df['crash'] == 1)]
data = data.values
event_heatmap(data, img)
''' NAV AGENT TRAJ '''
data = df['drone_position'].loc[(df['agent_type'] == 'nav_agent') ]
data = data.values
event_heatmap(data, img)
''' DROP AGENT TRAJ '''
data = df['drone_position'].loc[(df['agent_type'] == 'drop_agent') ]
data = data.values
event_heatmap(data, img)

