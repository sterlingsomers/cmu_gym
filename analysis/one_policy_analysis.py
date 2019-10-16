import os.path
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn import manifold
from scipy.stats import entropy

use_sterling = False
mapw = maph = 20
''' Dropping/Crash locations heatmap '''
def event_heatmap(obs, img):
    counts = np.zeros([mapw,maph])
    for i in range(len(obs)):
        x = obs[i][0]
        y = obs[i][1]
        counts[x,y] = counts[x,y] + 1
    counts = counts / counts.max() # You want the most visited cell to have value 1. If you divide with counts.sum(axis)

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

def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    """
    Specifically, the Kullback–Leibler divergence from Q to P, denoted DKL(P‖Q), is
    a measure of the information gained when one revises one's beliefs from the
    prior probability distribution Q to the posterior probability distribution P. In
    other words, it is the amount of information lost when Q is used to approximate
    P.
    """

    return entropy(p, q)

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
    pickle_in = open(path + '.tj','rb')
    obs = pickle.load(pickle_in)
    return obs

# For Sterling's data
def create_dataframe_trials(obs, value_feature_map):
    data = []
    for trial in range(len(obs)):
        print('TRIAL:', trial)
        for epis in range(len(obs[trial])-1): # -1 because we include the map
            print('EPISODE:',epis)
            # Get position of the flag
            # indx = np.nonzero(obs[epis]['flag'])[0][0] # find the timestep where you switch to DRPPING AGENT !!!
            # print('Switch to drop_agent happened at timestep=',indx)
            epis_length = obs[trial][epis]['flag'].__len__()
            # flag2 = 0 # Indicate that drop agent is in charge (the flag=1 happens only once and then everything else is 0)
            for timestep in range(epis_length):
                print(' ---> timestep:', timestep)
                hiker=0
                ''' TRIALS '''
                trials = trial
                ''' EPISODES '''
                episode = epis
                ''' TIMESTEPS '''
                tstep = timestep
                ''' FLAG (BOOLEAN) '''
                flag = obs[trial][epis]['flag'][timestep]
                ''' AGENT TYPE (STRING) '''
                agent_type = 'one_policy'
                ''' ACTIONS (NUMERIC) '''
                actions = obs[trial][epis]['actions'][timestep]
                ''' ACTIONS NAME (STRING) '''
                action_label = act_label(actions)
                ''' ACTIONS PROB DISTR (NUMPY VEC) '''
                action_dstr = obs[trial][epis]['action_probs'][timestep]
                ''' ACTR ACTIONS PROB DISTR (NUMPY VEC) '''
                actr_action_dstr = obs[trial][epis]['actr_actions'][timestep]
                ''' reward '''
                reward = round(obs[trial][epis]['rewards'][timestep],2)
                ''' VALUES '''
                values = round(obs[trial][epis]['values'][timestep],2)
                ''' DRONE X,Y POSITION (NUMPY) '''
                drone_pos = np.array(obs[trial][epis]['drone_pos'][timestep]).transpose()[0][-2:]
                ''' DRONE Z ALTITUDE (INT) '''
                drone_alt = np.array(obs[trial][epis]['drone_pos'][timestep]).transpose()[0][0]
                ''' DRONE HEADING (INT) '''
                headings = obs[trial][epis]['headings'][timestep]
                ''' DRONE CRASH (BOOLEAN) '''
                crash = obs[trial][epis]['crash'][timestep]
                # ''' DRONE CRASH EPIS FLAG (BOOLEAN) ''' NO CAUZ WE CARE ABOUT LOCATIONS OF CRASH -- MAYBE DOESNT MATTER WE CARE ONLY IF THE DRONE CRASHED IN A LOCATION, HOWEVER YOU WONT KNOW WHERE IF YOU DONT KMOW THE TIMESTEP HTAT IT CRASHED
                # crash = obs[epis]['crash']
                ''' HIKER X,Y POSITION (NUMPY) '''
                hiker_pos = np.array(obs[trial][epis]['hiker_pos']).transpose()[0][-2:]
                ''' PACK X,Y POSITION (NUMPY) '''
                pack_pos = np.array(obs[trial][epis]['pack position'])
                ''' PACK-HIKER DISTANCE (INT) '''
                packhiker_dist = obs[trial][epis]['pack-hiker_dist']
                ''' PACK CONDITION (STRING) '''
                pack_condition = obs[trial][epis]['pack condition']
                ''' STUCK EPIS (BOOLEAN) '''
                stuck = obs[trial][epis]['stuck_epis']
                ''' FC (VECTOR) '''
                fc_rep = obs[trial][epis]['fc'][timestep]
                ''' ALTS IN COLUMNS (VECTOR) '''
                # For slice we want only the bottom of the 5x5 slice as that indicates the alt of the objects. Alt is projected upwards.
                slice_rep = obs[trial][epis]['ego'][timestep]
                slice_rep = slice_rep.astype(int)
                slice_rep_bottom = slice_rep[-1:][0]
                alts = [int(value_feature_map[j]['alt']) for j in slice_rep_bottom]

                # If hiker (feature value = 50) exists in the slice
                if 50 in slice_rep_bottom:
                    hiker=1 # Is hiker in the current egocentric input?

                data.append([trials, episode, tstep, flag, agent_type, actions, action_label, action_dstr, actr_action_dstr, round(reward,3),
                             round(values,3), drone_pos, drone_alt, headings, crash, hiker_pos, pack_pos, packhiker_dist,
                             pack_condition, stuck, fc_rep, alts, hiker])

    # Construct dataframe
    data = np.array(data, dtype=object)  # object helps to keep arbitary type of data
    """ KEEP THE SAME ORDER BETWEEN COLUMNS AND DATA (data.append and columns=[] lines)!!!"""
    columns = ['trial', 'episode', 'timestep', 'epis_ends', 'agent_type', 'actions', 'action_label', 'action_dstr', 'actr_action_dstr','rewards', 'values',
               'drone_position', 'drone_alt', 'heading', 'crash', 'hiker', 'pack_position', 'packhiker_dist',
               'pack_condition', 'stuck', 'fc', 'altitudes_in_slice', 'hiker_in_ego']

    # datab = np.reshape(data, [data.shape[0], data.shape[1]])
    df = pd.DataFrame(data, columns=columns)
    df.to_pickle(path + '.df')
    print('...dataframe saved')
    # To load
    # df = pd.read_pickle('/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/df_dataframe.df')

def create_dataframe(obs, value_feature_map):
    data = []
    for epis in range(len(obs)): # -1 because we include the map
        print('EPISODE:',epis)
        # Get position of the flag
        # indx = np.nonzero(obs[epis]['flag'])[0][0] # find the timestep where you switch to DRPPING AGENT !!!
        # print('Switch to drop_agent happened at timestep=',indx)
        epis_length = obs[epis]['flag'].__len__()
        # flag2 = 0 # Indicate that drop agent is in charge (the flag=1 happens only once and then everything else is 0)
        for timestep in range(epis_length):
            print(' ---> timestep:', timestep)
            hiker=0
            ''' MAP NAME '''
            map_name = obs[epis]['map_name']
            ''' EPISODES '''
            episode = epis
            ''' TIMESTEPS '''
            tstep = timestep
            ''' FLAG (BOOLEAN) '''
            flag = obs[epis]['flag'][timestep]
            ''' AGENT TYPE (STRING) '''
            agent_type = 'one_policy'
            ''' ACTIONS (NUMERIC) '''
            actions = obs[epis]['actions'][timestep]
            ''' ACTIONS NAME (STRING) '''
            action_label = act_label(actions)
            ''' ACTIONS PROB DISTR (NUMPY VEC) '''
            action_dstr = obs[epis]['action_probs'][timestep]
            ''' reward '''
            reward = round(obs[epis]['rewards'][timestep],2)
            ''' VALUES '''
            values = round(obs[epis]['values'][timestep],2)
            ''' DRONE X,Y POSITION (NUMPY) '''
            drone_pos = np.array(obs[epis]['drone_pos'][timestep]).transpose()[0][-2:]
            ''' DRONE Z ALTITUDE (INT) '''
            drone_alt = np.array(obs[epis]['drone_pos'][timestep]).transpose()[0][0]
            ''' DRONE HEADING (INT) '''
            headings = obs[epis]['headings'][timestep]
            ''' DRONE CRASH (BOOLEAN) '''
            crash = obs[epis]['crash'][timestep]
            # ''' DRONE CRASH EPIS FLAG (BOOLEAN) ''' NO CAUZ WE CARE ABOUT LOCATIONS OF CRASH -- MAYBE DOESNT MATTER WE CARE ONLY IF THE DRONE CRASHED IN A LOCATION, HOWEVER YOU WONT KNOW WHERE IF YOU DONT KMOW THE TIMESTEP HTAT IT CRASHED
            # crash = obs[epis]['crash']
            ''' HIKER X,Y POSITION (NUMPY) '''
            hiker_pos = np.array(obs[epis]['hiker_pos']).transpose()[0][-2:]
            ''' PACK X,Y POSITION (NUMPY) '''
            pack_pos = np.array(obs[epis]['pack position'])
            ''' PACK-HIKER DISTANCE (INT) '''
            packhiker_dist = obs[epis]['pack-hiker_dist']
            ''' PACK CONDITION (STRING) '''
            pack_condition = obs[epis]['pack condition']
            ''' STUCK EPIS (BOOLEAN) '''
            stuck = obs[epis]['stuck_epis']
            ''' FC (VECTOR) '''
            fc_rep = obs[epis]['fc'][timestep]
            ''' ALTS IN COLUMNS (VECTOR) '''
            # For slice we want only the bottom of the 5x5 slice as that indicates the alt of the objects. Alt is projected upwards.
            slice_rep = obs[epis]['ego'][timestep]
            slice_rep = slice_rep.astype(int)
            slice_rep_bottom = slice_rep[-1:][0]
            alts = [int(value_feature_map[j]['alt']) for j in slice_rep_bottom]

            ''' IS HIKER IN EGO '''
            # If hiker (feature value = 50) exists in the slice
            if 50 in slice_rep_bottom:
                hiker=1 # Is hiker in the current egocentric input?

            ''' MAP VOL (NUMPY) (5,20,20) ''' # for  ego representation graphics
            map_vol = obs[epis]['map_volume'][timestep]['vol']

            ''' MAP IMG (NUMPY) (20,20,3) ''' # for  ego representation graphics
            map_img = obs[epis]['map_volume'][timestep]['img']

            data.append([map_name, episode, tstep, flag, agent_type, actions, action_label, action_dstr, round(reward,3),
                         round(values,3), drone_pos, drone_alt, headings, crash, hiker_pos, pack_pos, packhiker_dist,
                         pack_condition, stuck, fc_rep, alts, hiker, map_vol, map_img])

    # Construct dataframe
    data = np.array(data, dtype=object)  # object helps to keep arbitary type of data
    """ KEEP THE SAME ORDER BETWEEN COLUMNS AND DATA (data.append and columns=[] lines)!!!"""
    columns = ['map_name', 'episode', 'timestep', 'epis_ends', 'agent_type', 'actions', 'action_label', 'action_dstr','rewards', 'values',
               'drone_position', 'drone_alt', 'heading', 'crash', 'hiker', 'pack_position', 'packhiker_dist',
               'pack_condition', 'stuck', 'fc', 'altitudes_in_slice', 'hiker_in_ego', 'map_volume', 'map_img']

    #TODO: Optional, load Tensorboard TSNE data and stack them to the dataframe!!!
    # datab = np.reshape(data, [data.shape[0], data.shape[1]])
    df = pd.DataFrame(data, columns=columns)
    df.to_pickle(path + '.df')
    print('...dataframe saved')
    # To load
    # df = pd.read_pickle('/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/df_dataframe.df')

folder = '/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/'
map_name = 'BoxCanyon'
drone_init_loc = 'D1118'
hiker_loc = 'H1010'
# path = folder + map_name + '_' + drone_init_loc + '_' + hiker_loc + '_' + '200'

path ='/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/all_data'

obs = load_data()
''' Create Pandas Dataframe if it doesn't exist'''
if os.path.isfile(path + '.df')==False:
    value_feature = load_feat2value()
    if use_sterling:
        create_dataframe_trials(obs,value_feature)
    else:
        create_dataframe(obs, value_feature)
if use_sterling:
    img = obs[0][0]['map_volume'][0]['img'] # somehow it doesnt work. Youhave the name of the map though but you need drone and hiker
else:
    img = obs[0]['map_volume'][0]['img']
df = pd.read_pickle(path + '.df')
# Everything below works for one map multiple runs!!!
agent_typ = 'one_policy'
''' DROP LOCATIONS '''
print("DROPS")
data = df['drone_position'].loc[(df['agent_type'] == agent_typ) & (df['action_label'] == 'drop')]
data = data.values
event_heatmap(data, img)
''' NAV AGENT CRASHES '''
print("CRASHES")
# if not any(df.loc[(df['crash'] > 0)].values): # MEMORY CONSUMPTION, COUNTS IS FASTER!
# if not df.loc[(df['crash'] > 0)].empty:
# data = df['drone_position'].loc[(df['agent_type'] == agent_typ) & (df['crash'] > 0)]
# data = data.values
# event_heatmap(data, img)
''' NAV AGENT TRAJ '''
print("TRAJECTORIES")
data = df['drone_position'].loc[(df['agent_type'] == agent_typ) ]
data = data.values
event_heatmap(data, img)
''' NAV AGENT TRAJ (W/O CRASH AND STUCK) ''' # For crash we need to identify the epis that are crashed: df[epis].loc[crash cond] then take out these, or use smth else
print("TRAJECTORIES NO STUCK")
data = df.loc[(df['stuck'] != 1)]
data = data['drone_position'].loc[(data['agent_type'] == agent_typ) ]
data = data.values
event_heatmap(data, img)
''' KL DIVERGENCE (NET || ACTR)''' # Although non summetric cross entropy is also non symmetric and is used as a loss function to bring close an approx dstr to ground truth distribution
# We include the stuck/crash as we care about every decision
netbeh = df['action_dstr'].values
# netbeh = np.round(netbeh,3)
actrbeh = df['actr_action_dstr'].values
# actrbeh = np.vstack(actrbeh)

actrbeh = np.concatenate(actrbeh,axis=0) #np.vstack(actrbeh)
netbeh = np.concatenate(netbeh,axis=0)
KLbeh = [entropy(x + 1e-7,y + 1e-7) for x,y in zip(netbeh,actrbeh)]
print('KL total = ', np.array(KLbeh).sum())
'''https://scipy.github.io/devdocs/generated/scipy.spatial.distance.jensenshannon.html '''
DJS = [np.sqrt( ( entropy(x + 1e-7,y + 1e-7) + entropy(y + 1e-7,x + 1e-7) )/2 ) for x,y in zip(netbeh,actrbeh)]
print('Distance Jensen-Shannon total = ', np.array(DJS).sum())
''' TSNE '''
# data = df['fc'].values
# data = np.concatenate(data,axis=0)
# color = data['actions'].values
# tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,perplexity=100)
# X_tsne = tsne.fit_transform(data)
# plt.scatter(X_tsne[:,0],X_tsne[:,1],c=color,alpha=0.2)
# plt.show()