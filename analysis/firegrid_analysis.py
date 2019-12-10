import os.path
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn import manifold
from scipy.stats import entropy

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
    if action == 0:
        action_label = 'up'
    elif action == 1:
        action_label = 'down'
    elif action == 2:
        action_label = 'right'
    elif action == 3:
        action_label = 'left'
    elif action == 4:
        action_label = 'stop'
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
    pickle_in = open(path + filename + '.dct','rb')
    obs = pickle.load(pickle_in)
    return obs

def create_dataframe(obs):
    data = []
    for epis in range(len(obs)):
        print('EPISODE:',epis)
        epis_length = obs[epis]['fc'].__len__()
        for timestep in range(epis_length):
            print(' ---> timestep:', timestep)
            ''' EPISODES '''
            episode = epis
            ''' TIMESTEPS '''
            tstep = timestep
            ''' AGENT TYPE (STRING) '''
            agent_type = 'suboptimal'
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
            ''' VALUES '''
            values_goal = round(obs[epis]['values_goal'][timestep],2)
            ''' VALUES '''
            values_fire = round(obs[epis]['values_fire'][timestep],2)
            ''' BURN (BOOLEAN) '''
            burn = obs[epis]['burn'][timestep]
            ''' FC (VECTOR) '''
            fc_rep = obs[epis]['fc'][timestep]
            ''' MAP IMG (NUMPY) (9,9,3) ''' # for  ego representation graphics
            map_img = obs[epis]['map'][timestep]

            data.append([episode, tstep, agent_type, actions, action_label, action_dstr, round(reward,3),
                         round(values,3), round(values_goal,3), round(values_fire,3), burn,
                         fc_rep, map_img])

    # Construct dataframe
    data = np.array(data, dtype=object)  # object helps to keep arbitary type of data
    """ KEEP THE SAME ORDER BETWEEN COLUMNS AND DATA (data.append and columns=[] lines)!!!"""
    columns = ['episode', 'timestep', 'agent_type', 'actions', 'action_label', 'action_dstr','rewards', 'values',
               'values_goal', 'values_fire', 'burn', 'fc', 'map_img']

    #TODO: Optional, load Tensorboard TSNE data and stack them to the dataframe!!!
    # datab = np.reshape(data, [data.shape[0], data.shape[1]])
    df = pd.DataFrame(data, columns=columns)
    df.to_pickle(path + filename + '.df')
    print('...dataframe saved')
    # To load
    # df = pd.read_pickle('/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/df_dataframe.df')

path = '/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/firegrid/'
filename = '2019_Nov19_time17-25'
obs = load_data()
''' Create Pandas Dataframe if it doesn't exist'''
if os.path.isfile(path + filename + '.df')==False:
        create_dataframe(obs)

df = pd.read_pickle(path + filename + '.df')


''' TSNE '''
# data = df['fc'].values
# data = np.concatenate(data,axis=0)
# color = data['actions'].values
# tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,perplexity=100)
# X_tsne = tsne.fit_transform(data)
# plt.scatter(X_tsne[:,0],X_tsne[:,1],c=color,alpha=0.2)
# plt.show()