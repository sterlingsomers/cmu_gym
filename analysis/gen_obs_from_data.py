import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itertools
import pickle
from scipy.misc import imresize


'''Get the data'''
#pickle_in = open('/Users/paulsomers/COGLE/gym-gridworld/data/tree_grass_trees_100.tj','rb')
pickle_in = open('/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/All_maps_random_500.tj','rb')
obs = pickle.load(pickle_in)
possible_actions_map = {
    1: [[0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1]],
    2: [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1]],
    3: [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0]],
    4: [[-1, 1], [0, 1], [1, 1], [1, 0], [1, -1]],
    5: [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1]],
    6: [[1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]],
    7: [[1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0]],
    8: [[1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]
}

def create_nextstep_image(map_volume, altitude,heading):
    factor = 5
    canvas = np.zeros((5, 5, 3), dtype=np.uint8)
    slice = np.zeros((5, 5))
    drone_position = np.where(
        map_volume['vol'] == map_volume['feature_value_map']['drone'][altitude]['val'])
    drone_position_flat = [int(drone_position[1]), int(drone_position[2])]
    column_number = 0
    for xy in possible_actions_map[heading]:
        try:
            # no hiker if using original
            column = map_volume['vol'][:, drone_position_flat[0] + xy[0], drone_position_flat[1] + xy[1]]
        except IndexError:
            column = [1., 1., 1., 1., 1.]
        slice[:, column_number] = column
        column_number += 1
        # print("ok")
    # put the drone in
    # cheat
    slice[altitude, 2] = int(map_volume['vol'][drone_position])
    combinations = list(itertools.product(range(0, canvas.shape[0]), range(0, canvas.shape[0])))
    for x, y in combinations:
        if slice[x, y] == 0.0:
            canvas[x, y, :] = [255, 255, 255]
        else:
            canvas[x, y, :] = map_volume['value_feature_map'][slice[x, y]]['color']

    # increase the image size, then put the hiker in
    canvas = imresize(canvas, factor * 100, interp='nearest')

    return imresize(np.flip(canvas, 0), 20 * map_volume['vol'].shape[2], interp='nearest')

dict = {}

t = 0
''' Collect observations '''
for epis in range(len(obs)):
    print('Episode:',epis)
    # Get position of the flag
    indx = np.nonzero(obs[epis]['flag'])[0][0]
    print('indx=',indx)
    traj_length = obs[epis]['flag'].__len__()-1 # We take out the last obs as the drone has dropped
    print('traj=',traj_length)
    if (traj_length - indx >= 5): # collect trajectories that have multiple steps (5 or more) before dropping
        for i in range(traj_length-indx):
            sub_dict = {}
            print('iter:',i,'indx+i=', indx+i)
            sub_dict['obs'] = obs[epis]['observations'][indx+i] # first i=0
            #sub_dict['fc'] = obs[epis]['fc'][indx + i]
            if (indx+i)== (traj_length-1):
                sub_dict['target'] = 1
            else:
                sub_dict['target'] = 0
            dict[t] = sub_dict
            t = t + 1