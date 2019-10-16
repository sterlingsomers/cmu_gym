import numpy as np
import pandas as pd
import copy
import itertools
import pickle
from scipy.misc import imresize
from gym_gridworld.envs import create_np_map as CNP
from matplotlib import pyplot as plt

pickle_in = open('/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/feat2value2feat_mapping.dct','rb')
map_volume = pickle.load(pickle_in)
file_name = 'all_data.df'# , 'df_dataframe.df'
obs = pd.read_pickle('/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/' + file_name)
map_volume['vol']=obs['map_volume'][0]
map_volume['img']=obs['map_img'][0]
map_volume['name']=obs['map_name'][0]
map_volume['orig'] = CNP.map_to_volume_dict(int(map_volume['name'][0:3]), int(map_volume['name'][4:]), 20, 20)
# map volume
# drone:altitude
# drone:heading
# hiker position
# factor = 5
# map_vol for next step image, map_img for generate obs.
# 5x5 plane descriptions
planes = {}
planes[1] = [[(0, 2), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4)], np.zeros((5, 5, 3))]
planes[2] = [[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (0, 4), (1, 3), (2, 3), (1, 2)], np.zeros((5, 5, 3))]
planes[3] = [[(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (1, 3), (2, 3), (3, 3), (2, 4)], np.zeros((5, 5, 3))]
planes[4] = [[(0, 4), (1, 3), (2, 3), (3, 3), (4, 4), (2, 2), (3, 2), (3, 1), (4, 0)], np.zeros((5, 5, 3))]
planes[5] = [[(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (4, 2)], np.zeros((5, 5, 3))]
planes[6] = [[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (2, 1), (3, 1), (3, 2), (4, 0)], np.zeros((5, 5, 3))]
planes[7] = [[(2, 0), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)], np.zeros((5, 5, 3))]
planes[8] = [[(0, 0), (4, 0), (1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (1, 3), (0, 4)], np.zeros((5, 5, 3))]
hikers = {}
hikers[0] = [[(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4)],
                  np.zeros((5, 5, 3))]
# hiker_image = np.zeros((5, 5, 3))
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

def create_nextstep_image(map_volume, drone_position, heading):
    factor = 5
    altitude = drone_position[0]
    canvas = np.zeros((5, 5, 3), dtype=np.uint8)
    slice = np.zeros((5, 5))

    drone_position_flat = [int(drone_position[1]), int(drone_position[2])]
    column_number = 0
    for xy in possible_actions_map[heading]:
        if drone_position_flat[0] + xy[0] >= 0 and drone_position_flat[1] + xy[1] >= 0 and drone_position_flat[0] + \
                xy[0] <= map_volume['vol'].shape[1] - 1 and drone_position_flat[1] + xy[1] <= \
                map_volume['vol'].shape[2] - 1:

            # try:
            # no hiker if using original
            column = map_volume['vol'][:, drone_position_flat[0] + xy[0], drone_position_flat[1] + xy[1]]

        # except IndexError:
        else:
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
    ego = np.flip(slice, 0)
    return imresize(np.flip(canvas, 0), 20 * map_volume['vol'].shape[2], interp='nearest')

def generate_observation(map_volume, altitude, heading, hiker_position, drone_position):
    factor = 5
    obs = {}
    obs['volume'] = map_volume
    map = copy.deepcopy(map_volume['orig']['img']) # "empty" map (no drone no hiker)
    # put the drone in the image layer # we need to use the drone no need for np.where
    # drone_position = np.where(
    #     map_volume['vol'] == map_volume['feature_value_map']['drone'][altitude]['val'])
    # drone_position = (int(drone_position[1]) * factor, int(drone_position[2]) * factor)
    # for point in planes[heading][0]:
    #     image_layers[altitude][drone_position[0] + point[0], drone_position[1] + point[1], :] = \
    #         map_volume['feature_value_map']['drone'][altitude]['color']

    # put the hiker in the image layers
    # hiker_position = (int(hiker_position[1] * factor), int(hiker_position[2]) * factor)
    # for point in hikers[0][0]:
    #     image_layers[0][hiker_position[0] + point[0], hiker_position[1] + point[1], :] = \
    #         map_volume['feature_value_map']['hiker']['color']

    # map = original_map_volume['img']
    map = imresize(map, factor * 100, interp='nearest')  # resize by factor of 5
    # add the hiker
    hiker_position = (int(hiker_position[0] * factor), int(hiker_position[1]) * factor)
    for point in hikers[0][0]:
        map[hiker_position[0] + point[0], hiker_position[1] + point[1], :] = \
            map_volume['feature_value_map']['hiker']['color']
    # add the drone
    drone_pos = np.where(
        map_volume['vol'] == map_volume['feature_value_map']['drone'][altitude]['val'])
    drone_position = (int(drone_pos[1]) * factor, int(drone_pos[2]) * factor)
    for point in planes[heading][0]:
        map[drone_position[0] + point[0], drone_position[1] + point[1], :] = \
            map_volume['feature_value_map']['drone'][altitude]['color']

    '''DRAW THE PACKAGE DROPPED'''
    # print('pack drop flag',package_dropped)
    # if package_dropped:
    #     package_dropped = 0
    #     package_position = (int(package_position[0] * 5), int(package_position[1]) * 5)
    #     for point in package[package_state][0]:
    #         # print(point, package_position)
    #         map[package_position[0] + point[0], package_position[1] + point[1], :] = [94, 249, 242]

    nextstepimage = create_nextstep_image(map_volume, drone_pos, heading)
    plt.imshow(np.concatenate([map, nextstepimage], axis=1))
    plt.show()

generate_observation(map_volume, obs['drone_alt'][0]-1, obs['heading'][0], obs['hiker'][0], obs['drone_position'][0])