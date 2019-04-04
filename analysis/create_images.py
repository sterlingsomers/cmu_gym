from scipy.misc import imresize
import numpy as np
import itertools
import pickle

factor = 5

pickle_in = open('/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/obs_test.tj','rb')
obs = pickle.load(pickle_in)

# def create_nextstep_image():
#     canvas = np.zeros((5, 5, 3), dtype=np.uint8)
#     slice = np.zeros((5, 5))
#     drone_position = np.where(
#         map_volume['vol'] == map_volume['feature_value_map']['drone'][altitude]['val'])
#     drone_position_flat = [int(drone_position[1]), int(drone_position[2])]
#     # hiker_found = False
#     # hiker_point = [0, 0]
#     # hiker_background_color = None
#     column_number = 0
#     for xy in possible_actions_map[heading]:
#         try:
#             # no hiker if using original
#             column = map_volume['vol'][:, drone_position_flat[0] + xy[0], drone_position_flat[1] + xy[1]]
#
#         except IndexError:
#             column = [1., 1., 1., 1., 1.]
#         slice[:, column_number] = column
#         column_number += 1
#
#     # put the drone in
#     # cheat
#     slice[self.altitude, 2] = int(map_volume['vol'][drone_position])
#     combinations = list(itertools.product(range(0, canvas.shape[0]), range(0, canvas.shape[0])))
#     for x, y in combinations:
#         if slice[x, y] == 0.0:
#             canvas[x, y, :] = [255, 255, 255]
#         # elif slice[x, y] == 50.0:
#         #     canvas[x, y, :] = hiker_background_color
#         #     hiker_point = [x, y]
#         else:
#             canvas[x, y, :] = map_volume['value_feature_map'][slice[x, y]]['color']
#
#     # increase the image size, then put the hiker in
#     canvas = imresize(canvas, factor * 100, interp='nearest')
#
#
#     return imresize(np.flip(canvas, 0), 2 0 *map_volume['vol'].shape[2], interp='nearest')
#
# def generate_observation(self):
#     obs = {}
#     obs['volume'] = map_volume
#     # image_layers = copy.deepcopy(image_layers)
#     map = map_volume = CNP.map_to_volume_dict(_map[0],_map[1], mapw, maph)#copy.deepcopy(self.original_map_volume['img'])
#
#     # put the drone in the image layer
#     drone_position = np.where(
#         map_volume['vol'] == map_volume['feature_value_map']['drone'][altitude]['val'])
#     drone_position = (int(drone_position[1]) * factor, int(drone_position[2]) * factor)
#     # for point in self.planes[self.heading][0]:
#     #     image_layers[self.altitude][drone_position[0] + point[0], drone_position[1] + point[1], :] = \
#     #         self.map_volume['feature_value_map']['drone'][self.altitude]['color']
#
#     # put the hiker in the image layers
#     hiker_position = (int(hiker_position[1] * factor), int(hiker_position[2]) * factor)
#     # for point in self.hikers[0][0]:
#     #     image_layers[0][hiker_position[0] + point[0], hiker_position[1] + point[1], :] = \
#     #         self.map_volume['feature_value_map']['hiker']['color']
#
#     # map = self.original_map_volume['img']
#     map = imresize(map, factor * 100, interp='nearest')  # resize by factor of 5
#     # add the hiker
#     hiker_position = (int(hiker_position[1] * 5), int(hiker_position[2]) * 5)
#     # map[hiker_position[0]:hiker_position[0]+5,hiker_position[1]:hiker_position[1]+5,:] = self.hiker_image
#     for point in hikers[0][0]:
#         map[hiker_position[0] + point[0], hiker_position[1] + point[1], :] = \
#             map_volume['feature_value_map']['hiker']['color']
#     # add the drone
#     drone_position = np.where(
#         map_volume['vol'] == map_volume['feature_value_map']['drone'][altitude]['val'])
#     drone_position = (int(drone_position[1]) * 5, int(drone_position[2]) * 5)
#     for point in planes[heading][0]:
#         map[drone_position[0] + point[0], drone_position[1] + point[1], :] = \
#             map_volume['feature_value_map']['drone'][altitude]['color']
#     # map[drone_position[0]:drone_position[0] + 5,drone_position[1]:drone_position[1] + 5] = self.plane_image(self.heading,self.map_volume['feature_value_map']['drone'][self.altitude]['color'])
#
#     # map = imresize(map, (1000,1000), interp='nearest')
#
#     '''vertical slices at drone's position'''
#     drone_position = np.where(
#         map_volume['vol'] == map_volume['feature_value_map']['drone'][altitude]['val'])
#
#
#     nextstepimage = create_nextstep_image()
#     obs['nextstepimage'] = nextstepimage
#     obs['img'] = map
#     return obs

def create_image_from_volume(map_volume, altitude):
    canvas = np.zeros((map_volume['vol'].shape[1], map_volume['vol'].shape[1], 3), dtype=np.uint8)
    og_vol = map_volume #original_map_volume
    combinations = list(itertools.product(range(0, canvas.shape[0]), range(0, canvas.shape[0])))
    for x, y in combinations:
        if og_vol['vol'][altitude][x, y] == 0.0:
            canvas[x, y, :] = [255, 255, 255]
        else:
            canvas[x, y, :] = og_vol['value_feature_map'][og_vol['vol'][altitude][x, y]]['color']

    return imresize(canvas, factor * 100, interp='nearest')
# obs['altitude'][epis]
altitude = obs['drone_pos'][0][0][0]
im = create_image_from_volume(obs['map_volume'],altitude)