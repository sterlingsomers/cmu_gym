import pickle
import math
import numpy as np
import os
import random



include_fc = False

all_data = pickle.load(open('all_data100.lst', "rb"))

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
action_to_category_map = {
    0: ['left','down'],
    1: ['diagonal_left','down'],
    2: ['center', 'down'],
    3: ['diagonal_right','down'],
    4: ['right', 'down'],
    5: ['left','level'],
    6: ['diagonal_left','level'],
    7: ['center', 'level'],
    8: ['diagonal_right','level'],
    9: ['right', 'level'],
    10: ['left','up'],
    11: ['diagonal_left','up'],
    12: ['center', 'up'],
    13: ['diagonal_right','up'],
    14: ['right', 'up'],
    15: ['drop']
}

def distance_to_hiker(drone_position,hiker_position):
    distance = np.linalg.norm(drone_position-hiker_position)
    return distance


def altitudes_from_egocentric_slice(egocentric_slice):
    alts = np.count_nonzero(egocentric_slice, axis=0)
    alts = [int(x) for x in alts]
    return alts


def egocentric_representation(drone_position, drone_heading, volume):
    ego_slice = np.zeros((5,5))
    column_number = 0
    for xy in possible_actions_map[drone_heading]:
        try:
            column = volume[:, int(drone_position[1]) + xy[0], int(drone_position[2]) + xy[1]]
        except IndexError:
            column = [1., 1., 1., 1., 1.]
        ego_slice[:,column_number] = column
        column_number += 1
    return np.flip(ego_slice,0)

def heading_to_hiker(drone_heading, drone_position, hiker_position):
    '''Outputs either 90Left, 45Left, 0, 45Right,90Right, or both 90Left and 90Right'''
    category_to_angle_range = {1:[0,45],2:[45,90],3:[90,135],4:[135,180],5:[180,225],6:[225,270],7:[270,315],8:[315,360]}
    category_angle = {1:0,2:45,3:90,4:135,5:180,6:225,7:270,8:315}
    drone = drone_position[-2:]
    hiker = hiker_position[-2:]
    x1, x2 = drone[-2:]
    y1, y2 = hiker[-2:]

    rads = math.atan2(y1 - x1, y2 - x2)
    deg = math.degrees(rads) + 90 - (category_angle[drone_heading])
    if deg < -180:
        deg = deg + 360
    if deg > 180:
        deg = deg - 360
    return deg




def angle_categories(angle):
    '''Values -180 to +180. Returns a fuzzy set dictionary.'''
    returndict = {'hiker_left': 0, 'hiker_diagonal_left': 0, 'hiker_center': 0, 'hiker_diagonal_right': 0, 'hiker_right': 0}
    if angle < -90:
        returndict['hiker_left'] = 1
    if angle >= -90 and angle < -60:
        returndict['hiker_left'] = abs(angle + 60) / 30.0
    if angle >= -75 and angle < -45:
        returndict['hiker_diagonal_left'] = 1 + (angle + 45) / 30
    if angle >= -45 and angle < -15:
        returndict['hiker_diagonal_left'] = abs(angle + 15) / 30.0
    if angle >= - 30 and angle < 0:
        returndict['hiker_center'] = 1 + angle/30.0
    if angle >= 0 and angle < 30:
        returndict['hiker_center'] = 1 - angle/30.0
    if angle >=15 and angle < 45:
        returndict['hiker_diagonal_right'] = (angle - 15)/30.0
    if angle >=45 and angle < 75:
        returndict['hiker_diagonal_right'] = 1 - (angle - 45)/30.0
    if angle >=60 and angle < 90:
        returndict['hiker_right'] = (angle - 60)/30.0
    if angle >=90:
        returndict['hiker_right'] = 1

    return returndict





    # t = lambda x: abs(1 - abs(x - 45 * int(x//22.5 / 2)) / 22.5)
    #
    # returndict = {'left':0,'diagonal_left':0,'center':0,'diagonal_right':0,'right':0}
    # if angle < -90:
    #     returndict['left'] = 1
    # elif angle >= -90 and angle < -67.5:
    #     returndict['left'] = t(angle)
    # elif angle >= -67.5 and angle < -22.5:
    #     returndict['diagonal_left'] = t(angle)
    # elif angle >= -22.5 and angle < 22.5:
    #     returndict['center'] = t(angle)
    # elif angle >= 22.5 and angle < 67.5:
    #     returndict['diagonal_right'] = t(angle)
    # elif angle >= 67.5 and angle < 90:
    #     returndict['right'] = t(angle)
    # else:
    #     returndict['right'] = 1
    # if angle == -180 or angle == 180:
    #     returndict['left'] = 1
    #     returndict['right'] = 1
    #
    # return returndict



def bin_chunks_by_action(allchunks):
    nav = []
    drop = []
    by_action_count = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0}
    by_action = {}
    for episode in allchunks:
        for step in episode['nav']:
            nav.append(step)
    #now nav just has all the steps, combined from episodes.
    nav = [x for x in nav if not x['stuck']]
    no_finds = False
    #first, search the entire thing and look for unique actions
    #must have n% examples of actions to be considered a regular action
    for i in range(len(nav)):
        step = nav[i]
        by_action_count[step['action']] += 1
    for step in nav:
        n = by_action_count[step['action']] / len(nav) * 100
        if n > 10:
            by_action[step['action']] = []
    #by_action now includes all actions that were chosen n% least once by the network.  Now to populate those evenly (add their index, easier for duplicate checking and memory managmenet)
    #we need to include a maximum number of example equal to the minimum viable examples
    num_actions = [by_action_count[x] for x in by_action]
    min_samples = min(num_actions)
    #shuffle nav to select at random
    random.shuffle(nav)
    while not no_finds:
        for action in by_action:
            for i in range(len(nav)):
                if len(by_action[action]) >= min_samples:
                    break
                step = nav[i]
                if step['action'] == action:
                    by_action[action].append(i)
                    continue
            #if you get here, and haven't continued, nothing was found
            no_finds = True
    #now we have all the indexes of chunks that should be added.
    index_values = []
    for key,value in by_action.items():
        index_values = index_values + value
    #now they are all gathered in one list
    all_navs = [nav[val] for val in index_values]

    return [{'nav':all_navs,'drop':[]}]#to mimic what's expected in the loop


#bin_chunks_by_action(all_data)

#each of the entries in all data is an episode, comprised of nav and drop steps.
#dumped into two types of memories, nav and drop
nav = []
drop = []

all_data = bin_chunks_by_action(all_data)

for episode in all_data:
    for step in episode['nav']:
        action_values = {'drop':0,'left': 0, 'diagonal_left': 0,
                         'center': 0, 'diagonal_right': 0, 'right': 0,
                         'up':0,'down':0,'level':0}
        #angle to hiker: negative = left, positive right
        egocentric_angle_to_hiker = heading_to_hiker(step['heading'],step['drone'],step['hiker'])
        angle_categories_to_hiker = angle_categories(egocentric_angle_to_hiker)
        egocentric_slice = egocentric_representation(step['drone'],step['heading'],step['volume'])
        #compile all that into chunks [slot, value, slot, value]
        chunk = []
        for key,value in angle_categories_to_hiker.items():
            chunk.extend([key,[key,value]])
        #need the altitudes from the slice
        altitudes = altitudes_from_egocentric_slice(egocentric_slice)
        altitudes = [x -1 for x in altitudes]
        alt = step['altitude']
        chunk.extend(['current_altitude',['current_altitude',int(alt)]])
        chunk.extend(['ego_left',['ego_left',altitudes[0] - alt],
                      'ego_diagonal_left', ['ego_diagonal_left',altitudes[1] - alt],
                      'ego_center', ['ego_center',altitudes[2] - alt],
                      'ego_diagonal_right', ['ego_diagonal_right',altitudes[3] - alt],
                      'ego_right', ['ego_right',altitudes[4] - alt]])
        chunk.extend(['type','nav'])
        #also want distance  to hiker
        chunk.extend(['distance_to_hiker',['distance_to_hiker',distance_to_hiker(np.array(step['drone']),np.array(step['hiker']))]])
        #split action into components [up, level, down, left, right, etc]
        components = action_to_category_map[step['action']]
        for component in components:
            action_values[component] = 1
        for key,value in action_values.items():
            chunk.extend([key,[key,value]])

        if include_fc:
            chunk.extend(['fc',['fc',step['fc']]])

        nav.append(chunk)
        print('step')
    for step in episode['drop']:
        action_values = {'drop': 0, 'left': 0, 'diagonal_left': 0,
                         'center': 0, 'diagonal_right': 0, 'right': 0,
                         'up':0,'down':0,'level':0}
        # angle to hiker: negative = left, positive right
        egocentric_angle_to_hiker = heading_to_hiker(step['heading'], step['drone'], step['hiker'])
        angle_categories_to_hiker = angle_categories(egocentric_angle_to_hiker)
        egocentric_slice = egocentric_representation(step['drone'], step['heading'], step['volume'])
        # compile all that into chunks [slot, value, slot, value]
        chunk = []
        for key, value in angle_categories_to_hiker.items():
            chunk.extend([key, [key,value]])
        # need the altitudes from the slice
        altitudes = altitudes_from_egocentric_slice(egocentric_slice)
        altitudes = [x - 1 for x in altitudes]
        alt = step['altitude'] #to be consistant with the numpy
        chunk.extend(['current_altitude', ['current_altitude',int(alt)]])
        chunk.extend(['ego_left', ['ego_left',altitudes[0] - alt],
                      'ego_diagonal_left', ['ego_diagonal_left',altitudes[1] - alt],
                      'ego_center', ['ego_center',altitudes[2] - alt],
                      'ego_diagonal_right',  ['ego_diagonal_right',altitudes[3] - alt],
                      'ego_right', ['ego_right',altitudes[4] - alt]])
        chunk.extend(['type', 'drop'])
        chunk.extend(['distance_to_hiker',['distance_to_hiker',distance_to_hiker(np.array(step['drone']),np.array(step['hiker']))]])
        # split action into components [up, level, down, left, right, etc]
        components = action_to_category_map[step['action']]
        for component in components:
            action_values[component] = 1
        for key, value in action_values.items():
            chunk.extend([key, [key,value]])
        drop.append(chunk)

        if include_fc:
            chunk.extend(['fc',['fc',step['fc']]])

    print("episode complete")
memory = [*nav, *drop]


#file_path = os.path.join(data_path,filename)
with open('chunks.pkl','wb') as handle:
    pickle.dump(memory,handle)

print("done.")