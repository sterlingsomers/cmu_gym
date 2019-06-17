import pickle
import math
import numpy as np
import os
import random
import copy

from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt

from pandas import DataFrame




include_fc = True

all_data = pickle.load(open('all_data2000.lst', "rb"))

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
actions = ['_'.join(x) if len(x) > 1 else x[0] for x in action_to_category_map.values()]



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


def convert_data_to_chunks(all_data):
    nav = []
    drop = []
    for episode in all_data:
        for step in episode['nav']:
            action_values = {'drop': 0, 'left': 0, 'diagonal_left': 0,
                             'center': 0, 'diagonal_right': 0, 'right': 0,
                             'up': 0, 'down': 0, 'level': 0}
            # angle to hiker: negative = left, positive right
            egocentric_angle_to_hiker = heading_to_hiker(step['heading'], step['drone'], step['hiker'])
            angle_categories_to_hiker = angle_categories(egocentric_angle_to_hiker)
            egocentric_slice = egocentric_representation(step['drone'], step['heading'], step['volume'])
            # compile all that into chunks [slot, value, slot, value]
            chunk = []
            for key, value in angle_categories_to_hiker.items():
                chunk.extend([key, [key, value]])
            # need the altitudes from the slice
            altitudes = altitudes_from_egocentric_slice(egocentric_slice)
            altitudes = [x - 1 for x in altitudes]
            alt = step['altitude']
            chunk.extend(['altitude', ['altitude', int(alt)]])
            chunk.extend(['ego_left', ['ego_left', altitudes[0] - alt],
                          'ego_diagonal_left', ['ego_diagonal_left', altitudes[1] - alt],
                          'ego_center', ['ego_center', altitudes[2] - alt],
                          'ego_diagonal_right', ['ego_diagonal_right', altitudes[3] - alt],
                          'ego_right', ['ego_right', altitudes[4] - alt]])



            # also want distance  to hiker
            chunk.extend(['distance_to_hiker',
                          ['distance_to_hiker', distance_to_hiker(np.array(step['drone']), np.array(step['hiker']))]])
            # split action into components [up, level, down, left, right, etc]
            # components = action_to_category_map[step['action']]
            # for component in components:
            #     action_values[component] = 1
            # for key, value in action_values.items():
            #     chunk.extend([key, [key, value]])

            #last part of the observation side will be the vector
            if include_fc:
                fc_list = step['fc'].tolist()[0]
                chunk.extend(['fc', ['fc', fc_list]])

            #no longer splitting the actions. Use all 15
            #actr_actions = ['_'.join(x) if len(x) > 1 else x for x in action_to_category_map.values()]
            for key,value in action_to_category_map.items():
                if len(value) > 1:
                    actr_action = '_'.join(value)
                else:
                    actr_action = value[0]
                if key == step['action']:
                    chunk.extend([actr_action,[actr_action,1]])
                else:
                    chunk.extend([actr_action,[actr_action,0]])
            chunk.extend(['type', 'nav'])




            nav.append(chunk)
            print('step')
        for step in episode['drop']:
            action_values = {'drop': 0, 'left': 0, 'diagonal_left': 0,
                             'center': 0, 'diagonal_right': 0, 'right': 0,
                             'up': 0, 'down': 0, 'level': 0}
            # angle to hiker: negative = left, positive right
            egocentric_angle_to_hiker = heading_to_hiker(step['heading'], step['drone'], step['hiker'])
            angle_categories_to_hiker = angle_categories(egocentric_angle_to_hiker)
            egocentric_slice = egocentric_representation(step['drone'], step['heading'], step['volume'])
            # compile all that into chunks [slot, value, slot, value]
            chunk = []
            for key, value in angle_categories_to_hiker.items():
                chunk.extend([key, [key, value]])
            # need the altitudes from the slice
            altitudes = altitudes_from_egocentric_slice(egocentric_slice)
            altitudes = [x - 1 for x in altitudes]
            alt = step['altitude']  # to be consistant with the numpy
            chunk.extend(['altitude', ['altitude', int(alt)]])
            chunk.extend(['ego_left', ['ego_left', altitudes[0] - alt],
                          'ego_diagonal_left', ['ego_diagonal_left', altitudes[1] - alt],
                          'ego_center', ['ego_center', altitudes[2] - alt],
                          'ego_diagonal_right', ['ego_diagonal_right', altitudes[3] - alt],
                          'ego_right', ['ego_right', altitudes[4] - alt]])
            chunk.extend(['type', 'drop'])
            chunk.extend(['distance_to_hiker',
                          ['distance_to_hiker', distance_to_hiker(np.array(step['drone']), np.array(step['hiker']))]])
            # split action into components [up, level, down, left, right, etc]
            # components = action_to_category_map[step['action']]
            # for component in components:
            #     action_values[component] = 1
            # for key, value in action_values.items():
            #     chunk.extend([key, [key, value]])
            # drop.append(chunk)

            # no longer splitting the actions. Use all 15
            # actr_actions = ['_'.join(x) if len(x) > 1 else x for x in action_to_category_map.values()]
            for key, value in action_to_category_map.items():
                if len(value) > 1:
                    actr_action = '_'.join(value)
                else:
                    actr_action = value[0]
                if key == step['action']:
                    chunk.extend([actr_action, [actr_action, 1]])
                else:
                    chunk.extend([actr_action, [actr_action, 0]])

            # if include_fc:
            #     chunk.extend(['fc', ['fc', step['fc']]])

        print("episode complete")
    memory = [nav, drop]
    return memory

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

def average_distance_between_vectors(numpyArray):
    tot = 0

    for i in range(numpyArray.shape[0]-1):
        tot += ((((numpyArray[i+1:])**2).sum(1))**.5).sum()
    average = tot/((numpyArray.shape[0]-1)*(numpyArray.shape[0])/2.)
    return average


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

def convert_data_to_ndarray(data):
    '''Converts the list of steps into ndarray'''
    ndarray_size = 12
    if include_fc:
        ndarray_size += 256
    ndarray = np.zeros((len(data),ndarray_size))
    labels = []
    for d in range(len(data)):
        astep = []
        value = None
        for i in range(1,len(data[d]),2):
            value = None
            try:
                value = float(data[d][i][1])
            except ValueError:
                pass
            except TypeError:
                #i expect the TypeError when it's a list
                value = data[d][i][1]
            if not value == None:
                astep.append(value)
                if data[d][i][0] not in labels:
                    labels.append(data[d][i][0])
        ndarray[d][:12] = astep[:12]
        if include_fc:
            ndarray[d][12:] = [x for x in astep[12]]

    return ndarray,labels

def pick_two_from_each_group(dict_by_action,dist,step=0.1):
    #first, remove all empty actions
    new_dict = {}
    combos_to_actions = {('down', 'left'): [], ('down', 'diagonal_left'): [], ('down', 'center'): [],
                         ('down', 'diagonal_right'): [], ('down', 'right'): [],
                         ('level', 'left'): [], ('level', 'diagonal_left'): [], ('level', 'center'): [],
                         ('level', 'diagonal_right'): [], ('level', 'right'): [],
                         ('up', 'left'): [], ('up', 'diagonal_left'): [], ('up', 'center'): [],
                         ('up', 'diagonal_right'): [], ('up', 'right'): [], ('drop'): []}
    #don't copy the dict, edit it in-i
    # for key,value in dict_by_action.items():
    #     if value:
    #         new_dict[key] = [x for x in value]
    for key in dict_by_action:
        chunks = dict_by_action[key]
        if not chunks:
            continue
        random.shuffle(chunks)
        #pick one randomly
        achunk = chunks[0]
        remove_index = 0
        combos_to_actions[key].append(chunks)
        for i in range(1,len(chunks)):
            if euclidean_between_chunks(chunks[i],achunk) > dist:
                remove_index = i
                combos_to_actions[key].append(chunks[i])
                break
        if remove_index:
            del combos_to_actions[key][remove_index]


    return combos_to_actions

def index_of_most_distal_chunks(chunksList):
    distance = 0
    indexes = []
    for i in range(len(chunksList)):
        for j in range(i + 1, len(chunksList)):
            if euclidean_between_chunks(chunksList[i],chunksList[j]) > distance:
                distance = euclidean_between_chunks(chunksList[i],chunksList[j])
                c1 = chunksList[i]
                c2 = chunksList[j]
                indexes = [i,j+i]
    return indexes,distance



def random_distal_chunks(dict_by_action):
    combos_to_actions = {('down', 'left'): [], ('down', 'diagonal_left'): [], ('down', 'center'): [],
                         ('down', 'diagonal_right'): [], ('down', 'right'): [],
                         ('level', 'left'): [], ('level', 'diagonal_left'): [], ('level', 'center'): [],
                         ('level', 'diagonal_right'): [], ('level', 'right'): [],
                         ('up', 'left'): [], ('up', 'diagonal_left'): [], ('up', 'center'): [],
                         ('up', 'diagonal_right'): [], ('up', 'right'): [], ('drop'): []}
    change = False
    for key in dict_by_action:
        greater_than_average = []
        chunks = copy.deepcopy(dict_by_action[key])
        if not chunks:
            continue
        min_distance = average_euclidean(chunks)
        random.shuffle(chunks)
        achunk = chunks[0]
        remove_index = 0#can't be a chosen index, since it starts at 1, so use as T/F
        for i in range(1,len(chunks)):
            if euclidean_between_chunks(achunk,chunks[i]) > min_distance:
                change = True
                remove_index = i
                combos_to_actions[key].append(chunks[i])
                combos_to_actions[key].append(achunk)
                break
        if remove_index:
            del dict_by_action[key][remove_index]

    return combos_to_actions, change


def virtual_center(bin):
    observation_dimensions = {'hiker_left':[],'hiker_diagonal_left':[],'hiker_center':[],
                        'hiker_diagonal_right':[],'hiker_right':[],'altitude':[],
                        'ego_left':[], 'ego_diagonal_left':[], 'ego_center':[],
                        'ego_diagonal_right':[], 'ego_right':[], 'distance_to_hiker':[]}
    virtual_centers = {'hiker_left':0,'hiker_diagonal_left':0,'hiker_center':0,
                        'hiker_diagonal_right':0,'hiker_right':0,'altitude':0,
                        'ego_left':0, 'ego_diagonal_left':0, 'ego_center':0,
                        'ego_diagonal_right':0, 'ego_right':0, 'distance_to_hiker':0}
    for key in observation_dimensions:
        for obs in bin:
            observation_dimensions[key].append(access_by_key(key,obs)[1])

    for key in virtual_centers:
        virtual_centers[key] = sum(observation_dimensions[key]) / float(len(observation_dimensions[key]))

    return virtual_centers

def get_furthest_index(center,data,exclusion):

    center_chunk = []
    #build the center chunk
    for key,val in center.items():
        slv = [key,[key,val]]
        center_chunk.extend(slv)
        #quick fix to make it the same format as other chunks
        if key == 'ego_right':
            center_chunk.extend(['type',['type','nav']])
    furthest_chunk_index = -1
    distance = 0
    for i in range(len(data)):
        distance_from_center = euclidean_between_chunks(center_chunk,data[i])
        if distance_from_center > distance and i not in exclusion:
            furthest_chunk_index = i
            distance = distance_from_center

    # for chunk in data:
    #     distance_from_center = euclidean_between_chunks(center_chunk,chunk)
    #     if distance_from_center > distance:
    #         furthest_chunk = chunk
    return furthest_chunk_index



def convert_centroids_to_chunks(category,centroids,original_labels,kind='nav'):
    chunks = []
    action_values = {'drop': 0, 'left': 0, 'diagonal_left': 0,
                     'center': 0, 'diagonal_right': 0, 'right': 0,
                     'up': 0, 'down': 0, 'level': 0}
    actr_actions = ['_'.join(x) if len(x) > 1 else x[0] for x in action_to_category_map.values()]

    for centroid in centroids:
        chunk = []
        if include_fc:
            centroid, fc = list(centroid[:12].astype(float)), list(centroid[12:].astype(float))

        for slot,value in zip(original_labels,centroid):
            chunk.append(slot)
            chunk.append([slot,value])
        chunk.append('fc')
        chunk.append(['fc', fc])
        for key in actr_actions:
            chunk.append(key)
            chunk.append([key, int(key == category)])


        chunk.append('type')
        chunk.append(kind)
        # chunk.append(['type',kind])
        chunks.append(chunk)
    return chunks



def get_data_of_one_kind(data,keyword='nav'):
    kind = []
    for episode in data:
        for step in episode[keyword]:
            kind.append(copy.deepcopy(step))
    return kind

def average_euclidean(chunks):
    tot = 0
    for i in range(len(chunks)-1):
        tot += euclidean_between_chunks(chunks[i],chunks[i+1])
    return tot/len(chunks)/2.

def euclidean_between_chunks(chunk1,chunk2):
    distance = (chunk1[1][1]-chunk2[1][1])**2     \
                + (chunk1[3][1]-chunk2[3][1])**2  \
                + (chunk1[5][1]-chunk2[5][1])**2  \
                + (chunk1[7][1]-chunk2[7][1])**2  \
                + (chunk1[9][1]-chunk2[9][1])**2    \
                + (chunk1[11][1] - chunk2[11][1])**2    \
                + (chunk1[13][1] - chunk2[13][1])**2    \
                + (chunk1[15][1] - chunk2[15][1])**2    \
                + (chunk1[17][1] - chunk2[17][1])**2    \
                + (chunk1[19][1] - chunk2[19][1])**2    \
                + (chunk1[21][1] - chunk2[21][1])**2    \
                + (chunk1[25][1] - chunk2[25][1])**2
    return math.sqrt(distance)

def access_by_key(key, list):
    '''Assumes key,vallue pairs and returns the value'''

    if not key in list:
        raise KeyError(f'Key {key} not in list {list}.')

    return list[list.index(key)+1]

def remap( x, oMin, oMax, nMin, nMax ):
    #https://stackoverflow.com/questions/929103/convert-a-number-range-to-another-range-maintaining-ratio
    #range check
    #oMin = original Minimum, n = new
    if oMin == oMax:
        print("Warning: Zero input range")
        return None

    if nMin == nMax:
        print("Warning: Zero output range")
        return None

    #check reversed input range
    reverseInput = False
    oldMin = min( oMin, oMax )
    oldMax = max( oMin, oMax )
    if not oldMin == oMin:
        reverseInput = True

    #check reversed output range
    reverseOutput = False
    newMin = min( nMin, nMax )
    newMax = max( nMin, nMax )
    if not newMin == nMin :
        reverseOutput = True

    portion = (x-oldMin)*(newMax-newMin)/(oldMax-oldMin)
    if reverseInput:
        portion = (oldMax-x)*(newMax-newMin)/(oldMax-oldMin)

    result = portion + newMin
    if reverseOutput:
        result = newMax - portion

    return result

def get_chunks_with_action(chunks,action):
    '''Returns a list to subdivide the chunks'''
    #rList = []
    # for chunk in chunks:
    #     if access_by_key(action_key1,chunk)
    rList = [x for x in chunks if access_by_key(action,x)[1]]# and access_by_key(action_key2,x)[1]]
    return rList

def sort_chunks_by_action(chunksList,drop=False):
    # combos_to_actions = {('down', 'left'): 0, ('down', 'diagonal_left'): 1, ('down', 'center'): 2,
    #                      ('down', 'diagonal_right'): 3, ('down', 'right'): 4,
    #                      ('level', 'left'): 5, ('level', 'diagonal_left'): 6, ('level', 'center'): 7,
    #                      ('level', 'diagonal_right'): 8, ('level', 'right'): 9,
    #                      ('up', 'left'): 10, ('up', 'diagonal_left'): 11, ('up', 'center'): 12,
    #                      ('up', 'diagonal_right'): 13, ('up', 'right'): 14, ('drop'): 15}
    combos_to_actions = {}
    actr_actions = ['_'.join(x) if len(x) > 1 else x[0] for x in action_to_category_map.values()]
    for key,value in action_to_category_map.items():
        actr_action = ''
        if len(value) > 1:
            actr_action = '_'.join(value)
        else:
            actr_action = value[0]
        combos_to_actions[actr_action] = key

    if not drop:
        del combos_to_actions['drop']
    for key in combos_to_actions:
        combos_to_actions[key] = get_chunks_with_action(chunksList,key)
    return combos_to_actions

#bin_chunks_by_action(all_data)

#each of the entries in all data is an episode, comprised of nav and drop steps.
#dumped into two types of memories, nav and drop
#nav = []
#drop = []

#all_navs = get_data_of_one_kind(all_data,keyword='nav')
#all_drops = get_data_of_one_kind(all_data,keyword='drop')

#navs_memory = convert_data_to_chunks(all_navs)
#sort  by action
#navs_by_action = sort_chunks_by_action(all_navs)
#drops_by_action = sort_chunks_by_action(all_drops)


#all_data = bin_chunks_by_action(all_data)

#convert_data_to_ndarray(all_navs)
#memory = convert_data_to_chunks(all_data) #[[nav list,drop list]]

#nav_avg = average_euclidean(memory[0])
#tt = access_by_key('down',memory[0][0])

#down_left = get_chunks_with_action(memory[0], 'down', 'left')
memory = convert_data_to_chunks(all_data)#[[nav list,drop list]]
navs_by_action = sort_chunks_by_action(memory[0])
drops_by_action = sort_chunks_by_action(memory[1])

#once binned, the chunks should not contain their action data anymore


reduced_navs = {('down', 'left'): [], ('down', 'diagonal_left'): [], ('down', 'center'): [],
                         ('down', 'diagonal_right'): [], ('down', 'right'): [],
                         ('level', 'left'): [], ('level', 'diagonal_left'): [], ('level', 'center'): [],
                         ('level', 'diagonal_right'): [], ('level', 'right'): [],
                         ('up', 'left'): [], ('up', 'diagonal_left'): [], ('up', 'center'): [],
                         ('up', 'diagonal_right'): [], ('up', 'right'): [], ('drop'): []}
#action_chunks_by_key = {('down', 'left'):['down']}
# change = True
# while change:
#     one_pass,change = random_distal_chunks(navs_by_action)
#     if change:
#         for key in one_pass:
#             reduced_navs[key].extend(one_pass[key])

#beore I can do a distance measure, I have to transpose the distances to 0-1,
#otherwise, they dominate the distance measure
#same with altitudes
nav_distances = []
nav_altitudes = []
nav_egos = []
egoses = ['ego_left', 'ego_diagonal_left','ego_center','ego_diagonal_right','ego_right']
for key in navs_by_action:
    dists = [float(access_by_key('distance_to_hiker',x)[1]) for x in navs_by_action[key]]
    alts = [int(access_by_key('altitude',x)[1]) for x in navs_by_action[key]]
    for egotype in egoses:
        print('egotype:',egotype)
        nav_egos.extend([int(access_by_key(egotype,x)[1]) for x in navs_by_action[key]])
    nav_distances.extend(dists)
    nav_altitudes.extend(alts)
max_nav_distance = max(nav_distances)
min_nav_distances = min(nav_distances)
max_nav_alt = max(nav_altitudes)
min_nav_alt = min(nav_altitudes)
max_nav_ego = max(nav_egos)
min_nav_ego = min(nav_egos)

#modify the by_actions to the new transponsed values
for key in navs_by_action:
    for chunk in navs_by_action[key]:
        for i in range(len(chunk)):
            if chunk[i] == 'distance_to_hiker':
                chunk[i+1][1] = remap(chunk[i+1][1], min_nav_distances, max_nav_distance, 0, 1)
            if chunk[i] == 'altitude':
                chunk[i+1][1] = remap(chunk[i+1][1], min_nav_alt, max_nav_alt, 0, 1)
            if chunk[i] in egoses:
                chunk[i+1][1] = remap(chunk[i+1][1], min_nav_ego, max_nav_ego, 0, 1)



indexes = {}
index_list = []
#before gathering the most distant, empty the bins
garbage = []
for key in navs_by_action:
    if not navs_by_action[key]:
        garbage.append(key)
for key in garbage:

    del navs_by_action[key]
#Now we have no empty bins

#K-means
#Before k-means, we need vectors.
#make a new dictionary, same keys, where the values will be np arrays

#first, remove the actions from the bins
for bin in navs_by_action:
    for i in range(len(navs_by_action[bin])):
        clean_example = []
        for x in navs_by_action[bin][i]:
            if type(x) == list:
                if x[0] not in actions:
                    clean_example.append(x)
            else:
                if x not in actions:
                    clean_example.append(x)
        navs_by_action[bin][i] = clean_example


original_labels_nav = []
navs_by_action_array = {}
for key, val in navs_by_action.items():
    navs_by_action_array[key],original_labels_nav = convert_data_to_ndarray(val)

#don't cluster, just pick randomly 100 examples from each bin.
#if it doesn't have 100, it doesn't continue
#put those examples as chunks
for key in navs_by_action:
    random.shuffle(navs_by_action[key])
    navs_by_action[key] = navs_by_action[key][:100]

print('stop')



# #nav by action clusters will hold the center cluster for each action type
# #it can then be used to make a single centroid chunk - just to try
# ######MEAN SHIFT VERSION
# nav_by_action_clusters = {}
# keys_to_delete = []
# for key in navs_by_action_array:
#     X = navs_by_action_array[key]
#     if X.shape[0] <= 3:
#         keys_to_delete.append(key)
#         continue
#     ms = MeanShift()
#     ms.fit(X)
#     nav_by_action_clusters[key] = ms.cluster_centers_
# #######K-MEANS VERSION
# # nav_by_action_k_clusters = {}
# # keys_to_delete = []
# # for key in navs_by_action_array:
#
#
# #don't delete them, use them as centers - they are so rare
# #keep commented out below to keep them
# # for key in keys_to_delete:
# #     del nav_by_action_clusters[key]
#
# #convert whatever centroids there are into chunks again
# nav_by_action_clusters_chunks = {}
# for key in nav_by_action_clusters:
#     nav_by_action_clusters_chunks[key] = convert_centroids_to_chunks(key, nav_by_action_clusters[key],original_labels_nav,kind='nav')

#convert the dictionary back into a list of all chunks
nav_complete_list = []
for key,value in navs_by_action.items():#nav_by_action_clusters_chunks.items()
    count = 0
    for chunk in value:
        count += 1
        # if count >= 7:
        #     continue#break
        nav_complete_list.append(chunk)


#X,original_labels = navs_by_action_array[('up','left')]
#ms = MeanShift()
#ms.fit(X)
#labels = ms.labels_
#cluster_centers = ms.cluster_centers_
with open('chunks_cluster_centers_15actions_2000_fc_100randommax.pkl','wb') as handle:
    pickle.dump(nav_complete_list,handle)

print('stop')
##commented out below to try k-means
# #find the virtual center of each bin, then find the farthest n points
# ran_out = False
# #first convert the data structure so it's a dictionary that stores the center
# #so it doesn't have to be calculated multiple times
# # new_navs_by_action = {}
# # for key in navs_by_action:
# #     new_navs_by_action[key] = {'vals':navs_by_action[key],'center':{},'ordered_indicies':[]}
# #
# # for key in new_navs_by_action:
# #     new_navs_by_action[key]['center'] = virtual_center(new_navs_by_action[key]['vals'])
# #
# # furthest_nav_indexs_ordered = {'hiker_left':[],'hiker_diagonal_left':[],'hiker_center':[],
# #                         'hiker_diagonal_right':[],'hiker_right':[],'altitude':[],
# #                         'ego_left':[], 'ego_diagonal_left':[], 'ego_center':[],
# #                         'ego_diagonal_right':[], 'ego_right':[], 'distance_to_hiker':[]}
# #
# # for key in new_navs_by_action:
# #     for i in range(len(new_navs_by_action[key]['vals'])):
# #         new_navs_by_action[key]['ordered_indicies'].append(get_furthest_index(new_navs_by_action[key]['center'],new_navs_by_action[key]['vals'],new_navs_by_action[key]['ordered_indicies']))
# #
# #
# # #find smallest 'vals' size, in order to reduce size so samples are even
# # vals_len = -1
# # smallest_list = 1000000000000
# # for key in new_navs_by_action:
# #     if len(new_navs_by_action[key]['vals']) < smallest_list:
# #         smallest_list = len(new_navs_by_action[key]['vals'])
# # #create the final set
# # nav_complete_list = []
# # for key in new_navs_by_action:
# #     for i in range(smallest_list):
# #         index_of_chunk = new_navs_by_action[key]['ordered_indicies'][i]
# #         nav_complete_list.append(new_navs_by_action[key]['vals'][index_of_chunk])
# #         #nav_complete_list.append(new_navs_by_action[key]['vals'][new_navs_by_action[key]['ordered_indicies'][i]])
# #
# #
# #
# #
# # print("ok")
# #
# #
# # max_mins = {'distance':[max_nav_distance,min_nav_distances],
# #             'ego':[max_nav_ego,min_nav_ego],
# #             'altitude':[max_nav_alt,min_nav_alt]}
# #
# #
# # # find_fail = False
# # # new_navs_by_action = copy.deepcopy(navs_by_action)
# # # while True:
# # #     for key in new_navs_by_action:
# # #         new_indexes = []
# # #         chunks = new_navs_by_action[key]
# # #         new_indexes,dist = index_of_most_distal_chunks(chunks)
# # #         if not new_indexes:
# # #             find_fail = True
# # #             continue
# # #         if not key in indexes:
# # #             indexes[key] = new_indexes
# # #         else:
# # #             indexes[key].extend(new_indexes)
# # #         for ind in new_indexes:
# # #             new_navs_by_action[key] = [i for j, i in enumerate(new_navs_by_action[key]) if j not in new_indexes]
# # #     if find_fail:
# # #         break
# # #
# # #
# # #
# # #
# # #
# # #     print("ok")
# # # print("done")
# # #HERE - now that I have the index - add it to a new list, and clear out the old, n times untiles the first combination (up left) gives up
# #
# #
# #
# # #average_distance_navs = average_euclidean(memory[0])
# # #average_distance_drops = average_euclidean(memory[0])
# #
# # #the aim here will be to pick two items from each action category
# # #first pick a random one, then find one greater than average distance away
# #
# #
# #
# # #file_path = os.path.join(data_path,filename)
# # with open('chunks_maxdistance.pkl','wb') as handle:
# #     pickle.dump(nav_complete_list,handle)
# #
# # with open('max_mins_from_data.pkl','wb') as handle2:
# #     pickle.dump(max_mins,handle2)
# #
# # print("done.")