import os
import glob
from pyactup import *
import pickle
import random
import math

from pathlib import Path

import time

from multiprocessing import Pool
from functools import partial

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from scipy.stats import entropy
from scipy.spatial import distance

import itertools

#set a random seed for pseudo-random
random.seed(42)

###some global variables that should be useful in multiple spots
action_slots = ['left_down','diagonal_left_down','center_down','diagonal_right_down','right_down',
                   'left_level','diagonal_left_level','center_level','diagonal_right_level','right_level',
                   'left_up','diagonal_left_up','center_up','diagonal_right_up','right_up', 'drop']
#excluds entropy and FC
observation_slots = ['hiker_left', 'hiker_diagonal_left', 'hiker_center', 'hiker_diagonal_right', 'hiker_right',
              'ego_left', 'ego_diagonal_left', 'ego_center', 'ego_diagonal_right', 'ego_right',
              'altitude', 'distance_to_hiker']

combos_to_actions = {'left_down':0,'diagonal_left_down':1,'center_down':2,
                     'diagonal_right_down':3,'right_down':4,
                     'left_level':5,'diagonal_left_level':6,'center_level':7,
                     'diagonal_right_level':8,'right_level':9,
                     'left_up':10,'diagonal_left_up':11,'center_up':12,
                     'diagonal_right_up':13,'right_up':14,'drop':15}

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

FC_distances = []



def convert_dict_chunk_to_vector(chunk,keys):
    '''cannot just take the values because it does not preserve order. this function makes a list of values, all in the same order'''
    values = [chunk[slot] for slot in keys]
    return values

def convert_vector_to_dict_chunk(vector,keys):
    '''len of vector must equal len of keys'''
    return {slot:value for slot,value in zip(keys,vector) if not slot == 'action_probs'}

def strip_slots(chunk, slots):
    '''strips actions away, leaving only observations'''
    for action in slots:
        del chunk[action]
    return chunk

def unpack_chunks(chunks):
    temp_chunks = []
    all_chunks = []
    for action in chunks:
        for chunk in chunks[action]:
            chunk = chunk[:-4]
            achunk = {}
            for i in range(1,len(chunk),2):
                slot,value = chunk[i][0], chunk[i][1]
                achunk[slot] = value
            for possible_action in action_slots:
                achunk[possible_action] = int(possible_action == action)

            # print('test')
            all_chunks.append(achunk)
    return all_chunks

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
    if drone == hiker:
        return 500
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

    if angle >= 179.9:
        returndict['hiker_right'] = 1
        returndict['hiker_left'] = 1
    if angle <= -179.9:
        returndict['hiker_right'] = 1
        returndict['hiker_left'] = 1

    if angle == 500:
        returndict['hiker_right'] = 0
        returndict['hiker_left'] = 0

    return returndict

def convert_data_to_chunks(all_data,include_fc=False):
    nav = []
    drop = []
    for episode in all_data:
        for step in episode['nav']:
            # action_values = {'drop': 0, 'left': 0, 'diagonal_left': 0,
            #                  'center': 0, 'diagonal_right': 0, 'right': 0,
            #                  'up': 0, 'down': 0, 'level': 0}
            # angle to hiker: negative = left, positive right
            egocentric_angle_to_hiker = heading_to_hiker(step['heading'], step['drone'], step['hiker'])
            angle_categories_to_hiker = angle_categories(egocentric_angle_to_hiker)
            egocentric_slice = egocentric_representation(step['drone'], step['heading'], step['volume'])
            # compile all that into chunks [slot, value, slot, value]
            chunk = {}
            for key, value in angle_categories_to_hiker.items():
                chunk[key] = value#.extend([key, [key, value]])
            # need the altitudes from the slice
            altitudes = altitudes_from_egocentric_slice(egocentric_slice)
            altitudes = [x - 1 for x in altitudes]
            alt = step['altitude']
            #chunk.extend(['altitude', ['altitude', int(alt)]])
            chunk['altitude'] = int(alt)
            chunk['ego_left'] = altitudes[0]
            chunk['ego_diagonal_left'] = altitudes[1]
            chunk['ego_center'] = altitudes[2]
            chunk['ego_diagonal_right'] = altitudes[3]
            chunk['ego_right'] = altitudes[4]
            # chunk.extend(['ego_left', ['ego_left', altitudes[0]],
            #               'ego_diagonal_left', ['ego_diagonal_left', altitudes[1]],
            #               'ego_center', ['ego_center', altitudes[2]],
            #               'ego_diagonal_right', ['ego_diagonal_right', altitudes[3]],
            #               'ego_right', ['ego_right', altitudes[4]]])


            chunk['distance_to_hiker'] = distance_to_hiker(np.array(step['drone']), np.array(step['hiker']))
            # also want distance  to hiker
            # chunk.extend(['distance_to_hiker',
            #               ['distance_to_hiker', distance_to_hiker(np.array(step['drone']), np.array(step['hiker']))]])
            # split action into components [up, level, down, left, right, etc]
            # components = action_to_category_map[step['action']]
            # for component in components:
            #     action_values[component] = 1
            # for key, value in action_values.items():
            #     chunk.extend([key, [key, value]])

            #last part of the observation side will be the vector
            if include_fc:
                fc_list = tuple(step['fc'].tolist()[0])
                chunk['fc'] = fc_list#.extend(['fc', ['fc', fc_list]])

            # if step['action'] == 15:
            #     print('15')

            #add the action probabilities, to keep track of
            chunk['action_probs'] = step['action_probs']


            #no longer splitting the actions. Use all 15
            #actr_actions = ['_'.join(x) if len(x) > 1 else x for x in action_to_category_map.values()]
            for key,value in action_to_category_map.items():
                if len(value) > 1:
                    actr_action = '_'.join(value)
                else:
                    actr_action = value[0]
                if key == step['action']:
                    chunk[actr_action] = 1#.extend([actr_action,[actr_action,1]])
                else:
                    chunk[actr_action] = 0#.extend([actr_action,[actr_action,0]])

            # chunk.extend(['drop',['drop',drop_val]])
            # chunk.extend(['type', 'nav'])




            nav.append(chunk)


    return nav

def multi_blends(chunk, memory, slots ):
    return [memory.blend(slot,**chunk) for slot in slots]

def vector_similarity(x,y):
    return distance.euclidean(x,y) / 4.5#max(FC_distances)
    # return distance.cosine(x,y)


def custom_similarity(x,y):
    return 0
    return abs(x - y)

#set the similarity function
set_similarity_function(custom_similarity, *observation_slots)
set_similarity_function(vector_similarity, 'fc')

if __name__ == "__main__":
    results = []
    all_chunks = []
    FC_distances = []


    normalized_data_file = 'normalized_all_chunks_fc_MOD1.lst'
    chunks_path = Path("/Users/paulsomers/COGLE/gym-gridworld/")

    if not os.path.isfile(os.path.join(chunks_path,normalized_data_file)):

        ###load the data
        ###Need the raw data, then convert it into chunks because you need the intial distributions for the actions
        data_file_name = 'all_data308-110_9-7_4-1_9-115000.lst'
        data_path = Path('/Users/paulsomers/COGLE/gym-gridworld/data_tools/')
        all_data = pickle.load(open(os.path.join(data_path, data_file_name), 'rb'))

        ###convert the data to chunks
        all_chunks = convert_data_to_chunks(all_data,include_fc=True)

        #find all the euclidean distances between all fc to normalize the euclidean
        FCs = []

        for chunk in all_chunks:
            FCs.append(chunk['fc'])

        #normalize the fc
        normalizer = preprocessing.Normalizer(norm='l2').fit(FCs)
        FCs = normalizer.transform(FCs)#[normalizer.transform(x) for x in FCs]
        for chunk,fc in zip(all_chunks,FCs):
            chunk['fc'] = tuple(fc.tolist())

        split_FCs = [FCs[i::100] for i in range(100)]  #[test_chunks[i::10] for i in range(10)]
        for FC_list in split_FCs:
            FC_combination = list(itertools.combinations(FC_list,2))
            sub_distances = [distance.euclidean(x[0],x[1]) for x in FC_combination]
            FC_distances.append(max(sub_distances))



        with open(os.path.join(chunks_path, 'FC_distances.pkl'), 'wb') as handle:
            pickle.dump(FC_distances, handle)

        #the chunks values should be normalized
        #make a dictionary of all values, use max(values) to normalize
        data_by_slot = {slot:[] for slot in observation_slots}
        for chunk in all_chunks:
            for slot in observation_slots:
                data_by_slot[slot].append(chunk[slot])
        acount = 0
        for chunk in all_chunks:
            print(acount)
            for slot in observation_slots:#['altitude', 'distance_to_hiker', 'ego_right','ego_diagonal_right']:#
                chunk[slot] = chunk[slot] / max(data_by_slot[slot])
            acount += 1

        with open(os.path.join(chunks_path, normalized_data_file), 'wb') as handle:
            pickle.dump(all_chunks, handle)

    else:
        # all_chunks = pickle.load(open(chunks_path + normalized_data_file, 'rb'))
        FC_distances = pickle.load(open(os.path.join(chunks_path, 'FC_distances.pkl'), 'rb'))
        all_chunks = pickle.load(open(os.path.join(chunks_path, normalized_data_file), 'rb'))




    #vectorize the data, ensuring the order
    #include the action probs
    vectorized_data = [convert_dict_chunk_to_vector(chunk,observation_slots+['fc']) for chunk in all_chunks]
    vectorized_targets = [convert_dict_chunk_to_vector(chunk,action_slots + ['action_probs']) for chunk in all_chunks]

    X_train, X_test, Y_train, Y_test = train_test_split(vectorized_data,vectorized_targets, test_size=0.20, random_state=42)

    #now there is a vectorized training set and test set - turn them back to dictionaries (chunks that can be read by pyactup)
    #cannot store the numpy array (action_probs) so, perserve the order from this point on - you can go back to X_train, Y_train to match the action_probs
    training_chunks = []
    for obs,act in zip(X_train,Y_train):
        chunk = convert_vector_to_dict_chunk(obs+act, observation_slots+['fc']+action_slots)
        training_chunks.append(chunk)

    test_chunks = [convert_vector_to_dict_chunk(test,observation_slots+['fc']) for test in X_test]


    ###now run the model
    temperatures = [0.25,0.65,1.0]#.3,0.4,0.5]#.6,0.7,0.8,0.9,1.0]
    mismatches = [8.0,10.0,12.0,14.0,16.0]
    parameters = [(x,y) for x in temperatures for y in mismatches]
    used_parameters = {'temperature': [], 'mismatches': []}
    os.chdir(chunks_path)
    datafiles = glob.glob("20190912*")
    for file in datafiles:
        dats = pickle.load(open(file, 'rb'))
        used_parameters['temperature'].append(dats['temperature'])
        used_parameters['mismatches'].append(dats['mismatch'])

    for parameter in parameters:
        #because of issues with crashing - find all parameters that have been done, and skip them
        if parameter in zip(used_parameters['temperature'],used_parameters['mismatches']):
            continue


        m = Memory(noise=0.0, decay=0.0, temperature=parameter[0], threshold=-100.0, mismatch=parameter[1], optimized_learning=False)
        m._maximum_similarity = 5
        #put the training chunks in memory
        for chunk in training_chunks:
            m.learn(**chunk)

        m.advance()
        p = Pool(processes=20)
        # split_chunks = [test_chunks[i::10] for i in range(10)]
        multi_p = partial(multi_blends, memory=m,slots=action_slots)
        data = p.map(multi_p, test_chunks[0:100])
        # for test_chunk in test_chunks:
            # results.append([m.blend(x,**test_chunk) for x in action_slots])

        JS = []
        matches = []
        for result,datum in zip(data,Y_test):
            JS.append(distance.jensenshannon(result,datum[-1][0]))
            matches.append(int(np.argmax(result)==np.argmax(datum[:-1])))
        avgJS = sum(JS)/len(JS)
        avgMatch = sum(matches)/len(matches)

        save_dict = {'data':data,'Y_test':Y_test,'JS':JS,'matches':matches,'AVGJS':avgJS, 'avgMatch':avgMatch, 'temperature':parameter[0],'mismatch':parameter[1]}

        timestr = time.strftime("%Y%m%d-%H%M%S")
        with open(os.path.join(chunks_path, timestr), 'wb') as handle:
            pickle.dump(save_dict, handle)
        p.close()
       #del m
       #del data
       #del p
       #del multi_p
       #del JS
       #del matches


    print('stop')