from collections import namedtuple
# from pysc2.lib import actions

# import pygame
import numpy as np
import sys
from actorcritic.agent import ActorCriticAgent, ACMode
from common.preprocess import ObsProcesser, ActionProcesser, FEATURE_KEYS
from common.util import calculate_n_step_reward, general_n_step_advantage, combine_first_dimensions, \
    dict_of_lists_to_list_of_dicst
import tensorflow as tf
from absl import flags
import random
# from time import sleep
from actorcritic.policy import FullyConvPolicy, MetaPolicy, RelationalPolicy


import copy
import math
import itertools
import os
import json
import operator
import pickle
import actr #Version 7.11.1 tested (may work on others)
import threading
import time
from scipy import spatial
from scipy.interpolate import interp1d
from scipy.stats import entropy

from sklearn import preprocessing

PPORunParams = namedtuple("PPORunParams", ["lambda_par", "batch_size", "n_epochs"])
#ADD some global stuff for ACT-R

#stats
stats = {'crashes':0,'successes':0}

actr_time = 0

actr_initialized = False
min_max = {'ego_left':[],
           'ego_diagonal_left':[],
           'ego_diagonal_right':[],
           'ego_center':[],
           'ego_right':[],
           'distance_to_hiker':[],
           'hiker_left':[],
           'hiker_diagonal_left':[],
           'hiker_center':[],
           'hiker_diagonal_right':[],
           'hiker_right':[],
           'distance_to_hiker':[],
           'altitude':[],
           'fc':[]}


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
action_list = ['left_down','diagonal_left_down','center_down','diagonal_right_down','right_down',
                   'left_level','diagonal_left_level','center_level','diagonal_right_level','right_level',
                   'left_up','diagonal_left_up','center_up','diagonal_right_up','right_up']

# combos_to_actions = {('down','left'):0,('down','diagonal_left'):1,('down','center'):2,
#                      ('down','diagonal_right'):3,('down','right'):4,
#                      ('level','left'):5,('level','diagonal_left'):6,('level','center'):7,
#                      ('level','diagonal_right'):8,('level','right'):9,
#                      ('up','left'):10,('up','diagonal_left'):11,('up','center'):12,
#                      ('up','diagonal_right'):13,('up','right'):14,('drop'):15}

combos_to_actions = {'left_down':0,'diagonal_left_down':1,'center_down':2,
                     'diagonal_right_down':3,'right_down':4,
                     'left_level':5,'diagonal_left_level':6,'center_level':7,
                     'diagonal_right_level':8,'right_level':9,
                     'left_up':10,'diagonal_left_up':11,'center_up':12,
                     'diagonal_right_up':13,'right_up':14,'drop':15}

allchunks = []

fc_distances = []

interp_dict = {}


def distance_to_hiker(drone_position,hiker_position):
    distance = np.linalg.norm(drone_position-hiker_position)
    return distance


def altitudes_from_egocentric_slice(ego_slice):
    alts = np.count_nonzero(ego_slice, axis=0)
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


def remap( x, oMin, oMax, nMin, nMax ):
    #https://stackoverflow.com/questions/929103/convert-a-number-range-to-another-range-maintaining-ratio
    #range check
    oSpan = oMax - oMin
    nSpan = nMax - nMin

    valueScaled = float(x - oMin) / float(oSpan)

    return nMin + (valueScaled * nSpan)


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


def access_by_key(key, list):
    '''Assumes key,vallue pairs and returns the value'''
    if not key in list:
        raise KeyError("Key not in list")

    return list[list.index(key)+1]


def similarity(val1, val2):
    '''Linear tranformation, abslute difference'''
    #val2 is recalled, val1 is the observation
    # import pdb; pdb.set_trace()
    if val1 == val2:
        return 0
    # import pdb; pdb.set_trace()
    # if val1 == 'NAV' and val2 == 'DROP':
    #     return -5

    if not type(val1) == list:
        return 0

    # if val1[0] == 'ALTITUDE':# or val1[0] == 'DISTANCE_TO_HIKER':
    #     return 0

    if val1[0] == 'FC':
        # return 0
        # r1 = spatial.distance.minkowski(val1[1], val2[1],1) * - 1
        r2 = spatial.distance.euclidean(val1[1], val2[1]) * - 1
        # r3 = spatial.distance.cosine(val1[1], val2[1]) * -1
        return r2

    # return 0
    return_value = abs(val1[1] - val2[1]) * -1
    return return_value






def compute_S(blend_trace, keys_list):
    '''For blend_trace @ time'''
    #probablities
    probs = [x[3] for x in access_by_key('MAGNITUDES',access_by_key('SLOT-DETAILS',blend_trace[0][1])[0][1])]
    #feature values in probe
    FKs = [access_by_key(key.upper(),access_by_key('RESULT-CHUNK',blend_trace[0][1])) for key in keys_list]
    chunk_names = [x[0] for x in access_by_key('CHUNKS', blend_trace[0][1])]

    #Fs is all the F values (may or may not be needed for tss)
    #They are organized by chunk, same order as probs
    vjks = []
    for name in chunk_names:
        chunk_fs = []
        for key in keys_list:
            chunk_fs.append(actr.chunk_slot_value(name,key))
        vjks.append(chunk_fs)

    #tss is a list of all the to_sum
    #each to_sum is Pj x dSim(Fs,vjk)/dFk
    #therefore, will depend on your similarity equation
    #in this case, we need max/min of the features because we use them to normalize
    max_val = 4#max(map(max, zip(*feature_sets)))
    min_val = 1#min(map(min, zip(*feature_sets)))
    n = max_val - min_val
    n = max_val
    #n = 1
    #this case the derivative is:
    #           Fk > vjk -> -1/n
    #           Fk = vjk -> 0
    #           Fk < vjk -> 1/n
    #compute Tss
    #there should be one for each feature
    #you subtract the sum of each according to (7)
    tss = {}
    ts2 = []
    for i in range(len(FKs)):
        if not i in tss:
            tss[i] = []
        for j in range(len(probs)):
            if FKs[i][1] > vjks[j][i][1]:
                dSim = -1/max(min_max[vjks[j][i][0].lower()])
            elif FKs[i][1] == vjks[j][i][1]:
                dSim = 0
            else:
                dSim = 1/max(min_max[vjks[j][i][0].lower()])
            tss[i].append(probs[j] * dSim)
        ts2.append(sum(tss[i]))

    #vios
    viosList = []
    viosList.append([actr.chunk_slot_value(x,'action') for x in chunk_names])
    #viosList.append([actr.chunk_slot_value(x,'altitude_change') for x in chunk_names])
    #viosList.append([actr.chunk_slot_value(x, 'diagonal_right_turn') for x in chunk_names])
    #viosList.append([actr.chunk_slot_value(x, 'right_turn') for x in chunk_names])
    #viosList.append([actr.chunk_slot_value(x, 'ascending') for x in chunk_names])
    #viosList.append([actr.chunk_slot_value(x, 'drop_action') for x in chunk_names])
    #compute (7)
    rturn = []
    for vios in viosList:
        results = []
        for i in range(len(FKs)):
            tmp = 0
            sub = []
            for j in range(len(probs)):
                if FKs[i][1] > vjks[j][i][1]:
                    dSim = -1/max(min_max[vjks[j][i][0].lower()])
                elif FKs[i] == vjks[j][i]:
                    dSim = 0
                else:
                    dSim = 1/max(min_max[vjks[j][i][0].lower()])
                tmp = probs[j] * (dSim - ts2[i]) * vios[j][1]#sum(tss[i])) * vios[j]
                sub.append(tmp)
            results.append(sub)

        #print("compute S complete")
        rturn.append(results)
    return rturn

def reset_actr():
    global actr_initialized
    global allchunks
    global fc_distances
    global min_max
    global interp_dict


    if not actr_initialized or actr_initialized:
        model_name = 'egocentric_allocentric_salience_entropy_pitch_yaw.lisp'
        model_path = '/Users/paulsomers/COGLE/gym-gridworld/'

        chunk_file_name = 'chunk_dict_ego_entropy.pkl'
        #chunk_path = os.path.join(model_path,'data')
        chunk_path = '/Users/paulsomers/COGLE/gym-gridworld/'
        actr.add_command('similarity_function',similarity)
        actr.add_command('ticker', actr_time)

        # actr_thread = threading.Thread(target=actr.load_act_r_model, args=[os.path.join(model_path,model_name)])
        actr.load_act_r_model(os.path.join(model_path,model_name))
        # actr_thread = threading.Thread(target=actr.run_full_time, args=[3600, True])
        # actr_thread.daemon = True
        # actr_thread.start()
        # # actr.load_act_r_model(os.path.join(model_path,model_name))
        # actr.record_history("blending-trace")

        # max_mins_name = 'max_mins_from_data.pkl'
        # max_mins = pickle.load(open(os.path.join(chunk_path,max_mins_name),'rb'))
        min_max_name = 'min_max_dict_ego.pkl'
        min_max = pickle.load(open(os.path.join(chunk_path,min_max_name),'rb'))

        # Need to normalize the vectors
        print('Creating normalization')
        fc_array = np.array([np.array(x) for x in min_max['fc']])
        # scalar = preprocessing.StandardScaler(with_std=True).fit(fc_array)
        # transform_fc = scalar.transform(fc_array)
        normalizer = preprocessing.Normalizer(norm='max').fit(fc_array)

        interp_dict['fc'] = normalizer
        print('Normalization transform created')

        # the interpt dict will be pre-loaded to transform values to 0-1, based on their key
        to_transform = ['ego_left', 'ego_diagonal_left', 'ego_center', 'ego_diagonal_right', 'ego_right',
                        'distance_to_hiker', 'altitude']
        for trans in to_transform:
            if min_max[trans]:
                interp_dict[trans] = None
                min_val = min(min_max)
                max_val = max(min_max)
                func = interp1d([min_val, max_val], [0, 1])
                interp_dict[trans] = func
        # ignore_list = ['left', 'diagonal_left', 'center', 'diagonal_right', 'right', 'type', 'drop', 'up', 'down', 'level']
        ignore_list = ['left_down','diagonal_left_down','center_down','diagonal_right_down','right_down',
                       'left_level','diagonal_left_level','center_level','diagonal_right_level','right_level',
                       'left_up','diagonal_left_up','center_up','diagonal_right_up','right_up', 'drop', 'type', 'fc']

        action_category_values = {'yaw':{'diagonal_left_down':-0.5,'diagonal_left_level':-0.5,
                                         'diagonal_left_up':-0.5,'diagonal_right_up':0.5,
                                         'diagonal_right_down':0.5,'diagonal_right_level':0.5,
                                         'left_down':-1.0,'left_up':-1.0,
                                         'left_level':-1.0,'right_up':1.0,
                                         'right_level':1.0, 'right_down':1.0,
                                         'center_level':0.0,'center_down':0.0,'center_up':0.0},
                                  'pitch':{'diagonal_left_down':-1.0,'diagonal_left_level':0.0,
                                         'diagonal_left_up':1.0,'diagonal_right_up':1.0,
                                         'diagonal_right_down':-1.0,'diagonal_right_level':0.0,
                                         'left_down':-1.0,'left_up':1.0,
                                         'left_level':0.0,'right_up':1.0,
                                         'right_level':0.0, 'right_down':-1.0,
                                         'center_level':0.0,'center_down':-1.0,'center_up':1.0}}

        #load all the chunks
        allchunks = pickle.load(open(os.path.join(chunk_path,chunk_file_name),'rb'))

        for action_category in allchunks:
            action_chunks = allchunks[action_category]
            random.shuffle(action_chunks)
            action_chunks = action_chunks[:200] #select the first 100 (after randomized)

            #before addding, fc needs to be transformed, and floats have to be fixed to be json-able
            for chunk in action_chunks:

                #fc needs to be transformed
                fc_index = chunk.index('fc') + 1
                fc = [chunk[fc_index][1]]
                # fc = np.array(chunk[fc_index][1])
                # fc.reshape(1,-1)
                fc_transform = normalizer.transform(fc)
                chunk[fc_index] = ['fc',fc_transform.astype(float).tolist()[0]]

                chunk = [float(x) if type(x) == np.float64 else x for x in chunk]
                chunk = [int(x) if type(x) == np.int64 else x for x in chunk]

                #add the actions to the chunk
                # for action in ignore_list[:-1]:
                #     chunk.extend([action,[action,int(action == action_category)]])
                #add the actions in the new scheme
                for action in action_category_values:
                    if action_category == 'drop':
                        continue
                    chunk.extend([action,[action,action_category_values[action][action_category]]])
                    chunk.extend(['drop', ['drop', 0.0]])

                if action_category == 'drop':
                    chunk.extend(['drop', ['drop',1.0]])
                    chunk.extend(['yaw', ['yaw', 0.0]])
                    chunk.extend(['pitch', ['pitch',0.0]])


                actr.add_dm(chunk)

        # ignore_list = ['left', 'diagonal_left', 'center', 'diagonal_right', 'right', 'type', 'drop', 'up', 'down', 'level']
        ignore_list = ['left_down','diagonal_left_down','center_down','diagonal_right_down','right_down',
                       'left_level','diagonal_left_level','center_level','diagonal_right_level','right_level',
                       'left_up','diagonal_left_up','center_up','diagonal_right_up','right_up', 'drop', 'type', 'fc']








        #modifying the max_mins to include things from data collection, inorder to transponse
        # for key in max_mins:
        #     if key == 'ego':
        #         for aMinMax in min_max:
        #             if 'ego' in aMinMax:
        #                 min_max[aMinMax] = max_mins[key]
        #     elif key == 'distance':
        #         min_max['distance_to_hiker'] = max_mins[key]
        #     else:
        #         min_max[key] = max_mins[key]
    #print('asf')

    #distance of all FC, in order to scale the euclidean distance


    # fcs = []
    # if not actr_initialized:
    #
    #     for chunk in allchunks:
    #         fc_pair = access_by_key('fc', chunk)
    #         fcs.append(fc_pair[1])
    #
    #     for pair in itertools.combinations(fcs,2):
    #         fc_distances.append(float(spatial.distance.minkowski(pair[0], pair[1],2)))
    #         # fc_distances.append(float(np.linalg.norm(np.array(pair[0]) - np.array(pair[1]))))
    #
    #     with open('fc.pkl','wb') as handle:
    #         pickle.dump(fc_distances,handle)
    #     actr_initialized = True
    # else:
    #     fc_distances = pickle.load(open('fc.pkl','rb'))
    #
    # fc_distances = [min(fc_distances),max(fc_distances)]




    print("reset done.")
    actr_initialized = True
    # actr_thread = threading.Thread(target=actr.run_full_time, args=[3600,True])
    # actr_thread.daemon = True
    # actr_thread.start()


def create_actr_observation(step):
    transposes = ['ego_left', 'ego_diagonal_left', 'ego_center', 'ego_diagonal_right', 'ego_right',
                  'distance_to_hiker']

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
    alt = step['altitude']  #to be consistant with numpy
    chunk.extend(['altitude', ['altitude',int(alt)]])
    chunk.extend(['ego_left', ['ego_left',altitudes[0]],
                  'ego_diagonal_left', ['ego_diagonal_left',altitudes[1]],
                  'ego_center',  ['ego_center',altitudes[2]],
                  'ego_diagonal_right', ['ego_diagonal_right', altitudes[3]],
                  'ego_right', ['ego_right',altitudes[4]]])
    #include the entropy
    chunk.extend(['entropy',['entropy',float(entropy(step['action_probs'][0]))]])

    chunk.append('fc')

    step['fc'] = step['fc'].astype(float).tolist()[0]
    step['fc'] = interp_dict['fc'].transform([step['fc']]).astype(float).tolist()[0]
    chunk.append(['fc',step['fc']])
    chunk.extend(['type', 'nav'])
    # also want distance  to hiker
    chunk.extend(['distance_to_hiker', ['distance_to_hiker',distance_to_hiker(np.array(step['drone']), np.array(step['hiker']))]])
    # split action into components [up, level, down, left, right, etc]
    # components = action_to_category_map[step['action']]
    # for component in components:
    #     action_values[component] = 1
    # for key, value in action_values.items():
    #     chunk.extend([key, value])
    #json cannot serialize int64

    for trans in transposes:
        index_of_value = chunk.index(trans) + 1
        m = interp_dict[trans]
        val = chunk[index_of_value][1]
        chunk[index_of_value][1] = np.interp(val,[min(min_max[trans]),max(min_max[trans])],[0,1])

        # transformed_value = m(chunk[index_of_value][1])
        # chunk[index_of_value][1] = m(val)
        # print('normalizing')
    #just to make sure the values are compliant with json and actr
    chunk = [float(x) if type(x) == np.float64 else x for x in chunk]
    chunk = [int(x) if type(x) == np.int64 else x for x in chunk]

        # chunk[index_of_value][1] = remap(chunk[index_of_value][1], min(min_max[trans]),max(min_max[trans]),0,1)
    #transponse the transposes values to zero to 1 range

    return chunk

def actr_tick():
    global actr_time

    actr_time += 100
    print("TICK", actr_time)
    return actr_time

def handle_observation(observation):
    '''observation should have chunk format'''

    chunk = actr.define_chunks(observation)

    print("converting")
    # actr.schedule_simple_event_now("set-buffer-chunk",
    #                                ['imaginal', chunk[0]])
    # actr.mp_time()
    actr.schedule_set_buffer_chunk('imaginal',chunk[0],0)
    # actr.set_buffer_chunk('imaginal', chunk[0],False)
    # actr.run(100)



    # d = actr.get_history_data("blending-trace")
    # while d == None:
    #     time.sleep(3)
    #     d = actr.get_history_data("blending-trace")

    # actr.overwrite_buffer_chunk

    actr.run_until_action("call-blending-result-hooks")
    # d = json.loads(d)
    # actr.stop_recording_history("blending-trace")
    # actr.record_history("blending-trace")
    b = actr.buffer_read('blending')
    # while b == None:
    #     time.sleep(1)
    #     b = actr.buffer_read('blending')





    # b_content = actr.chunk_slot_value(b,'result-chunk')
    # first add the blend to the results dictionary
    # blend_return = access_by_key('RESULT-CHUNK', d[0][1])
    #I can do an overwrite and get rid of p2/p3

    #HACK - carry out the action here.
    # action_choice = {'left_down':0,'diagonal_left_down':0,'center_down':0,'diagonal_right_down':0,'right_down':0,
    #                  'left_level':0,'diagonal_left_level':0,'center_level':0,'diagonal_right_level':0,'right_level':0,
    #                  'left_up':0,'diagonal_left_up':0,'center_up':0,'diagonal_right_up':0,'right_up':0,
    #                  'drop':0}
    # for key in action_choice:
    #     action_choice[key] = actr.chunk_slot_value(b, key)
    # action = max(action_choice.items(), key=operator.itemgetter(1))[0]

    #new way to pick the action
    yaw_pitch_to_category = {'yaw': {0:['center_level','center_down','center_up'],
                                     0.5:['diagonal_right_down','diagonal_right_level','diagonal_right_up'],
                                     1.0:['right_up','right_level','right_down'],
                                     -0.5:['diagonal_left_up','diagonal_left_down','diagonal_left_level'],
                                     -1.0:['left_up','left_down','left_level']},
                             'pitch':{0:['left_level','center_level','diagonal_left_level','diagonal_right_level','right_level'],
                                      1:['left_up','right_up','center_up','diagonal_right_up','diagonal_left_up'],
                                      -1:['left_down','right_down','diagnoal_right_down','diagonal_left_down','center_down']}
                             }


    action_category_values = {'yaw': {'diagonal_left_down': -0.5, 'diagonal_left_level': -0.5,
                                      'diagonal_left_up': -0.5, 'diagonal_right_up': 0.5,
                                      'diagonal_right_down': 0.5, 'diagonal_right_level': 0.5,
                                      'left_down': -1.0, 'left_up': -1.0,
                                      'left_level': -1.0, 'right_up': 1.0,
                                      'right_level': 1.0, 'right_down': 1.0,
                                      'center_level': 0.0, 'center_down': 0.0, 'center_up': 0.0},
                              'pitch': {'diagonal_left_down': -1.0, 'diagonal_left_level': 0.0,
                                        'diagonal_left_up': 1.0, 'diagonal_right_up': 1.0,
                                        'diagonal_right_down': -1.0, 'diagonal_right_level': 0.0,
                                        'left_down': -1.0, 'left_up': 1.0,
                                        'left_level': 0.0, 'right_up': 1.0,
                                        'right_level': 0.0, 'right_down': -1.0,
                                        'center_level': 0.0, 'center_down': -1.0, 'center_up': 1.0}}
    yaw = actr.chunk_slot_value(b, 'yaw')
    pitch = actr.chunk_slot_value(b, 'pitch')
    drop = actr.chunk_slot_value(b, 'drop')

    action = None
    if drop > abs(yaw) and drop > abs(pitch):
        action = 'drop'
    else:
        #round them to the nearest 0.5 for yaw and near 1.0 for pitch
        yaw = round(yaw * 2.0) / 2
        pitch = round(pitch)
        yaw_categories = yaw_pitch_to_category['yaw'][yaw]
        pitch_categories = yaw_pitch_to_category['pitch'][pitch]
        action = [x for x in yaw_categories if x in pitch_categories][0]
        # print('stop...')



    actr.erase_buffer('blending')


    # for key in action_choice:
    #     action_choice[key] = access_by_key(key.upper(), blend_return)
    # action = max(action_choice.items(), key=operator.itemgetter(1))[0]

    return combos_to_actions[action]

    # action_choice_pitch = {'up': 0, 'down': 0, 'level': 0}
    # action_choice_yaw = {'left': 0, 'diagonal_left': 0, 'center': 0, 'diagonal_right': 0, 'right': 0}
    # action_choice_pitch['up'] = access_by_key('UP', blend_return)
    # action_choice_pitch['down'] = access_by_key('DOWN', blend_return)
    # action_choice_pitch['level'] = access_by_key('LEVEL', blend_return)
    # pitch_action = max(action_choice_pitch.items(), key=operator.itemgetter(1))[0]

    # action_choice_yaw['left'] = access_by_key('LEFT', blend_return)
    # action_choice_yaw['diagonal_left'] = access_by_key('DIAGONAL_LEFT', blend_return)
    # action_choice_yaw['center'] = access_by_key('CENTER', blend_return)
    # action_choice_yaw['diagonal_right'] = access_by_key('DIAGONAL_RIGHT', blend_return)
    # action_choice_yaw['right'] = access_by_key('RIGHT', blend_return)
    # yaw_action = max(action_choice_yaw.items(), key=operator.itemgetter(1))[0]


    # drop_action = access_by_key('DROP',blend_return)
    # if drop_action > action_choice_pitch[pitch_action] and drop_action > action_choice_yaw[yaw_action]:
    #     return combos_to_actions[('drop')]
    # else:
    #     return combos_to_actions[(pitch_action,yaw_action)]



    print("here.")

    #
    #
    # t = actr.get_history_data("blending-times")
    #
    # MP = actr.get_parameter_value(':mp')
    # # #get t
    # t = access_by_key('TEMPERATURE', d[0][1])
    # # #the values
    # # vs = [actr.chunk_slot_value(x,'value') for x in chunk_names]
    # #
    # #factors = ['current_altitude', 'heading', 'view_left', 'view_diagonal_left', 'view_center', 'view_diagonal_right', 'view_right']
    # factors = ['current_altitude', 'view_left', 'view_diagonal_left', 'view_center', 'view_diagonal_right', 'view_right']
    # # factors = ['needsFood', 'needsWater']
    # result_factors = ['action']
    # # result_factors = ['food','water']
    # results = compute_S(d, factors)  # ,'f3'])
    # results_dict = {'action':{}}
    # results_dict['blend'] = blend_return
    # for sums, result_factor in zip(results, result_factors):
    #     #print("For", result_factor)
    #
    #     for s, factor in zip(sums, factors):
    #         results_dict['action'][factor] = MP / t * sum(s)
    #         #print(factor, MP / t * sum(s))
    #
    # return results_dict

    #print("actual value is", actr.chunk_slot_value('OBSERVATION0', 'ACTUAL'))

    #print("done")

#reset ACTR before game starts
reset_actr()

class Runner(object):
    def __init__(
            self,
            envs,
            agent: ActorCriticAgent,
            n_steps=5,
            discount=0.99,
            do_training=True,
            ppo_par: PPORunParams = None,
            n_envs=1,
            policy_type = None
    ):
        self.envs = envs
        self.n_envs = n_envs
        self.agent = agent
        self.obs_processer = ObsProcesser()
        self.action_processer = ActionProcesser(dim=flags.FLAGS.resolution)
        self.n_steps = n_steps
        self.discount = discount
        self.do_training = do_training
        self.ppo_par = ppo_par
        self.batch_counter = 0
        self.episode_counter = 0
        self.score = 0.0
        # self.policy_type = FullyConvPolicy if ( (policy_type == 'FullyConv') or (policy_type == 'Relational')) else MetaPolicy
        if policy_type == 'FullyConv':
            self.policy_type = FullyConvPolicy
        elif policy_type == 'Relational':
            self.policy_type = RelationalPolicy
        else:
            self.policy_type = MetaPolicy

        assert self.agent.mode in [ACMode.A2C, ACMode.PPO]
        self.is_ppo = self.agent.mode == ACMode.PPO
        if self.is_ppo:
            assert ppo_par is not None
            # assert n_steps * envs.n_envs % ppo_par.batch_size == 0
            # assert n_steps * envs.n_envs >= ppo_par.batch_size
            assert n_steps * self.envs.num_envs % ppo_par.batch_size == 0
            assert n_steps * self.envs.num_envs >= ppo_par.batch_size
            self.ppo_par = ppo_par

    def reset(self):
        #self.score = 0.0
        obs = self.envs.reset()
        self.latest_obs = self.obs_processer.process(obs)

    def reset_demo(self):
        #self.score = 0.0
        obs = self.envs.reset()
        self.latest_obs = self.obs_processer.process([obs])

    def _log_score_to_tb(self, score):
        summary = tf.Summary()
        summary.value.add(tag='sc2/episode_score', simple_value=score)
        self.agent.summary_writer.add_summary(summary, self.episode_counter)

    def _handle_episode_end(self, timestep, length, last_step_r):
        #(MINE) This timestep is actually the last set of feature observations
        #score = timestep.observation["score_cumulative"][0]
        #self.score = (self.score + timestep) # //self.episode_counter # It is zero at the beginning so you get inf
        self.score = timestep
        print(">>>>>>>>>>>>>>>episode %d ended. Score %f | Total Steps %d | Last step Reward %f" % (self.episode_counter, self.score, length, last_step_r))
        self._log_score_to_tb(self.score) # logging score to tensorboard
        self.episode_counter += 1 # Is not used for stopping purposes judt for printing. You train for a number of batches (nsteps+training no matter reset)
        #self.reset() # Error if Monitor doesnt have the option to reset without an env to be done (THIS RESETS ALL ENVS!!! YOU NEED remot.send(env.reset) to reset a specific env. Else restart within the env

    def _train_ppo_epoch(self, full_input):
        total_obs = self.n_steps * self.envs.num_envs
        shuffle_idx = np.random.permutation(total_obs)
        batches = dict_of_lists_to_list_of_dicst({
            k: np.split(v[shuffle_idx], total_obs // self.ppo_par.batch_size)
            for k, v in full_input.items()
        })
        if self.policy_type == MetaPolicy: # We take out the if from the loop so you choose trainer BEFORE getting into the batch loop
            for b in batches:
                self.agent.train_recurrent(b)
        else:
            for b in batches:
                self.agent.train(b)

    def _train_ppo_recurrent_epoch(self, full_input, rnn_state):
        # HE SHUFFLES SO BE CAREFUL!!! RECHECK IT: rnn_state might need to get in the full_input
        total_obs = self.n_steps * self.envs.num_envs
        shuffle_idx = np.random.permutation(total_obs)
        batches = dict_of_lists_to_list_of_dicst({
            k: np.split(v[shuffle_idx], total_obs // self.ppo_par.batch_size)
            for k, v in full_input.items()
        })
        for b in batches:
            self.agent.train_recurrent(b, rnn_state) # IMPORTANT : όταν κανεις training δεν χρειαζεσαι την rnn_State, ξεκινας απο το 0 και αθτη παιρνη την μορφή πουπρεπει να εχει

    def run_batch(self):
        #(MINE) MAIN LOOP!!!
        # The reset is happening through Monitor (except the first one of the first batch (is in hte run_agent)
        mb_actions = []
        mb_obs = []
        mb_values = np.zeros((self.envs.num_envs, self.n_steps + 1), dtype=np.float32)
        mb_rewards = np.zeros((self.envs.num_envs, self.n_steps), dtype=np.float32)
        mb_done = np.zeros((self.envs.num_envs, self.n_steps), dtype=np.int32)

        latest_obs = self.latest_obs # (MINE) =state(t)

        for n in range(self.n_steps):
            # could calculate value estimate from obs when do training
            # but saving values here will make n step reward calculation a bit easier
            action_ids, value_estimate = self.agent.step(latest_obs)
            print('|step:', n, '|actions:', action_ids)  # (MINE) If you put it after the envs.step the SUCCESS appears at the envs.step so it will appear oddly
            # (MINE) Store actions and value estimates for all steps
            mb_values[:, n] = value_estimate
            mb_obs.append(latest_obs)
            mb_actions.append((action_ids))
            # (MINE)  do action, return it to environment, get new obs and reward, store reward
            #actions_pp = self.action_processer.process(action_ids) # Actions have changed now need to check: BEFORE: actions.FunctionCall(actions.FUNCTIONS.no_op.id, []) NOW: actions.FUNCTIONS.no_op()
            obs_raw = self.envs.step(action_ids)
            #obs_raw.reward = reward
            latest_obs = self.obs_processer.process(obs_raw[0]) # For obs_raw as tuple! #(MINE) =state(t+1). Processes all inputs/obs from all timesteps (and envs)
            #print('-->|rewards:', np.round(np.mean(obs_raw[1]), 3))
            mb_rewards[:, n] = [t for t in obs_raw[1]]
            mb_done[:, n] = [t for t in obs_raw[2]]

            #Check for all t (timestep/observation in obs_raw which t has the last state true, meaning it is the last state
            # IF MAX_STEPS OR GOAL REACHED
            # You can use as below for obs_raw[4] which is success of failure
            #print(obs_raw[2])
            indx=0 # env count
            for t in obs_raw[2]: # Monitor returns additional stuff such as epis_reward and epis_length etc apart the obs, r, done, info
                #obs_raw[2] = done = [True, False, False, True,...] each element corresponds to an env
                if t == True: # done=true
                    # Put reward in scores
                    epis_reward = obs_raw[3][indx]['episode']['r']
                    epis_length = obs_raw[3][indx]['episode']['l']
                    last_step_r = obs_raw[1][indx]
                    self._handle_episode_end(epis_reward, epis_length, last_step_r) # The printing score process is NOT a parallel process apparrently as you input every reward (t) independently
                indx = indx + 1 # finished envs count
            # for t in obs_raw:
            #     if t.last():
            #         self._handle_episode_end(t)

        #print(">> Avg. Reward:",np.round(np.mean(mb_rewards),3))
        mb_values[:, -1] = self.agent.get_value(latest_obs) # We bootstrap from last step if not terminal! although he doesnt use any check here

        n_step_advantage = general_n_step_advantage(
            mb_rewards,
            mb_values,
            self.discount,
            mb_done,
            lambda_par=self.ppo_par.lambda_par if self.is_ppo else 1.0
        )

        full_input = {
            # these are transposed because action/obs
            # processers return [time, env, ...] shaped arrays
            FEATURE_KEYS.advantage: n_step_advantage.transpose(),
            FEATURE_KEYS.value_target: (n_step_advantage + mb_values[:, :-1]).transpose() # if you add to the advantage the value you get the target for your value function training. Check onenote in APL-virtual
        }
        #(MINE) Probably we combine all experiences from every worker below
        full_input.update(self.action_processer.combine_batch(mb_actions))
        full_input.update(self.obs_processer.combine_batch(mb_obs))
        full_input = {k: combine_first_dimensions(v) for k, v in full_input.items()}

        if not self.do_training:
            pass
        elif self.agent.mode == ACMode.A2C:
            self.agent.train(full_input)
        elif self.agent.mode == ACMode.PPO:
            for epoch in range(self.ppo_par.n_epochs):
                self._train_ppo_epoch(full_input)
            self.agent.update_theta()

        self.latest_obs = latest_obs
        self.batch_counter += 1 # It is used only for printing reasons as the outer while loop takes care to stop the number of batches
        print('Batch %d finished' % self.batch_counter)
        sys.stdout.flush()

    def run_meta_batch(self):
        mb_actions = []
        mb_obs = []
        mb_values = np.zeros((self.envs.num_envs, self.n_steps + 1), dtype=np.float32)
        mb_rewards = np.zeros((self.envs.num_envs, self.n_steps), dtype=np.float32) # n x d array (ndarray)
        mb_done = np.zeros((self.envs.num_envs, self.n_steps), dtype=np.int32)
        # EVERYTHING IS HAPPENING ON PARALLEL!!!
        r_=np.zeros((self.envs.num_envs, 1), dtype=np.float32) # Instead of 1 you might use n_steps
        a_=np.zeros((self.envs.num_envs), dtype=np.int32)
        latest_obs = self.latest_obs # (MINE) =state(t)
        # rnn_state = self.agent.theta.state_init
        rnn_state = self.agent.theta.state_init
        for n in range(self.n_steps):
            action_ids, value_estimate, rnn_state_new = self.agent.step_recurrent(latest_obs, rnn_state, r_, a_) # Automatically returns [num_envs, outx] for each outx you want
            print('|step:', n, '|actions:', action_ids)
            # (MINE) Store actions and value estimates for all steps
            mb_values[:, n] = value_estimate
            mb_obs.append(latest_obs)
            mb_actions.append((action_ids))
            # (MINE)  do action, return it to environment, get new obs and reward, store reward
            obs_raw = self.envs.step(action_ids)
            latest_obs = self.obs_processer.process(obs_raw[0]) # For obs_raw as tuple! #(MINE) =state(t+1). Processes all inputs/obs from all timesteps (and envs)

            rnn_state = rnn_state_new
            r_ = obs_raw[1] # (nenvs,) but you need (nenvs,1)
            r_ = np.reshape(r_,[self.envs.num_envs,1]) # gets into recurrency as [nenvs,1] # The 1 might be used as timestep
            a_ = action_ids

            mb_rewards[:, n] = [t for t in obs_raw[1]]
            mb_done[:, n] = [t for t in obs_raw[2]]

            # Shouldnt this part below be OUT of the nstep loop? NO: You check if done=True and you extract the additional info that Monitor outputs
            indx=0 # env count
            for t in obs_raw[2]: # Monitor returns additional stuff such as epis_reward and epis_length etc apart the obs, r, done, info
                # obs_raw[2] = done = [True, False, False, True,...] each element corresponds to an env (index gives the env)
                if t == True: # done=true
                    # Put reward in scores
                    epis_reward = obs_raw[3][indx]['episode']['r']
                    epis_length = obs_raw[3][indx]['episode']['l']
                    last_step_r = obs_raw[1][indx]
                    self._handle_episode_end(epis_reward, epis_length, last_step_r) # The printing score process is NOT a parallel process apparrently as you input every reward (t) independently
                    # Here you have to reset the rnn_state of that env: rnn_state[i] = 0 or smth like that
                    rnn_state[0][indx] = np.zeros(256)
                    rnn_state[1][indx] = np.zeros(256)
                    #reset the relevant r_ and a_
                    r_[indx] = 0
                    a_[indx] = 0
                indx = indx + 1 # finished envs count

        mb_values[:, -1] = self.agent.get_recurrent_value(latest_obs, rnn_state, r_, a_) # Put at last slot the estimated future expected reward for bootstrap the Vt+1

        n_step_advantage = general_n_step_advantage(
            mb_rewards,
            mb_values,
            self.discount,
            mb_done,
            lambda_par=self.ppo_par.lambda_par if self.is_ppo else 1.0
        )
        # prev_rewards = [0] + mb_rewards[:, :-1]#.tolist() # from the rewards you take out the last element and replace it with 0
        prev_rewards = np.c_[np.zeros((self.envs.num_envs, 1), dtype=np.float32), mb_rewards[:, :-1]]
        # Below we add one zero action element and we take out the at so we get at=0:t-1
        prev_actions = [np.zeros((self.envs.num_envs), dtype=np.int32)] + mb_actions[:-1] # You have to pad this probably to have equal lengths with your data in terms of nsteps
        full_input = {
            FEATURE_KEYS.advantage: n_step_advantage.transpose(),
            FEATURE_KEYS.value_target: (n_step_advantage + mb_values[:, :-1]).transpose()
        }

        full_input.update(self.action_processer.combine_batch(mb_actions))
        full_input.update(self.obs_processer.combine_batch(mb_obs))
        full_input.update(self.action_processer.combine_batch(prev_actions))
        full_input.update(self.action_processer.combine_batch(prev_rewards))
        full_input = {k: combine_first_dimensions(v) for k, v in full_input.items()}

        if not self.do_training:
            pass
        elif self.agent.mode == ACMode.A2C:
            if self.policy_type == MetaPolicy:
                self.agent.train_recurrent(full_input,prev_rewards,prev_actions)
            else:
                self.agent.train(full_input)
        elif self.agent.mode == ACMode.PPO:
            for epoch in range(self.ppo_par.n_epochs):
                self._train_ppo_epoch(full_input)
            self.agent.update_theta()

        self.latest_obs = latest_obs
        self.batch_counter += 1
        print('Batch %d finished' % self.batch_counter)
        sys.stdout.flush()

    def run_trained_batch(self):
        step_data = {}

        #gameDisplay.fill((1, 50, 130))
        # STATE, ACTION, REWARD, NEXT STATE
        # YOU NEED TO DISPLAY FIRST IMAGE HERE AS YOU HAVE RESETED AND THERE ARE OBS THERE AS WELL (YOUR FIRST ONES!)
        #sleep(2.0)
        latest_obs = self.latest_obs # (MINE) =state(t)

        # action = agent(state)
        action_ids, value_estimate, fc, action_probs = self.agent.step_eval(latest_obs) # (MINE) AGENT STEP = INPUT TO NN THE CURRENT STATE AND OUTPUT ACTION
        step_data['volume'] = np.array(self.envs.map_volume['vol'], copy=True)
        step_data['heading'] = self.envs.heading
        step_data['hiker'] = self.envs.hiker_position
        step_data['altitude'] = self.envs.altitude
        step_data['drone'] = np.where(
        step_data['volume'] == self.envs.map_volume['feature_value_map']['drone'][self.envs.altitude]['val'])
        step_data['action_probs'] = action_probs

        step_data['fc'] = fc
        #     interp_dict['fc'].transform(fc)
        # fc = step_data['fc']

        chunks_and_distances = []
        # look at the fc
        # for chunk in allchunks:
        #     fc_tuple = access_by_key('fc', chunk)
        #     fc_from_memory = fc_tuple[1]
        #     dist = np.linalg.norm(np.array(fc) - np.array(fc_from_memory))
        #     cos = spatial.distance.cosine(fc, fc_from_memory) * -1
        #     mink = spatial.distance.minkowski(fc, fc_from_memory, 1) * -1
        #     # sim = remap(dist, min(fc_distances), max(fc_distances), 0, 1) * -1
        #     chunks_and_distances.append([chunk, dist, cos, mink])
        #
        # # order the chunks_and_distances
        # chunks_and_distances = sorted(chunks_and_distances, key=operator.itemgetter(1))
        # # print("ok")

        network_action_ids = np.array(action_ids, copy=True)
        actr_observation = create_actr_observation(step_data)
        actr_action = handle_observation(actr_observation)

        action_ids = np.array([actr_action])
        # nav_runner.envs.step(action)

        #reset actr every step (load chunks, etc.)
        # reset_actr()
        if flags.FLAGS.reset_actr:
            reset_actr()
            flags.FLAGS.reset_actr = False

        print('|actions:', 'net', network_action_ids, 'actr', action_ids)

        obs_raw = self.envs.step(action_ids) # It will also visualize the next observation if all the episodes have ended as after success it retunrs the obs from reset
        latest_obs = self.obs_processer.process(obs_raw[0:-3])  # Take only the first element which is the rgb image and ignore the reward, done etc
        print('-->|rewards:', np.round(np.mean(obs_raw[1]), 3))

        # if obs_raw[2]: # done is True
        #     # for r in obs_raw[1]: # You will double count here as t
        #     self._handle_episode_end(obs_raw[1])  # The printing score process is NOT a parallel process apparrently as you input every reward (t) independently

        self.latest_obs = latest_obs # (MINE) state(t) = state(t+1), the usual s=s'
        self.batch_counter += 1
        #print('Batch %d finished' % self.batch_counter)
        sys.stdout.flush()
        return obs_raw[0:-3], action_ids[0], value_estimate[0], obs_raw[1], obs_raw[2], obs_raw[3], fc, action_probs
