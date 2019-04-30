import logging
import os
import shutil
import sys
from datetime import datetime
from time import sleep
import numpy as np
import pickle
import matplotlib.pyplot as plt

from absl import flags
from actorcritic.agent import ActorCriticAgent, ACMode
from actorcritic.runner import Runner, PPORunParams
# from common.multienv import SubprocVecEnv, make_sc2env, SingleEnv, make_env

import tensorflow as tf
# import tensorflow.contrib.eager as tfe
# tf.enable_eager_execution()

#import argparse
from baselines import logger
from baselines.bench import Monitor
#from baselines.common.misc_util import boolean_flag
from baselines.common import set_global_seeds
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
#from baselines.common.vec_env.vec_normalize import VecNormalize
import gym
#import gym_gridworld
#from gym_grid.envs import GridEnv
import gym_gridworld

import copy
import math

import json
import operator
import actr #Version 7.11.1 tested (may work on others)

FLAGS = flags.FLAGS
flags.DEFINE_bool("visualize", False, "Whether to render with pygame.")
flags.DEFINE_integer("resolution", 32, "Resolution for screen and minimap feature layers.")
flags.DEFINE_integer("step_mul", 100, "Game steps per agent step.")
flags.DEFINE_integer("n_envs", 20, "Number of environments to run in parallel")
flags.DEFINE_integer("episodes", 10, "Number of complete episodes")
flags.DEFINE_integer("n_steps_per_batch", 32,
    "Number of steps per batch, if None use 8 for a2c and 128 for ppo")  # (MINE) TIMESTEPS HERE!!! You need them cauz you dont want to run till it finds the beacon especially at first episodes - will take forever
flags.DEFINE_integer("all_summary_freq", 50, "Record all summaries every n batch")
flags.DEFINE_integer("scalar_summary_freq", 5, "Record scalar summaries every n batch")
flags.DEFINE_string("checkpoint_path", "_files/models", "Path for agent checkpoints")
flags.DEFINE_string("summary_path", "_files/summaries", "Path for tensorboard summaries")
flags.DEFINE_string("model_name", "Drop_Agent", "Name for checkpoints and tensorboard summaries")
flags.DEFINE_integer("K_batches", 10000, # Batch is like a training epoch!
    "Number of training batches to run in thousands, use -1 to run forever") #(MINE) not for now
flags.DEFINE_string("map_name", "DefeatRoaches", "Name of a map to use.")
flags.DEFINE_float("discount", 0.95, "Reward-discount for the agent")
flags.DEFINE_boolean("training", False,
    "if should train the model, if false then save only episode score summaries")
flags.DEFINE_enum("if_output_exists", "overwrite", ["fail", "overwrite", "continue"],
    "What to do if summary and model output exists, only for training, is ignored if notraining")
flags.DEFINE_float("max_gradient_norm", 500.0, "good value might depend on the environment")
flags.DEFINE_float("loss_value_weight", 0.5, "good value might depend on the environment") # orig:1.0
flags.DEFINE_float("entropy_weight_spatial", 0.00000001,
    "entropy of spatial action distribution loss weight") # orig:1e-6
flags.DEFINE_float("entropy_weight_action", 0.001, "entropy of action-id distribution loss weight") # orig:1e-6
flags.DEFINE_float("ppo_lambda", 0.95, "lambda parameter for ppo")
flags.DEFINE_integer("ppo_batch_size", None, "batch size for ppo, if None use n_steps_per_batch")
flags.DEFINE_integer("ppo_epochs", 3, "epochs per update")
flags.DEFINE_enum("agent_mode", ACMode.A2C, [ACMode.A2C, ACMode.PPO], "if should use A2C or PPO")

### NEW FLAGS ####
#flags.DEFINE_bool("render", True, "Whether to render with pygame.")
# point_flag.DEFINE_point("feature_screen_size", 32,
#                         "Resolution for screen feature layers.")
# point_flag.DEFINE_point("feature_minimap_size", 32,
#                         "Resolution for minimap feature layers.")
flags.DEFINE_integer("rgb_screen_size", 128,
                        "Resolution for rendered screen.") # type None if you want only features
# point_flag.DEFINE_point("rgb_minimap_size", 64,
#                         "Resolution for rendered minimap.") # type None if you want only features
# flags.DEFINE_enum("action_space", 'FEATURES', sc2_env.ActionSpace._member_names_,  # pylint: disable=protected-access, # type None if you want only features
#                   "Which action space to use. Needed if you take both feature "
#                   "and rgb observations.") # "RGB" or "FEATURES", None if only one is specified in the dimensions
# flags.DEFINE_bool("use_feature_units", True,
#                   "Whether to include feature units.")
# flags.DEFINE_bool("disable_fog", False, "Whether to disable Fog of War.")

flags.DEFINE_integer("max_agent_steps", 0, "Total agent steps.")
#flags.DEFINE_integer("game_steps_per_episode", None, "Game steps per episode.")
#flags.DEFINE_integer("max_episodes", 0, "Total episodes.")
#flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

# flags.DEFINE_string("agent", "pysc2.agents.random_agent.RandomAgent",
#                     "Which agent to run, as a python path to an Agent class.")
# flags.DEFINE_enum("agent_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
#                   "Agent 1's race.")
#
# flags.DEFINE_string("agent2", "Bot", "Second agent, either Bot or agent class.")
# flags.DEFINE_enum("agent2_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
#                   "Agent 2's race.")
# flags.DEFINE_enum("difficulty", "very_easy", sc2_env.Difficulty._member_names_,  # pylint: disable=protected-access
#                   "If agent2 is a built-in Bot, it's strength.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
#flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")

#flags.DEFINE_string("map", None, "Name of a map to use.")




#ADD some global stuff for ACT-R

#stats
stats = {'crashes':0,'successes':0}

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
           'altitude':[],}


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

combos_to_actions = {('down','left'):0,('down','diagonal_left'):1,('down','center'):2,
                     ('down','diagonal_right'):3,('down','right'):4,
                     ('level','left'):5,('level','diagonal_left'):6,('level','center'):7,
                     ('level','diagonal_right'):8,('level','right'):9,
                     ('up','left'):10,('up','diagonal_left'):11,('up','center'):12,
                     ('up','diagonal_right'):13,('up','right'):14,('drop'):15}


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
    if val1 == val2:
        return 0
    if val1 == 'NAV' and val2 == 'DROP':
        return -5

    max_val = max(min_max[val1[0].lower()])
    min_val = min(min_max[val1[0].lower()])

    if val1 == None:
        return None
    value1 = val1[1]
    value2 = val2[1]
    #max_val = 4#max(map(max, zip(*feature_sets)))
    #min_val = 1#min(map(min, zip(*feature_sets)))
    #print("max,min,val1,val2",max_val,min_val,val1,val2)
    #The intent looks to be to transpose the values to a 0,1 range so that the math is all the same
    val1_t = (((value1 - min_val) * (0 + 1)) / (max_val - min_val)) + 0
    val2_t = (((value2 - min_val) * (0 + 1)) / (max_val - min_val)) + 0
    #print("val1_t,val2_t", val1_t, val2_t)
    #print("sim returning", abs(val1_t - val2_t) * -1)
    #print("sim returning", ((val1_t - val2_t)**2) * - 1)
    #return float(((val1_t - val2_t)**2) * - 1)
    #return abs(val1_t - val2_t) * - 1
    #return 0
    #print("sim returning", abs(val1_t - val2_t) * - 1)
    #return abs(val1_t - val2_t) * -1
    #print("sim returning", (abs(value1 - value2) * - 1)/max_val)
    return_value = abs(val1_t - val2_t) * -1#/max_val
    return return_value#(abs(value1 - value2) * - 1)/max_val

    #print("sim returning", abs(val1 - val2) / (max_val - min_val) * - 1)
    #return abs(val1 - val2) / (max_val - min_val) * - 1


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

    model_name = 'egocentric-salience.lisp'
    model_path = '/Users/paulsomers/COGLE/gym-gridworld/'

    chunk_file_name = 'chunks_maxdistance.pkl'
    #chunk_path = os.path.join(model_path,'data')
    chunk_path = ''
    actr.add_command('similarity_function',similarity)
    actr.load_act_r_model(os.path.join(model_path,model_name))
    actr.record_history("blending-trace")


    #load all the chunks
    allchunks = pickle.load(open(os.path.join(chunk_path,chunk_file_name),'rb'))
    for chunk in allchunks:
        #chunk1 = chunk[0:4]
        #chunk2 = chunk[6:]
        #chunk = chunk1 + chunk2
        #alt = chunk[3][1]
        #chunk[5][1] = chunk[5][1] - alt
        #chunk[7][1] = chunk[7][1] - alt
        #chunk[9][1] = chunk[9][1] - alt
        #chunk[11][1] = chunk[11][1] - alt
        #chunk[13][1] = chunk[13][1] - alt
        chunk = [float(x) if type(x) == np.float64 else x for x in chunk]
        chunk = [int(x) if type(x) == np.int64 else x for x in chunk]
        actr.add_dm(chunk)

    ignore_list = ['left', 'diagonal_left', 'center', 'diagonal_right', 'right', 'type', 'drop', 'up', 'down', 'level']
    for chunk in allchunks:
        for x, y in zip(*[iter(chunk)] * 2):
            #x, y[1]
            if not x in ignore_list and not x == 'isa':
                if y[1] not in min_max[x]:
                    min_max[x].append(y[1])
    #print('asf')
    print("reset done.")

def create_actr_observation(step):
    transposes = ['ego_left', 'ego_diagonal_left', 'ego_center', 'ego_diagonal_right', 'ego_right',
                  'distance_to_hiker', 'altitude']

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
    chunk.extend(['ego_left', ['ego_left',altitudes[0] - alt],
                  'ego_diagonal_left', ['ego_diagonal_left',altitudes[1] - alt],
                  'ego_center',  ['ego_center',altitudes[2] - alt],
                  'ego_diagonal_right', ['ego_diagonal_right', altitudes[3] - alt],
                  'ego_right', ['ego_right',altitudes[4] - alt]])
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
    chunk = [float(x) if type(x) == np.float64 else x for x in chunk]
    chunk = [int(x) if type(x) == np.int64 else x for x in chunk]
    for trans in transposes:
        index_of_value = chunk.index(trans) + 1
        chunk[index_of_value] = remap(chunk[index_of_value], min(min_max[trans]),max(min_max[trans]),0,1)
    #transponse the transposes values to zero to 1 range

    return chunk

def handle_observation(observation):
    '''observation should have chunk format'''

    chunk = actr.define_chunks(observation)

    print("converting")
    # actr.schedule_simple_event_now("set-buffer-chunk",
    #                                ['imaginal', chunk[0]])
    actr.schedule_set_buffer_chunk('imaginal',chunk[0],0)
    actr.run(10)

    d = actr.get_history_data("blending-trace")
    d = json.loads(d)

    # first add the blend to the results dictionary
    blend_return = access_by_key('RESULT-CHUNK', d[0][1])
    #HACK - carry out the action here.
    action_choice_pitch = {'up': 0, 'down': 0, 'level': 0}
    action_choice_yaw = {'left': 0, 'diagonal_left': 0, 'center': 0, 'diagonal_right': 0, 'right': 0}
    action_choice_pitch['up'] = access_by_key('UP', blend_return)
    action_choice_pitch['down'] = access_by_key('DOWN', blend_return)
    action_choice_pitch['level'] = access_by_key('LEVEL', blend_return)
    pitch_action = max(action_choice_pitch.items(), key=operator.itemgetter(1))[0]

    action_choice_yaw['left'] = access_by_key('LEFT', blend_return)
    action_choice_yaw['diagonal_left'] = access_by_key('DIAGONAL_LEFT', blend_return)
    action_choice_yaw['center'] = access_by_key('CENTER', blend_return)
    action_choice_yaw['diagonal_right'] = access_by_key('DIAGONAL_RIGHT', blend_return)
    action_choice_yaw['right'] = access_by_key('RIGHT', blend_return)
    yaw_action = max(action_choice_yaw.items(), key=operator.itemgetter(1))[0]


    drop_action = access_by_key('DROP',blend_return)
    if drop_action > action_choice_pitch[pitch_action] and drop_action > action_choice_yaw[yaw_action]:
        return combos_to_actions[('drop')]
    else:
        return combos_to_actions[(pitch_action,yaw_action)]



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

def plot(result_dict):
    salience_dict = result_dict['action']
    objects = list(salience_dict.keys())
    objects = objects[1:]
    performance = [x for x in list(salience_dict.values())[1:]]

    y_pos = np.arange(len(objects))
    #performance = [10, 8, 6, 4, 2, 1]

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    #plt.xticks(y_pos, objects)
    plt.ylabel('Salience')
    #plt.title('Programming language usage')

    plt.show()
    print("plotted.")
    plt.savefig('egocentric-salience-bars.eps', format='eps', dpi=300)


reset_actr()


FLAGS(sys.argv)

#TODO this runner is maybe too long and too messy..
full_chekcpoint_path = os.path.join(FLAGS.checkpoint_path, FLAGS.model_name)

if FLAGS.training:
    full_summary_path = os.path.join(FLAGS.summary_path, FLAGS.model_name)
else:
    full_summary_path = os.path.join(FLAGS.summary_path, "no_training", FLAGS.model_name)


def check_and_handle_existing_folder(f):
    if os.path.exists(f):
        if FLAGS.if_output_exists == "overwrite":
            shutil.rmtree(f)
            print("removed old folder in %s" % f)
        elif FLAGS.if_output_exists == "fail":
            raise Exception("folder %s already exists" % f)


def _print(i):
    print(datetime.now())
    print("# batch %d" % i)
    sys.stdout.flush()


def _save_if_training(agent):
    agent.save(full_chekcpoint_path)
    agent.flush_summaries()
    sys.stdout.flush()

def make_custom_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            # Monitor should take care of reset!
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=False) # SUBPROC NEEDS 4 OUTPUS FROM STEP FUNCTION
            return env
        return _thunk
    #set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

def main():
    if FLAGS.training:
        check_and_handle_existing_folder(full_chekcpoint_path)
        check_and_handle_existing_folder(full_summary_path)

    #(MINE) Create multiple parallel environements (or a single instance for testing agent)
    if FLAGS.training and FLAGS.n_envs != 1:
        #envs = SubprocVecEnv((partial(make_sc2env, **env_args),) * FLAGS.n_envs)
        #envs = SubprocVecEnv([make_env(i,**env_args) for i in range(FLAGS.n_envs)])
        envs = make_custom_env('gridworld-v0', FLAGS.n_envs, 1)
    else:
        #envs = make_custom_env('gridworld-v0', 1, 1)
        envs = gym.make('gridworld-v0')

        #envs = SingleEnv(make_sc2env(**env_args))
    #envs = gym.make('gridworld-v0')
    # envs = SubprocVecEnv([make_env(i) for i in range(FLAGS.n_envs)])
    # envs = VecNormalize(env)
    # use for debugging 'Breakout-v0', Grid-v0, gridworld-v0
    #envs = VecFrameStack(make_custom_env('gridworld-v0', FLAGS.n_envs, 1), 1) # One is number of frames to stack within each env
    #envs = make_custom_env('gridworld-v0', FLAGS.n_envs, 1)
    print("Requested environments created successfully")
    #env = gym.make('gridworld-v0')
    tf.reset_default_graph()
    # The following lines fix the problem with using more than 2 envs!!!
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    # #sess = tf.Session()
    #
    # agent = ActorCriticAgent(
    #     mode=FLAGS.agent_mode,
    #     sess=sess,
    #     spatial_dim=FLAGS.resolution, # Here you pass the resolution which is used in the step for the output probabilities map
    #     unit_type_emb_dim=5,
    #     loss_value_weight=FLAGS.loss_value_weight,
    #     entropy_weight_action_id=FLAGS.entropy_weight_action,
    #     entropy_weight_spatial=FLAGS.entropy_weight_spatial,
    #     scalar_summary_freq=FLAGS.scalar_summary_freq,
    #     all_summary_freq=FLAGS.all_summary_freq,
    #     summary_path=full_summary_path,
    #     max_gradient_norm=FLAGS.max_gradient_norm,
    #     num_actions=envs.action_space.n
    # )
    # Make drop agent
    drop_agent = ActorCriticAgent(
        mode=FLAGS.agent_mode,
        sess=[],
        graph=[],
        spatial_dim=FLAGS.resolution,
        # Here you pass the resolution which is used in the step for the output probabilities map
        unit_type_emb_dim=5,
        loss_value_weight=FLAGS.loss_value_weight,
        entropy_weight_action_id=FLAGS.entropy_weight_action,
        entropy_weight_spatial=FLAGS.entropy_weight_spatial,
        scalar_summary_freq=FLAGS.scalar_summary_freq,
        all_summary_freq=FLAGS.all_summary_freq,
        summary_path=full_summary_path,
        max_gradient_norm=FLAGS.max_gradient_norm,
        num_actions=envs.action_space.n
    )

    drop_graph = tf.Graph()
    drop_agent.graph = drop_graph
    with drop_graph.as_default():
        drop_agent.build_model()

    drop_sess = tf.Session(graph=drop_graph)
    drop_agent.sess = drop_sess




    with drop_graph.as_default():
        if os.path.exists(full_chekcpoint_path):
            drop_agent.load(full_chekcpoint_path)

    # Build Agent
    #agent.build_model()
    # if os.path.exists(full_chekcpoint_path):
    #     agent.load(full_chekcpoint_path) #(MINE) LOAD!!!
    # else:
    #     agent.init()


    if FLAGS.n_steps_per_batch is None:
        n_steps_per_batch = 128 if FLAGS.agent_mode == ACMode.PPO else 8
    else:
        n_steps_per_batch = FLAGS.n_steps_per_batch

    if FLAGS.agent_mode == ACMode.PPO:
        ppo_par = PPORunParams(
            FLAGS.ppo_lambda,
            batch_size=FLAGS.ppo_batch_size or n_steps_per_batch,
            n_epochs=FLAGS.ppo_epochs
        )
    else:
        ppo_par = None

    drop_runner = Runner(
        envs=envs,
        agent=drop_agent,
        discount=FLAGS.discount,
        n_steps=n_steps_per_batch,
        do_training=FLAGS.training,
        ppo_par=ppo_par
    )

    # Make navigation agent
    nav_agent = ActorCriticAgent(
        mode=FLAGS.agent_mode,
        sess=[],
        graph=[],
        spatial_dim=FLAGS.resolution,
        # Here you pass the resolution which is used in the step for the output probabilities map
        unit_type_emb_dim=5,
        loss_value_weight=FLAGS.loss_value_weight,
        entropy_weight_action_id=FLAGS.entropy_weight_action,
        entropy_weight_spatial=FLAGS.entropy_weight_spatial,
        scalar_summary_freq=FLAGS.scalar_summary_freq,
        all_summary_freq=FLAGS.all_summary_freq,
        summary_path=full_summary_path,
        max_gradient_norm=FLAGS.max_gradient_norm,
        num_actions=15
    )
    nav_graph = tf.Graph()
    nav_agent.graph = nav_graph
    with nav_graph.as_default():
        nav_agent.build_model()

    nav_sess = tf.Session(graph=nav_graph)
    nav_agent.sess = nav_sess


    with nav_graph.as_default():
        if os.path.exists('_files/models/Nav_Agent'):
            nav_agent.load('_files/models/Nav_Agent')

    nav_runner = Runner(
        envs=envs,
        agent=nav_agent,
        discount=FLAGS.discount,
        n_steps=n_steps_per_batch,
        do_training=FLAGS.training,
        ppo_par=ppo_par
    )

    # runner.reset() # Reset env which means you get first observation. You need reset if you run episodic tasks!!! SC2 is not episodic task!!!

    if FLAGS.K_batches >= 0:
        n_batches = FLAGS.K_batches  # (MINE) commented here so no need for thousands * 1000
    else:
        n_batches = -1


    if FLAGS.training:
        i = 0

        try:
            runner.reset()
            while True:
                #runner.reset()
                if i % 500 == 0:
                    _print(i)
                if i % 4000 == 0:
                    _save_if_training(agent)
                runner.run_batch()  # (MINE) HERE WE RUN MAIN LOOP for while true
                #runner.run_batch_solo_env()
                i += 1
                if 0 <= n_batches <= i: #when you reach the certain amount of batches break
                    break
        except KeyboardInterrupt:
            pass
    else: # Test the agent
        # try:
        #     runner.reset_demo() # Cauz of differences in the arrangement of the dictionaries
        #     while runner.episode_counter <= (FLAGS.episodes - 1):
        #         #runner.reset()
        #         # You need the -1 as counting starts from zero so for counter 3 you do 4 episodes
        #         runner.run_trained_batch()  # (MINE) HERE WE RUN MAIN LOOP for while true
        # except KeyboardInterrupt:
        #     pass
        try:
            import pygame
            import time
            import random
            # pygame.font.get_fonts() # Run it to get a list of all system fonts
            display_w = 1200
            display_h = 720

            BLUE = (128, 128, 255)
            DARK_BLUE = (1, 50, 130)
            RED = (255, 192, 192)
            BLACK = (0, 0, 0)
            WHITE = (255, 255, 255)

            sleep_time = 0

            pygame.init()
            gameDisplay = pygame.display.set_mode((display_w, display_h))
            gameDisplay.fill(DARK_BLUE)
            pygame.display.set_caption('Neural Introspection')
            clock = pygame.time.Clock()

            def screen_mssg_variable(text, variable, area):
                font = pygame.font.SysFont('arial', 16)
                txt = font.render(text + str(variable), True, WHITE)
                gameDisplay.blit(txt, area)

            def process_img(img, x,y):
                # swap the axes else the image will not be the same as the matplotlib one
                img = img.transpose(1,0,2)
                surf = pygame.surfarray.make_surface(img)
                surf = pygame.transform.scale(surf, (300, 300))
                gameDisplay.blit(surf, (x, y))

            #all_data = [{'nav':[],'drop':[]}] * FLAGS.episodes #each entry is an episode, sorted into nav or drop steps
            all_data = [{'nav':[],'drop':[],'stuck':False} for x in range(FLAGS.episodes)]
            step_data = {}
            dictionary = {}
            running = True
            while nav_runner.episode_counter <= (FLAGS.episodes - 1) and running==True:
                print('Episode: ', nav_runner.episode_counter)






                # Init storage structures
                #dictionary[nav_runner.episode_counter] = {}
                #mb_obs = []
                #mb_actions = []
                #mb_flag = []
                #mb_representation = []
                #mb_fc = []
                mb_rewards = []
                #mb_values = []
                #mb_drone_pos = []
                #mb_heading = []
                # dictionary[nav_runner.episode_counter]['observations'] = {}
                # dictionary[nav_runner.episode_counter]['actions'] = []
                # dictionary[nav_runner.episode_counter]['flag'] = []

                nav_runner.reset_demo()  # Cauz of differences in the arrangement of the dictionaries
                map_xy = nav_runner.envs.map_image
                map_alt = nav_runner.envs.alt_view
                process_img(map_xy, 20, 20)
                process_img(map_alt, 20, 400)
                pygame.display.update()

                #dictionary[nav_runner.episode_counter]['hiker_pos'] = nav_runner.envs.hiker_position
                #dictionary[nav_runner.episode_counter]['map_volume'] = map_xy

                # Quit pygame if the (X) button is pressed on the top left of the window
                # Seems that without this for event quit doesnt show anything!!!
                # Also it seems that the pygame.event.get() is responsible to REALLY updating the screen contents
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                sleep(sleep_time)
                # Timestep counter
                t=0

                drop_flag = 0
                done = 0
                step_count = 0
                while done==0:
                    if step_count >= 50:
                        print("too many steps")
                        done = True
                        done2 = True
                        step_data['stuck'] = True
                        all_data[nav_runner.episode_counter]['stuck'] = True
                        all_data[nav_runner.episode_counter]['nav'].append(step_data)
                        break
                    step_count += 1
                    step_data = {'stuck':False}
                    #I need the egocentric view + hiker's position
                    #then drone steps, need action
                    step_data['volume'] = np.array(nav_runner.envs.map_volume['vol'],copy=True)
                    step_data['heading'] = nav_runner.envs.heading
                    step_data['hiker'] = nav_runner.envs.hiker_position
                    step_data['altitude'] = nav_runner.envs.altitude
                    step_data['drone'] = np.where(step_data['volume'] == nav_runner.envs.map_volume['feature_value_map']['drone'][nav_runner.envs.altitude]['val'])

                    actr_observation = create_actr_observation(step_data)
                    action = handle_observation(actr_observation)
                    nav_runner.envs.step(action)
                    reset_actr()

                    if nav_runner.envs.check_for_crash():
                        done = True
                        done2 = True
                        stats['crashes'] += 1
                    if nav_runner.envs.check_for_hiker():
                        done = True
                        done2 = True
                        stats['successes'] += 1

                    #angle test code
                    # x1,x2 = step_data['drone'][-2:]
                    # y1,y2 = step_data['hiker'][-2:]
                    #
                    # rads = math.atan2(y1-x1, y2-x2)
                    # deg = math.degrees(rads) + 90 - ((step_data['heading'] * 45) - 45)
                    # if deg < -180:
                    #     deg = deg + 360



                    #mb_obs.append(nav_runner.latest_obs)
                    #mb_flag.append(drop_flag)
                    #mb_heading.append(nav_runner.envs.heading)

                    #drone_pos =
                    #mb_drone_pos.append(drone_pos)

                    # dictionary[nav_runner.episode_counter]['observations'].append(nav_runner.latest_obs)
                    # dictionary[nav_runner.episode_counter]['flag'].append(drop_flag)

                    # RUN THE MAIN LOOP


                    #obs, action, value, reward, done, representation, fc, grad_V, grad_pi = nav_runner.run_trained_batch(drop_flag) # Just one step. There is no monitor here so no info section
                    #obs, action, value, reward, done, representation, fc, action_probs, grad_V_allo, grad_V_ego, mask_allo, mask_ego = nav_runner.run_trained_batch(drop_flag) # Just one step. There is no monitor here so no info section
                    # obs, action, value, reward, done, representation, fc, action_probs, grad_V_allo, grad_V_ego = nav_runner.run_trained_batch(drop_flag) # Just one step. There is no monitor here so no info section

                    # step_data['action'] = action
                    # step_data['reward'] = reward
                    # step_data['fc'] = fc
                    # step_data['action_probs'] = action_probs

                    #START WITH RANDOM AGENT
                    obs = nav_runner.envs.generate_observation()
                    #value = random.randint(5,15)
                    #nav_runner.envs.step(value)
                    reward = 0
                    value = 0

                    #all_data[nav_runner.episode_counter]['nav'].append(step_data)

                    # dictionary[nav_runner.episode_counter]['actions'].append(action)
                    #mb_actions.append(action)
                    mb_rewards.append(reward)
                    #mb_representation.append(representation)
                    #mb_fc.append(fc)
                    #mb_values.append(value)

                    # # Saliencies
                    # cmap = plt.get_cmap('viridis')
                    # grad_V_allo = cmap(grad_V_allo) # (100,100,4)
                    # grad_V_allo = np.delete(grad_V_allo, 3, 2) # (100,100,3)
                    # # grad_V = np.stack((grad_V,) * 3, -1)
                    # grad_V_allo = grad_V_allo * 255
                    # grad_V_allo = grad_V_allo.astype(np.uint8)
                    # process_img(grad_V_allo, 400, 20)
                    #
                    # grad_V_ego = cmap(grad_V_ego)  # (100,100,4)
                    # grad_V_ego = np.delete(grad_V_ego, 3, 2)  # (100,100,3)
                    # # grad_pi = np.stack((grad_pi,) * 3, -1)
                    # grad_V_ego = grad_V_ego * 255
                    # grad_V_ego = grad_V_ego.astype(np.uint8)
                    # process_img(grad_V_ego, 400, 400)
                    #
                    # # Masks
                    # masked_map_xy = map_xy
                    # masked_map_xy[mask_allo == 0] = 0
                    # process_img(masked_map_xy, 800, 20)
                    # masked_map_alt = map_alt
                    # masked_map_alt[mask_ego == 0] = 0
                    # process_img(masked_map_alt, 800, 400)

                    screen_mssg_variable("Value    : ", np.round(value,3), (168, 350))
                    screen_mssg_variable("Reward: ", np.round(reward,3), (168, 372))
                    pygame.display.update()
                    pygame.event.get()
                    sleep(sleep_time)

                    # BLIT!!!
                    # First Background covering everything from previous session
                    gameDisplay.fill(DARK_BLUE)
                    map_xy = obs['img']
                    map_alt = obs['nextstepimage']
                    process_img(map_xy, 20, 20)
                    process_img(map_alt, 20, 400)

                    # Update finally the screen with all the images you blitted in the run_trained_batch
                    pygame.display.update() # Updates only the blitted parts of the screen, pygame.display.flip() updates the whole screen
                    pygame.event.get() # Show the last state and then reset
                    sleep(sleep_time)
                    t += 1

                    # Dropping Agent
                    if done==1:

                        print('=== DROPPING AGENT IN CHARGE ===')
                        drop_runner.latest_obs = nav_runner.latest_obs
                        done2 = 0
                        drop_flag = 1
                        # Store
                        #drone_pos = np.where(nav_runner.envs.map_volume['vol'] ==
                        #                     nav_runner.envs.map_volume['feature_value_map']['drone'][
                        #                         nav_runner.envs.altitude]['val'])
                        #mb_drone_pos.append(drone_pos)
                        #mb_obs.append(nav_runner.latest_obs)
                        #mb_flag.append(drop_flag)
                        #mb_heading.append(nav_runner.envs.heading)
                        while done2==0:
                            if step_count >= 50:
                                print("too many steps")
                                done = True
                                done2 = True
                                step_data['stuck'] = True
                                all_data[nav_runner.episode_counter]['stuck'] = True
                                all_data[nav_runner.episode_counter]['drop'].append(step_data)
                                break
                            step_count += 1
                            step_data = {'stuck':False}
                            step_data['volume'] = np.array(nav_runner.envs.map_volume['vol'], copy=True)
                            step_data['heading'] = nav_runner.envs.heading
                            step_data['hiker'] = nav_runner.envs.hiker_position
                            step_data['altitude'] = nav_runner.envs.altitude
                            step_data['drone'] = np.where(step_data['volume'] ==
                                                          nav_runner.envs.map_volume['feature_value_map']['drone'][
                                                              nav_runner.envs.altitude]['val'])

                            # Step
                            obs, action, value, reward, done2, representation, fc, action_probs, grad_V_allo, grad_V_ego, mask_allo, mask_ego = drop_runner.run_trained_batch(drop_flag)
                            # obs, action, value, reward, done2, representation, fc, action_probs, grad_V_allo, grad_V_ego = drop_runner.run_trained_batch(drop_flag)

                            mb_rewards.append(reward)

                            # Store
                            step_data['action'] = action
                            step_data['reward'] = reward
                            step_data['fc'] = fc
                            step_data['action_probs'] = action_probs

                            all_data[nav_runner.episode_counter]['drop'].append(step_data)

                            # Saliencies
                            # grad_V_allo = cmap(grad_V_allo)  # (100,100,4)
                            # grad_V_allo = np.delete(grad_V_allo, 3, 2)  # (100,100,3)
                            # # grad_V = np.stack((grad_V,) * 3, -1)
                            # grad_V_allo = grad_V_allo * 255
                            # grad_V_allo = grad_V_allo.astype(np.uint8)
                            # process_img(grad_V_allo, 400, 20)
                            #
                            # grad_V_ego = cmap(grad_V_ego)  # (100,100,4)
                            # grad_V_ego = np.delete(grad_V_ego, 3, 2)  # (100,100,3)
                            # # grad_pi = np.stack((grad_pi,) * 3, -1)
                            # grad_V_ego = grad_V_ego * 255
                            # grad_V_ego = grad_V_ego.astype(np.uint8)
                            # process_img(grad_V_ego, 400, 400)
                            #
                            # # Masks
                            # masked_map_xy = map_xy
                            # masked_map_xy[mask_allo == 0] = 0
                            # process_img(masked_map_xy, 800, 20)
                            # masked_map_alt = map_alt
                            # masked_map_alt[mask_ego == 0] = 0
                            # process_img(masked_map_alt, 800, 400)

                            screen_mssg_variable("Value    : ", np.round(value, 3), (168, 350))
                            screen_mssg_variable("Reward: ", np.round(reward, 3), (168, 372))
                            screen_mssg_variable("P(drop|s): ", np.round(action_probs[0][-1], 3), (168, 700))

                            pygame.display.update()
                            pygame.event.get()
                            sleep(sleep_time)

                            if action == 15:
                                # The update of the text will be at the same time with the update of state
                                screen_mssg_variable("Package state:", drop_runner.envs.package_state, (20, 350))
                                pygame.display.update()
                                pygame.event.get()  # Update the screen
                                #dictionary[nav_runner.episode_counter]['pack_hiker_dist'] = drop_runner.envs.pack_dist
                                sleep(sleep_time)

                            gameDisplay.fill(DARK_BLUE)
                            map_xy = obs[0]['img']
                            map_alt = obs[0]['nextstepimage']
                            process_img(map_xy, 20, 20)
                            process_img(map_alt, 20, 400)

                            # Update finally the screen with all the images you blitted in the run_trained_batch
                            pygame.display.update()  # Updates only the blitted parts of the screen, pygame.display.flip() updates the whole screen
                            pygame.event.get()  # Show the last state and then reset
                            sleep(sleep_time)
                            t = t +1

                        #dictionary[nav_runner.episode_counter]['observations'] = mb_obs
                        #dictionary[nav_runner.episode_counter]['flag'] = mb_flag
                        #dictionary[nav_runner.episode_counter]['actions'] = mb_actions
                        #dictionary[nav_runner.episode_counter]['rewards'] = mb_rewards
                        #dictionary[nav_runner.episode_counter]['representation'] = mb_representation
                        #dictionary[nav_runner.episode_counter]['fc'] = mb_fc
                        #dictionary[nav_runner.episode_counter]['values'] = mb_values
                        #dictionary[nav_runner.episode_counter]['drone_pos'] = mb_drone_pos
                        #dictionary[nav_runner.episode_counter]['headings'] = mb_heading


                        score = sum(mb_rewards)
                        print(">>>>>>>>>>>>>>>>>>>>>>>>>>> episode %d ended in %d steps. Score %f" % (nav_runner.episode_counter, t, score))
                        nav_runner.episode_counter += 1

                clock.tick(15)

            print("...saving dictionary.")
            with open('./data/all_data' + str(FLAGS.episodes) + '.lst', 'wb') as handle:
                pickle.dump(all_data, handle)

        except KeyboardInterrupt:
            pass

    print("Okay. Work is done")
    #_print(i)
    # if FLAGS.training:
    #     _save_if_training(agent)
    # if not FLAGS.training and FLAGS.save_replay:
    #     #envs.env.save_replay('/Users/constantinos/Documents/StarcraftMAC/MyAgents/')
    #     envs.env.save_replay('./Replays/MyAgents/')

    envs.close()


if __name__ == "__main__":
    main()
    print(stats)
