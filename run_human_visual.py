

import os
import glob
import shutil
import sys
from datetime import datetime
from time import sleep
import pickle
import pygame, time
import numpy as np
import gym_gridworld.envs.gridworld_env as GridWorld
from absl import flags
from pathlib import Path
import time
import random
import math
import itertools
from multiprocessing import Pool
from functools import partial
import copy

from pyactup import *

FLAGS = flags.FLAGS
from gym_gridworld.envs.map_dict import *

flags.DEFINE_integer("episodes", 3, "Number of complete episodes")




#human subject flags
flags.DEFINE_string("participant", 'model_joelData', "The participants name")
flags.DEFINE_string("map", 'canyon', "flatland, canyon, v-river, treeline, small-canyon, flatland")
flags.DEFINE_string('map_data', '', "any map or blank string for data from all maps")
flags.DEFINE_integer("configuration", '0', "0,1, or 2")
flags.DEFINE_bool("specific_map", True, "t/f")
flags.DEFINE_float("temperature", 0.5, '')
flags.DEFINE_float('mismatch', 10.0, '')
flags.DEFINE_float('noise', 0.2, '')


FLAGS(sys.argv)


def restore_last_move(human_data,environment):
    if len(human_data['maps']) == 0:
        return 0
    environment.reset(map=flags.FLAGS.map)
    for step in human_data['actions'][:-1]:
        environment.step(step)
    human_data['maps'] = human_data['maps'][:-1]
    human_data['actions'] = human_data['actions'][:-1]
    human_data['headings'] = human_data['headings'][:-1]
    human_data['altitudes'] = human_data['altitudes'][:-1]
    human_data['drone'] = human_data['drone'][:-1]
    human_data['hiker'] = human_data['hiker'][:-1]
    human_data['reward'] = human_data['reward'][:-1]

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

def get_map_and_configuration(vol):
    flat_vol = np.zeros((vol.shape[1],vol.shape[2]))
    for i in range(vol.shape[0]):
        stuff = np.nonzero(vol[i,:,:])
        volume = vol[i,:,:]
        flat_vol[stuff] = volume[stuff]
    distances = {}
    for map in map_dict:
        map_array = map_dict[map]['map']
        dist = np.linalg.norm(flat_vol-map_array)
        distances[map] = dist
    min_map = min(distances,key=distances.get)
    return min_map

def chunks_stats(allchunks):
    freq = {'left_obs':[],'center_obs':[],'right_obs':[], 'left_act':[],'center_act':[],'right_act':[], 'drop_act':[]}

    for chunk in allchunks:
        for freq_key in freq:
            freq[freq_key].append(0)
        for key in chunk:

            if chunk[key]:
                if 'left' in key and key in observation_slots:
                    freq['left_obs'][-1] = 1
                elif 'center' in key and key in observation_slots:
                    freq['center_obs'][-1] = 1
                elif 'right' in key and key in observation_slots:
                    freq['right_obs'][-1] = 1
                elif 'left' in key and not key in observation_slots:
                    freq['left_act'][-1] = 1
                elif 'center' in key and not key in observation_slots:
                    freq['center_act'][-1] = 1
                elif 'right' in key and not key in observation_slots:
                    freq['right_act'][-1] = 1
                elif 'drop' in key and not key in observation_slots:
                    freq['drop_act'][-1] = 1

    obs = [x for x in freq if '_obs' in x]
    acts = [x for x in freq if '_act' in x]
    pairs = list(itertools.product(obs,acts))
    matches = {}
    for ob,act in pairs:
        num_ob = 0
        num_correspond = 0
        num_not_correspond = 0
        for i in range(len(freq[ob])):
            if freq[ob][i]:
                num_ob += 1
                if freq[act][i]:
                    num_correspond += 1
                else:
                    num_not_correspond += 1

        matches[(ob,act)] = num_correspond / num_ob
    print('here')

def best_files_per_participant(file_list, map='', topN=False, success=True,temporal=False):
    #Given a file name, file all files for that where it is the map.
    participants_to_relevant_files = {}
    best_files = []
    for aFile in file_list:
        file_name_split = aFile.split('-')
        if file_name_split[0] not in participants_to_relevant_files:
            participants_to_relevant_files[file_name_split[0]] = []
        mission = pickle.load(open(aFile, 'rb'))
        try:
            map_config = get_map_and_configuration(mission['maps'][0]['vol'])
            #print('here')
        except IndexError as e:
            print(e)
            continue
        if not map in map_config:
            continue
        if success:
            if not mission['reward'][-1]:
                continue

        participants_to_relevant_files[file_name_split[0]].append(file_name_split[1] + '-' + file_name_split[2])


    #assume latest file is the best file
    if temporal:
        best_participant_file = {}
        for participant in participants_to_relevant_files:
            timestamps = [int(x.split('-')[1][:-3]) for x in participants_to_relevant_files[participant]]
            if not timestamps:
                print(participant, "no examples")
                continue
            max_time = max(timestamps)
            file_name = [x for x in file_list if participant in x and repr(max_time) in x]
            if len(file_name) > 1:
                print('uhoh')
            else:
                best_files.append(file_name[0])
        return best_files

    #just convert what's there to a best_files list
    #and return it
    for part in participants_to_relevant_files:
        for filename in participants_to_relevant_files[part]:
            best_files.append(part + '-' + filename)
    return best_files
    print('here')
    return 0


def create_chunks_from_files(filelist,map='', include_fc=False):
    chunks = []
    filecount = 0
    for file in filelist:
        mission = pickle.load(open(file,'rb'))
        try:
            map_config = get_map_and_configuration(mission['maps'][0]['vol'])
        except IndexError:
            continue
        if not map in map_config:
            continue
        filecount += 1
        for i in range(len(mission['altitudes'])):
            step = {}
            step['volume'] = mission['maps'][i]['vol']
            step['heading'] = mission['headings'][i]
            step['drone'] = mission['drone'][i]
            step['hiker'] = mission['hiker'][i]
            step['altitude'] = mission['altitudes'][i]
            step['action'] = mission['actions'][i]
            hiker_in_volume = np.where(step['volume'] == 36)
            step['volume'][hiker_in_volume] = 0.0
            step['volume'][step['drone']] = 36.0
            egocentric_angle_to_hiker = heading_to_hiker(step['heading'], step['drone'], step['hiker'])
            angle_categories_to_hiker = angle_categories(egocentric_angle_to_hiker)
            egocentric_slice = egocentric_representation(step['drone'], step['heading'], step['volume'])
            chunk = {}
            for key, value in angle_categories_to_hiker.items():
                chunk[key] = value
            altitudes = altitudes_from_egocentric_slice(egocentric_slice)
            altitudes = [x - 1 for x in altitudes]#[x - 1 for x in altitudes]
            alt = step['altitude']
            # chunk.extend(['altitude', ['altitude', int(alt)]])
            chunk['altitude'] = int(alt)
            chunk['ego_left'] = altitudes[0]
            chunk['ego_diagonal_left'] = altitudes[1]
            chunk['ego_center'] = altitudes[2]
            chunk['ego_diagonal_right'] = altitudes[3]
            chunk['ego_right'] = altitudes[4]
            chunk['distance_to_hiker'] = distance_to_hiker(np.array(step['drone']), np.array(step['hiker']))

            for key,value in action_to_category_map.items():
                if len(value) > 1:
                    actr_action = '_'.join(value)
                else:
                    actr_action = value[0]
                if key == step['action']:
                    chunk[actr_action] = 1
                else:
                    chunk[actr_action] = 0

            chunks.append(chunk)
    return chunks


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

def convert_observation_to_chunk(obs,max_dict):

    egocentric_angle_to_hiker = heading_to_hiker(obs['heading'], obs['drone'], obs['hiker'])
    angle_categories_to_hiker = angle_categories(egocentric_angle_to_hiker)
    egocentric_slice = egocentric_representation(obs['drone'], obs['heading'], obs['volume'])
    chunk = {}
    for key, value in angle_categories_to_hiker.items():
        chunk[key] = value  # .extend([key, [key, value]])
    # need the altitudes from the slice
    altitudes = altitudes_from_egocentric_slice(egocentric_slice)
    altitudes = [x - 1 for x in altitudes]
    alt = obs['altitude']
    # chunk.extend(['altitude', ['altitude', int(alt)]])
    chunk['altitude'] = int(alt)
    chunk['ego_left'] = altitudes[0]
    chunk['ego_diagonal_left'] = altitudes[1]
    chunk['ego_center'] = altitudes[2]
    chunk['ego_diagonal_right'] = altitudes[3]
    chunk['ego_right'] = altitudes[4]

    chunk['distance_to_hiker'] = distance_to_hiker(np.array(obs['drone']), np.array(obs['hiker']))
    chunk_alt = chunk.copy()
    for slot in chunk:  # ['altitude', 'distance_to_hiker', 'ego_right','ego_diagonal_right']:#
        # print(slot, chunk, data_by_slot)
        if chunk[slot] == 0 and max(max_dict[slot]) - min(max_dict[slot]) == 0:
            chunk[slot] = 0
        else:
            chunk[slot] = (((chunk[slot] - min(max_dict[slot])) * (1 - 0)) / (
                        max(max_dict[slot]) - min(max_dict[slot]))) + 0  # chunk[slot] / max(data_by_slot[slot])
        # except ZeroDivisionError as e:
        #     print(chunk[slot], min(data_by_slot[slot]), max(data_by_slot))
        #     raise ZeroDivisionError


    return chunk, chunk_alt

def multi_blends(chunk, memory, slots ):
    return [memory.blend(slot,**chunk) for slot in slots]

def vector_similarity(x,y):
    return 0#distance.euclidean(x,y) / 4.5#max(FC_distances)
    # return distance.cosine(x,y)

def distance_similarity(x,y):
    diff = abs(x - y)
    return 1 - diff**1/2

def custom_similarity(x,y):
    if abs(x - y) > 1:
        print('asdf')
        return 1
    return 1 - abs(x - y)

def multi_blends(slot, probe, memory):
    return memory.blend(slot,**probe)

#set the similarity function
set_similarity_function(custom_similarity, *observation_slots[:])
# set_similarity_function(distance_similarity, 'distance_to_hiker')
set_similarity_function(vector_similarity, 'fc')

data_by_slot = {} #needed to normalize observations

def main(data_by_slot):

    human_data_folder = Path('./data/human-data_Joel/')
    parameter = {'temp':flags.FLAGS.temperature,'mismatch':flags.FLAGS.mismatch,'noise':flags.FLAGS.noise}



    score = 0.0
    reward = 0.0
    environment = GridWorld.GridworldEnv()
    config = flags.FLAGS.configuration
    environment.reset(map=flags.FLAGS.map,config=config)
    human_data = {}
    human_data['maps'] = []
    human_data['actions'] = []
    human_data['headings'] = []
    human_data['altitudes'] = []
    human_data['drone'] = []
    human_data['hiker'] = []
    human_data['reward'] = []

    episode_counter = 0

    # pygame.font.get_fonts() # Run it to get a list of all system fonts
    display_w = 1200
    display_h = 720

    BLUE = (128, 128, 255)
    DARK_BLUE = (1, 50, 130)
    RED = (255, 192, 192)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    sleep_time = 0.0

    pygame.init()
    gameDisplay = pygame.display.set_mode((display_w, display_h))
    gameDisplay.fill(DARK_BLUE)
    pygame.display.set_caption('Package Drop')
    clock = pygame.time.Clock()

    def screen_msg_variable(text, variable, area):
        font = pygame.font.SysFont('arial', 16)
        txt = font.render(text + str(variable), True, WHITE)
        gameDisplay.blit(txt, area)

    def process_img(img, x,y):
        # swap the axes else the image will not be the same as the matplotlib one
        img = img.transpose(1,0,2)
        surf = pygame.surfarray.make_surface(img)
        surf = pygame.transform.scale(surf, (300, 300))
        gameDisplay.blit(surf, (x, y))

    def text_objects(text, font):
        textSurface = font.render(text, True, (255,0,0))
        return textSurface, textSurface.get_rect()

    def message_display(text):
        largeText = pygame.font.Font('freesansbold.ttf', 115)
        TextSurf, TextRect = text_objects(text, largeText)
        TextRect.center = ((display_w / 2), (display_h / 2))
        gameDisplay.blit(TextSurf, TextRect)

        pygame.display.update()

        time.sleep(2)


    if not os.path.isdir(human_data_folder):
        print("human data folder incorrect")
        return 0

    current_dir = os.getcwd()
    os.chdir(human_data_folder)
    data_files = glob.glob('*2020*')

    specific_map = ''
    if flags.FLAGS.specific_map:
        specific_map = flags.FLAGS.map


    best_files = best_files_per_participant(data_files,map=specific_map,topN=None,success=True,temporal=False)
    all_chunks = create_chunks_from_files(best_files,map=specific_map)#flags.FLAGS.map)
    os.chdir(current_dir)
    # pickle.dump(best_files,open('best_file_list.pkl','wb'))
    # pickle.dump(all_chunks,open('all_chunks_JOEL.pkl','wb'))

    normalized_data_file = 'human_normalized_all_chunks.lst'
    chunks_path = Path("./")

    if 1:#I need this to happen for now - so I get the max dictionary: #not os.path.isfile(os.path.join(chunks_path, normalized_data_file)):
        data_by_slot = {slot: [] for slot in observation_slots}
        for chunk in all_chunks:
            if not 'hiker_left' in observation_slots:
                print('adfafafafsadfad')
            for slot in observation_slots:
                data_by_slot[slot].append(chunk[slot])
        acount = 0
        for chunk in all_chunks:
            print(acount)
            if 'hiker_left' not in chunk:
                print('hiker left')
            for slot in observation_slots:
                if chunk[slot] == 0 and max(data_by_slot[slot]) - min(data_by_slot[slot]) == 0:
                    chunk[slot] = 0
                    # continue
                # print(chunk, slot, chunk[slot], data_by_slot[slot], 'max', max(data_by_slot[slot]), min(data_by_slot[slot]))# ['altitude', 'distance_to_hiker', 'ego_right','ego_diagonal_right']:#
                else:
                    chunk[slot] = (((chunk[slot] - min(data_by_slot[slot])) * (1 - 0)) / (max(data_by_slot[slot]) - min(data_by_slot[slot]))) + 0#chunk[slot] / max(data_by_slot[slot])
                # except ZeroDivisionError as e:
                #     print(e)
                #     print(chunk[slot], min(data_by_slot[slot]), max(data_by_slot))

            acount += 1

        with open(os.path.join(chunks_path, normalized_data_file), 'wb') as handle:
            pickle.dump(all_chunks, handle)

    else:
        # all_chunks = pickle.load(open(chunks_path + normalized_data_file, 'rb'))
        # FC_distances = pickle.load(open(os.path.join(chunks_path, 'FC_distances.pkl'), 'rb'))
        all_chunks = pickle.load(open(os.path.join(chunks_path, normalized_data_file), 'rb'))


    a = chunks_stats(all_chunks)
    m = Memory(noise=parameter['noise'], decay=None, temperature=parameter['temp'], threshold=-100.0, mismatch=parameter['mismatch'],
               optimized_learning=False)

    # pickle.dump(all_chunks,open('all_chunks_JOEL.pkl','wb'))
    for chunk in all_chunks:
        m.learn(**chunk)
        m.advance()


    dictionary = {}
    running = True
    done = 0
    while episode_counter <= (FLAGS.episodes - 1) and running==True and done ==False:

        print('Episode: ', episode_counter)
        human_data = {}
        human_data['maps'] = []
        human_data['actions'] = []
        human_data['headings'] = []
        human_data['altitudes'] = []
        human_data['drone'] = []
        human_data['hiker'] = []
        human_data['reward'] = []






        new_image = environment.generate_observation()
        obs = {}
        obs['heading'] = environment.heading
        drone_position = np.where(environment.map_volume['vol'] == environment.map_volume['feature_value_map']['drone'][environment.altitude]['val'])
        # drone_position_flat = [int(drone_position[1]), int(drone_position[2])]
        obs['drone'] = drone_position#_flat
        hiker_position = environment.hiker_position
        obs['hiker'] = hiker_position
        obs['volume'] = environment.map_volume['vol']
        obs['altitude'] = environment.altitude
        obs_chunk,chunk_alt = convert_observation_to_chunk(obs,data_by_slot)

        map_xy = new_image['img']#environment.generate_observation()['img']map_image
        map_alt = new_image['nextstepimage']#environment.alt_view
        process_img(map_xy, 20, 20)
        process_img(map_alt, 20, 400)
        pygame.display.update()


        # Quit pygame if the (X) button is pressed on the top left of the window
        # Seems that without this for event quit doesnt show anything!!!
        # Also it seems that the pygame.event.get() is responsible to REALLY updating the screen contents
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         print("human data being written.")
        #         with open('./data/human.tj', 'wb') as handle:
        #             pickle.dump(dictionary, handle)
        #         running = False
        # sleep(sleep_time)



        gameDisplay.fill(DARK_BLUE)

        #
        # drone_pos = np.where(nav_runner.envs.map_volume['vol'] == nav_runner.envs.map_volume['feature_value_map']['drone'][nav_runner.envs.altitude]['val'])


        #get the actions (multiprocessing)
        #do blending multiprocess (no salience needed, no trace needed)
        p = Pool(processes=8)
        multi_p = partial(multi_blends, memory=m, probe=obs_chunk)
        blends = p.map(multi_p, action_slots)
        p.close()
        p.join()

        #do blending single processing
        # blends = []
        # blend_to_sorts = []
        # activation_histories = []
        # for slot in action_slots:
        #     to_sort = {}
        #     m.activation_history = []
        #     blends.append(m.blend(slot, **obs_chunk))
        #     for i,chunk in enumerate(m.activation_history):
        #         to_sort[i] = chunk['retrieval_probability']
        #     sorted_x = sorted(to_sort.items(), key=lambda kv: kv[1], reverse=True)
        #     activation_histories.append(copy.deepcopy(m.activation_history))
        #     blend_to_sorts.append(sorted_x)
            # print('here')

        max_blend = max(blends)
        all_action_dict = {action_slots[x]:blends[x] for x in list(range(len(blends)))}
        all_action_dict = sorted(all_action_dict.items(), key=lambda kv: kv[1], reverse=True)
        index_of = blends.index(max_blend)
        # sorted_x_to_look_at = blend_to_sorts[index_of]
        # history_to_look_at = activation_histories[index_of]
        action = action_slots[index_of]
        action_value = combos_to_actions[action]

        observation, reward, done, info = environment.step(action_value)

        # human_data['maps'].append(environment.map_volume)
        # human_data['headings'].append(environment.heading)
        # human_data['altitudes'].append(environment.altitude)
        # human_data['actions'].append(action)
        # drone_pos = np.where(environment.map_volume['vol'] == environment.map_volume['feature_value_map']['drone'][environment.altitude]['val'])
        #
        # human_data['drone'].append(drone_pos)
        # human_data['hiker'].append(environment.hiker_position)


        obs = environment.generate_observation()
        map_xy = obs['img']
        map_alt = obs['nextstepimage']
        process_img(map_xy, 20, 20)
        process_img(map_alt, 20, 400)

        # Update finally the screen with all the images you blitted in the run_trained_batch
        pygame.display.update() # Updates only the blitted parts of the screen, pygame.display.flip() updates the whole screen
        pygame.event.get() # Show the last state and then reset
        sleep(sleep_time)

        # if t == 70:
        #     break
        if done:
            pygame.event.set_blocked([pygame.KEYDOWN])
            message_display("GAME OVER")
            environment.reset(map=flags.FLAGS.map, config=config)
            episode_counter += 1
            done = False


        clock.tick(15)


    print("human data being written.")
    timestr = timestr = time.strftime("%Y%m%d-%H%M%S")
    print(current_dir)
    with open('./data/model-data/' + flags.FLAGS.participant + '-' + timestr + '.tj', 'wb') as handle:
        pickle.dump(human_data, handle)


if __name__ == '__main__':
    main(data_by_slot)