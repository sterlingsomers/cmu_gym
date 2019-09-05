import os
# import pyactup
from pyactup import *
import pickle
import random

import timeit

random.seed(42)

chunk_file_name = 'chunk_dict_ego_entropy.pkl'
# #chunk_path = os.path.join(model_path,'data')
chunk_path = '/Users/paulsomers/COGLE/gym-gridworld/'

allchunks = pickle.load(open(os.path.join(chunk_path,chunk_file_name),'rb'))


action_list = ['left_down','diagonal_left_down','center_down','diagonal_right_down','right_down',
                   'left_level','diagonal_left_level','center_level','diagonal_right_level','right_level',
                   'left_up','diagonal_left_up','center_up','diagonal_right_up','right_up', 'drop']

slot_names = ['hiker_left', 'hiker_diagonal_left', 'hiker_center', 'hiker_diagonal_right', 'hiker_right',
              'ego_left', 'ego_diagonal_left', 'ego_center', 'ego_diagonal_right', 'ego_right',
              'altitude']

combos_to_actions = {'left_down':0,'diagonal_left_down':1,'center_down':2,
                     'diagonal_right_down':3,'right_down':4,
                     'left_level':5,'diagonal_left_level':6,'center_level':7,
                     'diagonal_right_level':8,'right_level':9,
                     'left_up':10,'diagonal_left_up':11,'center_up':12,
                     'diagonal_right_up':13,'right_up':14,'drop':15}



def strip_actions(chunk):
    '''strips actions away, leaving only observations'''
    for action in action_list:
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
            for possible_action in action_list:
                achunk[possible_action] = int(possible_action == action)

            # print('test')
            all_chunks.append(achunk)
    return all_chunks

def custom_similarity(x,y):
    # import pdb; pdb.set_trace()
    # print(x,y)
    return abs(x - y)

#set the similarity function
set_similarity_function(custom_similarity, *slot_names)

m = Memory(noise=0.0, decay=0.0, temperature=0.2, threshold=-100.0, mismatch=1.0, optimized_learning=True)

memory_chunks = unpack_chunks(allchunks)
random.shuffle(memory_chunks)

for chunk in memory_chunks[:15000]:
    m.learn(**chunk)

m.advance()
# for i in range(10):
#     observation = strip_actions(random.choice(memory_chunks))
#     #action_blends = [m.blend(x, **observation) for x in action_list]
#     timeit.timeit(stmt='action_blends = [m.blend(x, **observation) for x in action_list]')
#     print(action_blends)

print(timeit.timeit('observation = strip_actions(random.choice(memory_chunks));'
              'action_blends = [m.blend(x, **observation) for x in action_list]', globals=globals(), number=5))