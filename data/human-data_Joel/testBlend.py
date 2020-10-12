import pickle
import copy

from pyactup import *

parameter = {'temp':1, 'mismatch':5}

all_chunks = pickle.load(open('all_chunks_JOEL.pkl','rb'))

#some useful dictionaries
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




m = Memory(noise=0.0, decay=None, temperature=parameter['temp'], threshold=-100.0, mismatch=parameter['mismatch'],
                   optimized_learning=False)



for chunk in all_chunks:
    m.learn(**chunk)
    m.advance()


probe = {'hiker_left':0,'hiker_diagonal_left':0,'hiker_center':0,'hiker_diagonal_right':0.0,'hiker_right':0.0,
         'ego_left':0, 'ego_diagonal_left':0,'ego_center':0,'ego_diagonal_right':0,'ego_right':0,
         'altitude':0,'distance_to_hiker':0.0}

blends = []
blend_to_sorts = []
activation_histories = []
for slot in action_slots:
    to_sort = {}
    m.activation_history = []
    blends.append(m.blend(slot, **probe))
    for i,chunk in enumerate(m.activation_history):
        to_sort[i] = chunk['retrieval_probability']
    sorted_x = sorted(to_sort.items(), key=lambda kv: kv[1], reverse=True)
    activation_histories.append(copy.deepcopy(m.activation_history))
    blend_to_sorts.append(sorted_x)


max_blend = max(blends)
all_action_dict = {action_slots[x]:blends[x] for x in list(range(len(blends)))}
all_action_dict = sorted(all_action_dict.items(), key=lambda kv: kv[1], reverse=True)
index_of = blends.index(max_blend)
sorted_x_to_look_at = blend_to_sorts[index_of]
history_to_look_at = activation_histories[index_of]
action = action_slots[index_of]
action_value = combos_to_actions[action]
print(action)

print('debug line')