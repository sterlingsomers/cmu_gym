import pickle

import fnmatch
import os

import operator

import numpy as np
from scipy.stats import entropy
from scipy.spatial import distance
datas = []
#the following loop searches the whole directory and adds each file to datas
os.chdir('/Users/paulsomers/COGLE/gym-gridworld/data/fc-only')

for file in os.listdir('.'):
    if fnmatch.fnmatch(file, '*.tj'):
        datas.append(pickle.load(open(file,'rb')))

ordered_actions = ['left_down', 'diagonal_left_down','center_down','diagonal_right_down','right_down',
                   'left_level', 'diagonal_left_level', 'center_level', 'diagonal_right_level', 'right_level',
                   'left_up','diagonal_left_up','center_up','diagonal_right_up','right_up',
                   'drop']
# datas.append(pickle.load(open('eBEHAVE_FC_MP3_TESTMAP146-456_D18-11_HeadAlt1-1_H11-8_1.tj','rb')))

kls = []
JSs = []
#multiple runs version
# for data in datas:
#     for map in data:
#         for mission in data[map]:
#             if type(mission) == str:
#                 continue
#             for dist_net,dist_actr in zip(data[map][mission]['action_probs'],data[map][mission]['actr_probs']):
#                 dist_actr = dist_actr.values()
#                 dist_actr[dist_actr == 0] = 0.00000001
#                 dist_net[dist_net == 0] = 0.00000001
#                 single_kl = entropy(dist_net[0],dist_actr[0])
#                 single_js = distance.jensenshannon(dist_actr[0],dist_net[0])
#
#                 kls.append(single_kl)
#                 JSs.append(single_js)
combos_to_actions = {'left_down':0,'diagonal_left_down':1,'center_down':2,
                     'diagonal_right_down':3,'right_down':4,
                     'left_level':5,'diagonal_left_level':6,'center_level':7,
                     'diagonal_right_level':8,'right_level':9,
                     'left_up':10,'diagonal_left_up':11,'center_up':12,
                     'diagonal_right_up':13,'right_up':14,'drop':15}
all_correct = []
for data in datas:
    for mission in data:
        actr = []
        actr_actions =  []#max(action_choice.items(), key=operator.itemgetter(1))[0]
        all_actr = data[mission]['actr_probs'] #a list of dictionaries
        for distribution in all_actr:
            actr.append(list(distribution.values()))
            actr_actions.append(combos_to_actions[max(distribution.items(), key=operator.itemgetter(1))[0]])

        net = data[mission]['action_probs']
        for ACT,NET in zip(actr,net):
            single_kl = entropy(ACT,NET[0])
            single_js = distance.jensenshannon(ACT,NET[0])

            kls.append(single_kl)
            JSs.append(single_js)
            print('ok')

        correct = []
        for one,two in zip(actr_actions,data[mission]['actions']):
            correct.append(int(one == two))
        all_correct.extend(correct)



        print('test')



k = len(kls)
avg_divergence = (1/(k*(k-1))) * sum(kls)

avg_kl = sum(kls)/len(kls)
avg_JSs = sum(JSs)/len(JSs)

score = sum(all_correct)/len(all_correct)
print("done")