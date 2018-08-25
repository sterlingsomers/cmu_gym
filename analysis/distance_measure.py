import pickle
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

pickle_in = open('/Users/paulsomers/COGLE/gym-gridworld/data/tree_grass_grass_100_static_heading.tj','rb')
obs1 = pickle.load(pickle_in)
#fc1 = obs1[0]['fc']
pickle_in = open('/Users/paulsomers/COGLE/gym-gridworld/data/tree_grass_tree_100_static_heading.tj','rb')
obs2 = pickle.load(pickle_in)
#fc2 = obs2[0]['fc']





def find_divergent_step(case1,case2,case1index=0,case2index=0):
    '''returns the index of the step where the two cases@index diverge.
    if case1 has 100 runs, case1index indicates which run'''
    case1_sequence = case1[case1index]['drone_pos']
    case2_sequence = case2[case2index]['drone_pos']
    if len(case1_sequence) > len(case2_sequence):
        length = len(case1_sequence)
    else:
        length = len(case2_sequence)
    if case1_sequence == case2_sequence:
        return None
    for i in range(length):
        if case1_sequence[i] != case2_sequence[i]:
            return i
    return min(len(case1_sequence),len(case2_sequence))



def find_matching_cases(case1,case2,position,heading):
    '''This will search through the two cases to find matches in heading and position
    Cases are an obs list'''

    #r_dict will return {case1: [{index of run in collection: index of step in run},...],
    #                   case2: ...}
    #if either case1 or case2 are empty, then don't use it for analysis
    #position forat: (array([3]), array([5]), array([5]))
    r_dict = {'case1':[],'case2':[]}
    for i in range(len(case1)):
        positions = case1[i]['drone_pos']
        for x in range(len(positions)):
            if case1[i]['drone_pos'][x] == position and case1[i]['headings'][x] == heading:
                r_dict['case1'].append({i:x})
    for i in range(len(case2)):
        positions = case2[i]['drone_pos']
        for x in range(len(positions)):
            if case2[i]['drone_pos'][x] == position and case2[i]['headings'][x] == heading:
                r_dict['case2'].append({i:x})

    return r_dict

matching_cases = find_matching_cases(obs1, obs2, (np.array([3]), np.array([5]), np.array([5])),5)
index1 = 0
index2 = 0
divergent_step = find_divergent_step(obs1,obs2,index1,index2)
#now that I know where they diverge, what were they doing before that?


fig = plt.figure()
fig.add_subplot(2,2,1)
plt.imshow(obs1[index1]['observations'][divergent_step - 1]['rgb_screen'].reshape(50,50,3)) # 50x50
fig.add_subplot(2,2,2)
plt.imshow(obs1[index1]['observations'][divergent_step - 1]['alt_view'].reshape(50,50,3))
fig.add_subplot(2,2,3)
plt.imshow(obs2[index2]['observations'][divergent_step - 1]['rgb_screen'].reshape(50,50,3))
fig.add_subplot(2,2,4)
plt.imshow(obs2[index2]['observations'][divergent_step - 1]['alt_view'].reshape(50,50,3))
plt.show()

#measure distances at divergent - 1
fc1 = obs1[index1]['fc']
fc2 = obs2[index2]['fc']

print(np.linalg.norm(fc1[divergent_step-1]-fc2[divergent_step-1]))

# distances = []
# #euclidean distance
# #dist = np.linalg.norm(fc1[0]-fc2[0])
#
# poi = (obs1[1]['drone_pos'][0], obs1[1]['headings'][0])
#
# #look for the same in obs2
# #matches {index(i): step(x)}
# matches = {}
# for i in range(len(obs2)):
#     positions = obs2[i]['drone_pos']
#     print(positions)
#     break
#     for x in range(len(positions)):
#         poi2 = (obs2[i]['drone_pos'][x], obs2[i]['headings'][x])
#         if poi2 == poi:
#             matches[i] = x




#each i is a run, and each of those has a step.
#start with comparing the first step of each run
# for i in range(100):
#     fcs1 = np.array(obs1[i]['fc'][0])
#     fcs2 = np.array(obs2[i]['fc'][0])
#     print('')
#     #fc2 = np.array(obs2[i]['fc'])
#
#     distances.append(np.linalg.norm(fcs1-fcs2))
print("done")