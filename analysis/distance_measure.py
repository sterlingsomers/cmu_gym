import pickle
import numpy as np

pickle_in = open('/Users/paulsomers/COGLE/gym-gridworld/data/tree_grass_tree_100.tj','rb')
obs1 = pickle.load(pickle_in)
#fc1 = obs1[0]['fc']
pickle_in = open('/Users/paulsomers/COGLE/gym-gridworld/data/tree_grass_trees_100.tj','rb')
obs2 = pickle.load(pickle_in)
#fc2 = obs2[0]['fc']

distances = []
#euclidean distance
#dist = np.linalg.norm(fc1[0]-fc2[0])

poi = (obs1[1]['drone_pos'][0], obs1[1]['headings'][0])

#look for the same in obs2
#matches {index(i): step(x)}
matches = {}
for i in range(len(obs2)):
    positions = obs2[i]['drone_pos']
    for x in range(len(positions)):
        poi2 = (obs2[i]['drone_pos'][x], obs2[i]['headings'][x])
        if poi2 == poi:
            matches[i] = x




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