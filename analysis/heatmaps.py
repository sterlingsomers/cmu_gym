import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pylab import *

pickle_in = open('/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/230_70_static_100.tj','rb')
obs = pickle.load(pickle_in)

counts = np.zeros([10,10])
for i in range(len(obs)):
    #if obs[i]['headings'][0] == 6:
    drop_pos = obs[i]['drone_pos'][-1][-2:]
    drop_pos = np.array(drop_pos).ravel()
    x = drop_pos[0]
    y = drop_pos[1]
    counts[x,y] = counts[x,y] + 1