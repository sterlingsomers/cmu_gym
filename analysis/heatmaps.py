import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import pickle
#from astropy.convolution.kernels import Gaussian2DKernel

from pylab import ogrid

# You dont need so many data for one static case.
pickle_in = open('/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/390_50_static_100.tj','rb')
obs = pickle.load(pickle_in)

''' Dropping locations heatmap '''
counts = np.zeros([10,10])
for i in range(len(obs)):
    #if obs[i]['headings'][0] == 6:
    drop_pos = obs[i]['drone_pos'][-1][-2:]
    drop_pos = np.array(drop_pos).ravel()
    x = drop_pos[0]
    y = drop_pos[1]
    counts[x,y] = counts[x,y] + 1
counts = counts / counts.max()
plot = plt.subplot(111)
# Plot the map
plot.imshow(obs[0]['map_volume']) # 50x50

extent = plot.get_xlim()+ plot.get_ylim()
plot.imshow(counts, interpolation='catrom',cmap='jet', alpha= 0.5, extent=extent)

plt.show()

''' Trajectories heatmap in 2D '''
# TODO: Try to do it in 3D! Here you dont take into account the z-axis
# counts = np.zeros([10,10])
# for i in range(len(obs)):
#     trace = np.array(obs[i]['drone_pos'][:-1]) # Take out the last one as you remain still when you drop
#     for tr in trace:
#         drop_pos = tr.ravel()
#         x = drop_pos[1] # alt has been kept in the first element
#         y = drop_pos[2]
#         counts[x,y] = counts[x,y] + 1
#
# counts = counts / counts.max()
# #plt.imshow(counts, interpolation='catrom') # cmap='jet', 'magma'
#
# plot = plt.subplot(111)
# # Plot the map
# plot.imshow(obs[0]['map_volume']) # 50x50
#
# extent = plot.get_xlim()+ plot.get_ylim()
# plot.imshow(counts, interpolation='catrom',cmap='jet', alpha= 0.5, extent=extent)
#
# plt.show()