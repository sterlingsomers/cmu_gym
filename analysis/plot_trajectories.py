import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.misc import imresize
from pylab import ogrid

pickle_in = open('/Users/paulsomers/COGLE/gym-gridworld/data/tree_grass_tree_100_static_heading.tj','rb')
obs = pickle.load(pickle_in)

fig = plt.figure(1)
ax = fig.gca(projection='3d')
ax.set_xlim(-0.5,9.5)
ax.set_ylim(-0.5,9.5)
ax.set_zlim(0,3)
ax.set_xlabel('lat')
ax.set_ylabel('lon')
ax.set_zlabel('alt')


img = obs[0]['map_volume']
img = imresize(img, 0.2, interp='nearest')
img = img/255.0
X1, Y1 = ogrid[-0.5:img.shape[0], -0.5:img.shape[1]]

ax.plot_surface(X1, Y1, np.atleast_2d(0), rstride=1, cstride=1, facecolors=img, shade=False) # If you dont put shade it will be darker

# Plot all trajectories from a pickle file
# for i in range(len(obs)):
#     if obs[i]['headings'][0] == 6:
#         print(obs[i]['headings'][0])
#         trace = obs[i]['drone_pos']  # 0 and 8 are almost the same, 4 is weird, doesnt go down
#         trace_zxy = np.concatenate(trace, axis=1)
#         z = trace_zxy[0]
#         x = trace_zxy[1]
#         y = trace_zxy[2]
#
#         zf = trace[-1:][0][0][0]  # with -1 we select the last position
#         xf = trace[-1:][0][1][0]
#         yf = trace[-1:][0][2][0]
#
#         zi = trace[0][0][0]  # with -1 we select the last position
#         xi = trace[0][1][0]
#         yi = trace[0][2][0]
#
#         # plot trajectory
#         ax.plot(x, y, z, linewidth=2.0)  # , label='trajectory')
#         # plot indicator lines
#         ax.plot([xf, xf], [yf, yf], [0, zf], '--', alpha=0.8, linewidth=1.0,
#                 color='deeppink')  # You connect two points [x1,x2],[y1,y2],[z1,z2]
#         ax.plot([xi, xi], [yi, yi], [0, zi], '--', alpha=0.8, linewidth=1.0, color='fuchsia')
#
#         # plot map at the bottom
#
#
#         c = np.linspace(1, len(trace), len(trace))
#         c = c / 15
#         c = c * 145.0
#         # ax.scatter(x, y, z, c=z,label=' temporal trajectory')
#         # ax.scatter(x, y, z, c=c,label=' temporal trajectory')
# ax.legend()
# plt.show()



#print(obs[1]['headings'][0])
trace=obs[2]['drone_pos'] # 0 and 8 are almost the same, 4 is weird, doesnt go down
trace_zxy = np.concatenate( trace, axis=1 )
z = trace_zxy[0]
x = trace_zxy[1]
y = trace_zxy[2]


zf = trace[-1:][0][0][0] # with -1 we select the last position
xf = trace[-1:][0][1][0]
yf = trace[-1:][0][2][0]

zi = trace[0][0][0]# with -1 we select the last position
xi = trace[0][1][0]
yi = trace[0][2][0]

# plot trajectory
ax.plot(x, y, z, linewidth=2.0)#, label='trajectory')
# plot indicator lines
ax.plot([xf,xf],[yf,yf],[0,zf],'--',alpha=0.8, linewidth=1.0, color='deeppink') # You connect two points [x1,x2],[y1,y2],[z1,z2]
ax.plot([xi,xi],[yi,yi],[0,zi],'--',alpha=0.8, linewidth=1.0, color='fuchsia')

c=np.linspace(1,len(trace),len(trace))
c = c/15
c = c* 145.0
#ax.scatter(x, y, z, c=z,label=' temporal trajectory')
#ax.scatter(x, y, z, c=c,label=' temporal trajectory')
ax.legend()
plt.show()