import numpy as np
import matplotlib.pyplot as plt

import numpy.random
from matplotlib import cm
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

def plot1():
    # setup the figure and axes
    fig = plt.figure(figsize=(3, 3))
    ax1 = fig.add_subplot(111, projection='3d')
    # ax2 = fig.add_subplot(122, projection='3d')

    # fake data
    _x = np.arange(4)
    _y = np.arange(5)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    top = x + y
    bottom = np.zeros_like(top)
    width = depth = 1


    ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
    ax1.set_title('Shaded')

    # ax2.bar3d(x, y, bottom, width, depth, top, shade=False)
    # ax2.set_title('Not Shaded')

    plt.show()

def plot2():


    # To generate some test data
    x = np.random.randn(500)
    y = np.random.randn(500)

    XY = np.stack((x, y), axis=-1)

    def selection(XY, limitXY=[[-2, +2], [-2, +2]]):
        XY_select = []
        for elt in XY:
            if elt[0] > limitXY[0][0] and elt[0] < limitXY[0][1] and elt[1] > limitXY[1][0] and elt[1] < limitXY[1][1]:
                XY_select.append(elt)

        return np.array(XY_select)

    XY_select = selection(XY, limitXY=[[-2, +2], [-2, +2]])

    xAmplitudes = np.array(XY_select)[:, 0]  # your data here
    yAmplitudes = np.array(XY_select)[:, 1]  # your other data here

    fig = plt.figure()  # create a canvas, tell matplotlib it's 3d
    ax = fig.add_subplot(111, projection='3d')

    hist, xedges, yedges = np.histogram2d(x, y, bins=(7, 7), range=[[-2, +2], [-2,
                                                                               +2]])  # you can change your bins, and the range on which to take data
    # hist is a 7X7 matrix, with the populations for each of the subspace parts.
    xpos, ypos = np.meshgrid(xedges[:-1] + xedges[1:], yedges[:-1] + yedges[1:]) - (xedges[1] - xedges[0])

    xpos = xpos.flatten() * 1. / 2
    ypos = ypos.flatten() * 1. / 2
    zpos = np.zeros_like(xpos)

    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    dz = hist.flatten()

    cmap = cm.get_cmap('jet')  # Get desired colormap - you can change this!
    max_height = np.max(dz)  # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k - min_height) / max_height) for k in dz]

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    plt.title("X vs. Y Amplitudes for ____ Data")
    plt.xlabel("My X data source")
    plt.ylabel("My Y data source")
    plt.savefig("Your_title_goes_here")
    plt.show()

plot1()