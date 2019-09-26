import numpy as np
import matplotlib
# matplotlib.use("Agg")

import matplotlib.backends.backend_agg as agg
import pylab

x = [u'Total', u'Goal', u'Fire']
y = [0.9, 1.0, -0.5]
c = ['yellow','green','red']
width = 0.75 # the width of the bars
ind = np.arange(len(y))  # the x locations for the groups

fig = pylab.figure(figsize=[4, 4], # Inches
                   dpi=100,        # 100 dots per inch, so the resulting buffer is 400x400 pixels
                   )
ax = fig.gca()
ax.set_facecolor('xkcd:black')
fig.patch.set_facecolor('xkcd:black')
# ax.plot([1, 2, 4])
ax.barh(ind, y, width, color=c)
ax.set_yticks(ind)#+width/8)
ax.set_yticklabels(x, minor=False, color='white')
for i, v in enumerate(y):
    ax.text(v, i, " "+str(v), color='cyan', va='center', fontweight='bold')

canvas = agg.FigureCanvasAgg(fig)
canvas.draw()
renderer = canvas.get_renderer()
raw_data = renderer.tostring_rgb()

import pygame
from pygame.locals import *

pygame.init()

window = pygame.display.set_mode((400, 400), DOUBLEBUF)
screen = pygame.display.get_surface()

size = canvas.get_width_height()

surf = pygame.image.fromstring(raw_data, size, "RGB")
screen.blit(surf, (0,0))
pygame.display.flip()