

import os
import glob
import shutil
import sys
from datetime import datetime
from time import sleep
import pickle
import pygame, time
import numpy as np
import gym_gridworld.envs.gridworld_env as GridWorld
from absl import flags
from pathlib import Path
import time
import random
import math
import itertools
from multiprocessing import Pool
from functools import partial
import copy

from pyactup import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mping

FLAGS = flags.FLAGS
from gym_gridworld.envs.map_dict import *

flags.DEFINE_integer("episodes", 1, "Number of complete episodes")




#human subject flags
flags.DEFINE_string("participant", 'model_joelData', "The participants name")
flags.DEFINE_string("map", 'v-river', "flatland, canyon, v-river, treeline, small-canyon, flatland")
flags.DEFINE_string('map_data', '', "any map or blank string for data from all maps")
flags.DEFINE_integer("configuration", '2', "0,1, or 2")
flags.DEFINE_bool("specific_map", True, "t/f")
flags.DEFINE_float("temperature", 1.0, '')
flags.DEFINE_float('mismatch', 5.0, '')

human_data_folder = Path('./')
os.chdir(human_data_folder)
data_files = glob.glob('*2020*')

used_files = pickle.load(open('best_file_list.pkl','rb'))

these_files = [x for x in data_files if x in used_files]
these_data = [pickle.load(open(x,'rb')) for x in these_files]
for episode in these_data:
    for step in episode['maps']:
        plt.imshow(step['img'])
        plt.show()
        print('here')

print('here')

