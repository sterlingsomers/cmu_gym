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

def get_map_and_configuration(vol):
    flat_vol = np.zeros((vol.shape[1],vol.shape[2]))
    for i in range(vol.shape[0]):
        stuff = np.nonzero(vol[i,:,:])
        volume = vol[i,:,:]
        flat_vol[stuff] = volume[stuff]
    distances = {}
    map_array = map_dict[map]['map']
    dist = np.linalg.norm(flat_vol-map_array)
    distances[map] = dist
    min_map = min(distances,key=distances.get)

    #configuration
    hiker_location = np.where(vol == 50)
    drone_location = np.where(vol == 36)

    hiker_indexes = [i for i, x in enumerate(map_dict[min_map]) if ]

    return min_map

current_directory = os.getcwd()
human_data_folder = Path('./')
os.chdir(human_data_folder)
data_files = glob.glob('*2020*')
missions = [pickle.load(open(x,'rb')) for x in data_files]

#oragnize bby map {map : { configuration : {'data':[data points],'base-map':...}}
map_to_data = {}
a = get_map_and_configuration(missions[0]['maps'][0]['vol'])

print('here')

