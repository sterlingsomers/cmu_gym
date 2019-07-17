import gym
import pickle
import sys
import os
import time
import copy
import math
import itertools
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from scipy.spatial import distance
from PIL import Image as Image
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import threading
import random
# import pygame
from scipy.misc import imresize

from gym_gridworld.envs import create_np_map as CNP

#from mavsim_server import MavsimHandler

# define colors
# 0: black; 1 : gray; 2 : blue; 3 : green; 4 : red
COLORS = {0: [0.0, 0.0, 0.0], 1: [0.5, 0.5, 0.5], \
          2: [0.0, 0.0, 1.0], 3: [0.0, 1.0, 0.0], \
          4: [1.0, 0.0, 0.0], 6: [1.0, 0.0, 1.0], \
          7: [1.0, 1.0, 0.0]}


class GridworldEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    num_envs = 1

    def __init__(self, map_x=0, map_y=0, local_x=0, local_y=0, heading=1, altitude=2, hiker_x=5, hiker_y=5, width=20,
                 height=20, verbose=False):

        self.verbose = verbose # to show the environment or not
        self.dropping = True # This is for the reset to select the proper starting locations for hiker and drone
        self.restart_once_done = True  # restart or not once done
        self.drop = False
        self.countdrop = 0
        self.no_action_flag = False
        self.maps =[(1,4)]#[(321, 337)]#[(1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7)]
            # [(265,308),(20,94),(146,456),(149,341),(164,90),(167,174),
            #             (224,153),(241,163),(260,241),(265,311),(291,231),
            #             (308,110),(334,203),(360,112),(385,291),(330,352),(321,337)]#[(400,35), (350,90), (430,110),(390,50), (230,70)] #[(86, 266)] (70,50) # For testing, 70,50 there is no where to drop in the whole map
        #[(149, 341)]#[(149, 341),(241,163), (260,241),(291,231),(308,110),(330,352)]
        self.mapw = 20
        self.maph = 20
        self.dist_old = 1000
        self.drop_package_grid_size_by_alt = {1: 3, 2: 5, 3: 7}
        self.factor = 5
        self.reward = 0
        self.action_space = spaces.Discrete(16)
        self.actions = list(range(self.action_space.n))
        self.obs_shape = [100,100,3]
        self.observation_space = spaces.Box(low=0, high=255, shape=self.obs_shape,dtype=np.uint8)
        self.real_actions = False
        self.crash = 0
        self.package_dropped = 0
        self.package_position = ()
        self.altitude = random.randint(1,3)
        # self._max_episode_steps = 10 # Max timesteps

        self.masks = {
            'hiker': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 18, 21, 22],
            1: [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 18, 21, 22],
            2: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 18, 21, 22, 19, 24, 30, 32],
            3: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 18, 21, 22, 19, 24, 30, 32, 13, 17, 19, 25, 28]
        }

        if self.real_actions:
            self.mavsimhandler = MavsimHandler()
            stateThread = threading.Thread(target=self.mavsimhandler.read_state)
            stateThread.start()

        self.image_layers = {}

        # 5x5 plane descriptions
        self.planes = {}
        self.planes[1] = [[(0, 2), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4)], np.zeros((5, 5, 3))]
        self.planes[2] = [[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (0, 4), (1, 3), (2, 3), (1, 2)], np.zeros((5, 5, 3))]
        self.planes[3] = [[(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (1, 3), (2, 3), (3, 3), (2, 4)], np.zeros((5, 5, 3))]
        self.planes[4] = [[(0, 4), (1, 3), (2, 3), (3, 3), (4, 4), (2, 2), (3, 2), (3, 1), (4, 0)], np.zeros((5, 5, 3))]
        self.planes[5] = [[(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (4, 2)], np.zeros((5, 5, 3))]
        self.planes[6] = [[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (2, 1), (3, 1), (3, 2), (4, 0)], np.zeros((5, 5, 3))]
        self.planes[7] = [[(2, 0), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)], np.zeros((5, 5, 3))]
        self.planes[8] = [[(0, 0), (4, 0), (1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (1, 3), (0, 4)], np.zeros((5, 5, 3))]

        self.hikers = {}
        self.hikers[0] = [[(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4)],
                          np.zeros((5, 5, 3))]
        self.hiker_image = np.zeros((5, 5, 3))
        # self.hiker_image[:,:,:] = self.map_volume['feature_value_map']['hiker']['color']

        # package description
        self.package = {}
        self.package['OK'] = [[(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)],
                              np.zeros((5, 5, 3))]
        self.package['DAMAGED'] = [[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (4, 0), (3, 1), (1, 3), (0, 4)],
                                   np.zeros((5, 5, 3))]

        self.drop_probabilities = {"damage_probability": {0: 0.00, 1: 0.01, 2: 0.40, 3: 0.80},
                                   "stuck_probability": {"pine trees": 0.50, "pine tree": 0.25, "cabin": 0.50,
                                                         "flight tower": 0.15, "firewatch tower": 0.20},
                                   "sunk_probability": {"water": 0.50}
                                   }
        self.drop_rewards = {"OK": 1,#10, # try all positive rewards
                             # "OK_STUCK": 5,
                             # "OK_SUNK": 5,
                             "DAMAGED": -1,# 0#-10,
                             # "DAMAGED_STUCK": -15,
                             # "DAMAGED_SUNK": -15,
                             # "CRASHED": -30
                             }
        # self.alt_rewards = {0:-1, 1:1, 2:-0.5, 3:-0.8} # This is bad!
        self.alt_rewards = {0: 0, 1: 1, 2: 0.5, 3: 0.08}
        # self.alt_rewards = {0: 0, 1: 0, 2: 0, 3: 0}


        self.possible_actions_map = {
            1: [[0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1]],
            2: [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1]],
            3: [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0]],
            4: [[-1, 1], [0, 1], [1, 1], [1, 0], [1, -1]],
            5: [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1]],
            6: [[1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]],
            7: [[1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0]],
            8: [[1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]

        }

        self.actionvalue_heading_action = {
            0: {1: 'self.take_action(delta_alt=-1,delta_x=-1,delta_y=0,new_heading=7)',
                2: 'self.take_action(delta_alt=-1,delta_x=-1,delta_y=-1,new_heading=8)',
                3: 'self.take_action(delta_alt=-1,delta_x=0,delta_y=-1,new_heading=1)',
                4: 'self.take_action(delta_alt=-1,delta_x=1,delta_y=-1,new_heading=2)',
                5: 'self.take_action(delta_alt=-1,delta_x=1,delta_y=0,new_heading=3)',
                6: 'self.take_action(delta_alt=-1,delta_x=1,delta_y=1,new_heading=4)',
                7: 'self.take_action(delta_alt=-1,delta_x=0,delta_y=1,new_heading=5)',
                8: 'self.take_action(delta_alt=-1,delta_x=-1,delta_y=1,new_heading=6)'},
            1: {1: 'self.take_action(delta_alt=-1,delta_x=-1,delta_y=-1,new_heading=8)',
                2: 'self.take_action(delta_alt=-1,delta_x=0,delta_y=-1,new_heading=1)',
                3: 'self.take_action(delta_alt=-1,delta_x=1,delta_y=-1,new_heading=2)',
                4: 'self.take_action(delta_alt=-1,delta_x=1,delta_y=0,new_heading=3)',
                5: 'self.take_action(delta_alt=-1,delta_x=1,delta_y=1,new_heading=4)',
                6: 'self.take_action(delta_alt=-1,delta_x=0,delta_y=1,new_heading=5)',
                7: 'self.take_action(delta_alt=-1,delta_x=-1,delta_y=1,new_heading=6)',
                8: 'self.take_action(delta_alt=-1,delta_x=-1,delta_y=0,new_heading=7)'},
            2: {1: 'self.take_action(delta_alt=-1,delta_x=0,delta_y=-1,new_heading=1)',
                2: 'self.take_action(delta_alt=-1,delta_x=1,delta_y=-1,new_heading=2)',
                3: 'self.take_action(delta_alt=-1,delta_x=1,delta_y=0,new_heading=3)',
                4: 'self.take_action(delta_alt=-1,delta_x=1,delta_y=1,new_heading=4)',
                5: 'self.take_action(delta_alt=-1,delta_x=0,delta_y=1,new_heading=5)',
                6: 'self.take_action(delta_alt=-1,delta_x=-1,delta_y=1,new_heading=6)',
                7: 'self.take_action(delta_alt=-1,delta_x=-1,delta_y=0,new_heading=7)',
                8: 'self.take_action(delta_alt=-1,delta_x=-1,delta_y=-1,new_heading=8)'},
            3: {1: 'self.take_action(delta_alt=-1,delta_x=1,delta_y=-1,new_heading=2)',
                2: 'self.take_action(delta_alt=-1,delta_x=1,delta_y=0,new_heading=3)',
                3: 'self.take_action(delta_alt=-1,delta_x=1,delta_y=1,new_heading=4)',
                4: 'self.take_action(delta_alt=-1,delta_x=0,delta_y=1,new_heading=5)',
                5: 'self.take_action(delta_alt=-1,delta_x=-1,delta_y=1,new_heading=6)',
                6: 'self.take_action(delta_alt=-1,delta_x=-1,delta_y=0,new_heading=7)',
                7: 'self.take_action(delta_alt=-1,delta_x=-1,delta_y=-1,new_heading=8)',
                8: 'self.take_action(delta_alt=-1,delta_x=-0,delta_y=-1,new_heading=1)'},
            4: {1: 'self.take_action(delta_alt=-1,delta_x=1,delta_y=0,new_heading=3)',
                2: 'self.take_action(delta_alt=-1,delta_x=1,delta_y=1,new_heading=4)',
                3: 'self.take_action(delta_alt=-1,delta_x=0,delta_y=1,new_heading=5)',
                4: 'self.take_action(delta_alt=-1,delta_x=-1,delta_y=1,new_heading=6)',
                5: 'self.take_action(delta_alt=-1,delta_x=-1,delta_y=0,new_heading=7)',
                6: 'self.take_action(delta_alt=-1,delta_x=-1,delta_y=-1,new_heading=8)',
                7: 'self.take_action(delta_alt=-1,delta_x=0,delta_y=-1,new_heading=1)',
                8: 'self.take_action(delta_alt=-1,delta_x=1,delta_y=-1,new_heading=2)'},
            5: {1: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=0, new_heading=7)',
                2: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=-1, new_heading=8)',
                3: 'self.take_action(delta_alt=0, delta_x=0, delta_y=-1, new_heading=1)',
                4: 'self.take_action(delta_alt=0, delta_x=1, delta_y=-1, new_heading=2)',
                5: 'self.take_action(delta_alt=0, delta_x=1, delta_y=0, new_heading=3)',
                6: 'self.take_action(delta_alt=0, delta_x=1, delta_y=1, new_heading=4)',
                7: 'self.take_action(delta_alt=0, delta_x=0, delta_y=1, new_heading=5)',
                8: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=1, new_heading=6)'},
            6: {1: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=-1, new_heading=8)',
                2: 'self.take_action(delta_alt=0, delta_x=0, delta_y=-1, new_heading=1)',
                3: 'self.take_action(delta_alt=0, delta_x=1, delta_y=-1, new_heading=2)',
                4: 'self.take_action(delta_alt=0, delta_x=1, delta_y=0, new_heading=3)',
                5: 'self.take_action(delta_alt=0, delta_x=1, delta_y=1, new_heading=4)',
                6: 'self.take_action(delta_alt=0, delta_x=0, delta_y=1, new_heading=5)',
                7: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=1, new_heading=6)',
                8: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=0, new_heading=7)'},
            7: {1: 'self.take_action(delta_alt=0, delta_x=0, delta_y=-1, new_heading=1)',
                2: 'self.take_action(delta_alt=0, delta_x=1, delta_y=-1, new_heading=2)',
                3: 'self.take_action(delta_alt=0, delta_x=1, delta_y=0, new_heading=3)',
                4: 'self.take_action(delta_alt=0, delta_x=1, delta_y=1, new_heading=4)',
                5: 'self.take_action(delta_alt=0, delta_x=0, delta_y=1, new_heading=5)',
                6: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=1, new_heading=6)',
                7: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=0, new_heading=7)',
                8: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=-1, new_heading=8)'},
            8: {1: 'self.take_action(delta_alt=0, delta_x=1, delta_y=-1, new_heading=2)',
                2: 'self.take_action(delta_alt=0, delta_x=1, delta_y=0, new_heading=3)',
                3: 'self.take_action(delta_alt=0, delta_x=1, delta_y=1, new_heading=4)',
                4: 'self.take_action(delta_alt=0, delta_x=0, delta_y=1, new_heading=5)',
                5: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=1, new_heading=6)',
                6: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=0, new_heading=7)',
                7: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=-1, new_heading=8)',
                8: 'self.take_action(delta_alt=0, delta_x=-0, delta_y=-1, new_heading=1)'},
            9: {1: 'self.take_action(delta_alt=0, delta_x=1, delta_y=0, new_heading=3)',
                2: 'self.take_action(delta_alt=0, delta_x=1, delta_y=1, new_heading=4)',
                3: 'self.take_action(delta_alt=0, delta_x=0, delta_y=1, new_heading=5)',
                4: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=1, new_heading=6)',
                5: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=0, new_heading=7)',
                6: 'self.take_action(delta_alt=0, delta_x=-1, delta_y=-1, new_heading=8)',
                7: 'self.take_action(delta_alt=0, delta_x=0, delta_y=-1, new_heading=1)',
                8: 'self.take_action(delta_alt=0, delta_x=1, delta_y=-1, new_heading=2)'},
            10: {1: 'self.take_action(delta_alt=1, delta_x=-1, delta_y=0, new_heading=7)',
                 2: 'self.take_action(delta_alt=1, delta_x=-1, delta_y=-1, new_heading=8)',
                 3: 'self.take_action(delta_alt=1, delta_x=0, delta_y=-1, new_heading=1)',
                 4: 'self.take_action(delta_alt=1, delta_x=1, delta_y=-1, new_heading=2)',
                 5: 'self.take_action(delta_alt=1, delta_x=1, delta_y=0, new_heading=3)',
                 6: 'self.take_action(delta_alt=1, delta_x=1, delta_y=1, new_heading=4)',
                 7: 'self.take_action(delta_alt=1, delta_x=0, delta_y=1, new_heading=5)',
                 8: 'self.take_action(delta_alt=1, delta_x=-1, delta_y=1, new_heading=6)'},
            11: {1: 'self.take_action(delta_alt=1, delta_x=-1, delta_y=-1, new_heading=8)',
                 2: 'self.take_action(delta_alt=1, delta_x=0, delta_y=-1, new_heading=1)',
                 3: 'self.take_action(delta_alt=1, delta_x=1, delta_y=-1, new_heading=2)',
                 4: 'self.take_action(delta_alt=1, delta_x=1, delta_y=0, new_heading=3)',
                 5: 'self.take_action(delta_alt=1, delta_x=1, delta_y=1, new_heading=4)',
                 6: 'self.take_action(delta_alt=1, delta_x=0, delta_y=1, new_heading=5)',
                 7: 'self.take_action(delta_alt=1, delta_x=-1, delta_y=1, new_heading=6)',
                 8: 'self.take_action(delta_alt=1, delta_x=-1, delta_y=0, new_heading=7)'},
            12: {1: 'self.take_action(delta_alt=1, delta_x=0, delta_y=-1, new_heading=1)',
                 2: 'self.take_action(delta_alt=1, delta_x=1, delta_y=-1, new_heading=2)',
                 3: 'self.take_action(delta_alt=1, delta_x=1, delta_y=0, new_heading=3)',
                 4: 'self.take_action(delta_alt=1, delta_x=1, delta_y=1, new_heading=4)',
                 5: 'self.take_action(delta_alt=1, delta_x=0, delta_y=1, new_heading=5)',
                 6: 'self.take_action(delta_alt=1, delta_x=-1, delta_y=1, new_heading=6)',
                 7: 'self.take_action(delta_alt=1, delta_x=-1, delta_y=0, new_heading=7)',
                 8: 'self.take_action(delta_alt=1, delta_x=-1, delta_y=-1, new_heading=8)'},
            13: {1: 'self.take_action(delta_alt=1, delta_x=1, delta_y=-1, new_heading=2)',
                 2: 'self.take_action(delta_alt=1, delta_x=1, delta_y=0, new_heading=3)',
                 3: 'self.take_action(delta_alt=1, delta_x=1, delta_y=1, new_heading=4)',
                 4: 'self.take_action(delta_alt=1, delta_x=0, delta_y=1, new_heading=5)',
                 5: 'self.take_action(delta_alt=1, delta_x=-1, delta_y=1, new_heading=6)',
                 6: 'self.take_action(delta_alt=1, delta_x=-1, delta_y=0, new_heading=7)',
                 7: 'self.take_action(delta_alt=1, delta_x=-1, delta_y=-1, new_heading=8)',
                 8: 'self.take_action(delta_alt=1, delta_x=-0, delta_y=-1, new_heading=1)'},
            14: {1: 'self.take_action(delta_alt=1, delta_x=1, delta_y=0, new_heading=3)',
                 2: 'self.take_action(delta_alt=1, delta_x=1, delta_y=1, new_heading=4)',
                 3: 'self.take_action(delta_alt=1, delta_x=0, delta_y=1, new_heading=5)',
                 4: 'self.take_action(delta_alt=1, delta_x=-1, delta_y=1, new_heading=6)',
                 5: 'self.take_action(delta_alt=1, delta_x=-1, delta_y=0, new_heading=7)',
                 6: 'self.take_action(delta_alt=1, delta_x=-1, delta_y=-1, new_heading=8)',
                 7: 'self.take_action(delta_alt=1, delta_x=0, delta_y=-1, new_heading=1)',
                 8: 'self.take_action(delta_alt=1, delta_x=1, delta_y=-1, new_heading=2)'},
            15: {1: 'self.drop_package()',
                 2: 'self.drop_package()',
                 3: 'self.drop_package()',
                 4: 'self.drop_package()',
                 5: 'self.drop_package()',
                 6: 'self.drop_package()',
                 7: 'self.drop_package()',
                 8: 'self.drop_package()', }

        }

        print("here")


    def neighbors(self, arr, x, y, N):

        # https://stackoverflow.com/questions/32604856/slicing-outside-numpy-array
        # new_arr = np.zeros((N,N))

        left_offset = x - N // 2
        top_offset = y - N // 2

        # These are the 4 corners in real world coords
        left = max(0, x - N // 2)
        right = min(arr.shape[0], x + N // 2)
        top = max(0, y - N // 2)
        bottom = min(arr.shape[1], y + N // 2)

        window = arr[left:right + 1, top:bottom + 1]

        # newArr = np.zeros(self.original_map_volume['vol'][0].shape)
        # newArr[x-N//2:x+N//2+1,y-N//2:y+N//2+1] = window
        # return newArr
        return [window, left, top, right, bottom]

    def position_value(self, terrain, altitude, reward_dict, probability_dict):
        damage_probability = probability_dict['damage_probability'][altitude]
        # if terrain in probability_dict['stuck_probability'].keys():
        #     stuck_probability = probability_dict['stuck_probability'][terrain]
        # else:
        #     stuck_probability = 0.0
        # if terrain in probability_dict['sunk_probability'].keys():
        #     sunk_probability = probability_dict['sunk_probability'][terrain]
        # else:
        #     sunk_probability = 0.0
        damaged = np.random.random() < damage_probability
        # stuck = np.random.random() < stuck_probability
        # sunk = np.random.random() < sunk_probability
        self.package_state = 'DAMAGED' if damaged else 'OK'
        # package_state += '_STUCK' if stuck else ''
        # package_state += '_SUNK' if sunk else ''
        print("Package state:", self.package_state)
        reward = reward_dict[self.package_state]
        return reward

    def drop_package(self):
        # cannot drop at edge because next move could leave map
        local_coordinates = np.where(
            self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
        self.no_action_flag = False

        if local_coordinates[1] == 0 or \
            local_coordinates[2] == 0 or \
            local_coordinates[1] == self.map_volume['vol'].shape[1] - 1 or \
            local_coordinates[2] == self.map_volume['vol'].shape[1] - 1:
            print("NOACTION")
            self.no_action_flag = True
            self.reward = 0#-1 # might be redundant cauz u have a reward = 0 in the step function if the no action flag is true. Also this returns 0
            # self.package_state = 'OOB' # You might need it when you drop out of bounds
            #self.drop = True
            return 0
        self.drop = True
        alt = self.altitude
        drone_position = np.where(self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
        region = self.drop_package_grid_size_by_alt[self.altitude]
        neighbors, left, top, right, bottom = self.neighbors(self.original_map_volume['vol'][0], int(drone_position[1]),
                                              int(drone_position[2]), region)
        w = self.map_volume['vol'][0][left:right, top:bottom]
        is_hiker_in_neighbors = np.any(w == self.map_volume['feature_value_map']['hiker']['val'])

        x = np.random.randint(0, neighbors.shape[0])
        y = np.random.randint(0, neighbors.shape[1])

        value = neighbors[x, y] # It returns what kind of terrain is there in (number)
        pack_world_coords = (x + left, y + top) #(x,y) #
        terrain = self.original_map_volume['value_feature_map'][value]['feature'] # what kind of terrain is there (string) e.g. 'tree'
        reward = self.position_value(terrain, alt, self.drop_rewards, self.drop_probabilities) # OK:1, DMG: -1

        # distance in tiles ( we use Transpose and take the first element as the np.array for hiker pos is inside another array and καθετο vector
        self.pack_dist = max(abs(np.array(pack_world_coords) - np.array(self.hiker_position[-2:]).T[0])) # Chebyshev is better than Manhattan as with the latter you cannot go diagonal
        # if reward==1: # package state is OK
        if int(self.pack_dist) == 0:  # pack lands on top of the hiker. We need this condition to avoid the explosion of the inverse distance on 0
                self.reward = 3#  reward + self.alt_rewards[self.altitude] + 1# reward + is_hiker_in_neighbors + 1 # altitude is implied so you might need to put it in
        else:
                self.reward = (reward + self.alt_rewards[self.altitude])/((self.pack_dist** 2) + 1e-7)#reward + self.alt_rewards[self.altitude] + 1/((self.pack_dist** 2) + 1e-7)#

        self.package_position = pack_world_coords
        self.package_dropped = True
        x = eval(self.actionvalue_heading_action[7][self.heading]) # ????

    def take_action(self, delta_alt=0, delta_x=0, delta_y=0, new_heading=1):
        # print("stop")
        vol_shape = self.map_volume['vol'].shape

        local_coordinates = np.where(
            self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
        if int(local_coordinates[1]) + delta_y < 0 or \
                int(local_coordinates[2]) + delta_x < 0 or \
                int(local_coordinates[1] + delta_y > vol_shape[1] - 1) or \
                int(local_coordinates[2] + delta_x > vol_shape[2] - 1):
            return 0

        # todo update with shape below
        forbidden = [(0, 0), (vol_shape[1] - 1, 0),
                     (vol_shape[1] - 1, vol_shape[1] - 1), (0, vol_shape[1] - 1)]
        #print((int(local_coordinates[1]) + delta_y, int(local_coordinates[2]) + delta_x), forbidden)
        if (int(local_coordinates[1]) + delta_y, int(local_coordinates[2]) + delta_x) in forbidden:
            return 0

        new_alt = self.altitude + delta_alt if self.altitude + delta_alt < 4 else 3
        if new_alt < 0:
            return 0

        # put back the original
        self.map_volume['vol'][self.altitude][local_coordinates[1], local_coordinates[2]] = float(
            self.original_map_volume['vol'][local_coordinates])

        # self.map_volume['flat'][local_coordinates[1],local_coordinates[2]] = float(self.original_map_volume['flat'][local_coordinates[1],local_coordinates[2]])
        # self.map_volume['img'][local_coordinates[1],local_coordinates[2]] = self.original_map_volume['img'][local_coordinates[1],local_coordinates[2]]
        # put the hiker back
        self.map_volume['vol'][self.hiker_position] = self.map_volume['feature_value_map']['hiker']['val']
        # self.map_volume['flat'][self.hiker_position[1],self.hiker_position[2]] = self.map_volume['feature_value_map']['hiker']['val']
        # self.map_volume['img'][self.hiker_position[1],self.hiker_position[2]] = self.map_volume['feature_value_map']['hiker']['color']
        # put the drone in
        # self.map_volume['flat'][local_coordinates[1]+delta_y,local_coordinates[2]+delta_x] = self.map_volume['feature_value_map']['drone'][new_alt]['val']
        self.map_volume['vol'][new_alt][local_coordinates[1] + delta_y, local_coordinates[2] + delta_x] = \
        self.map_volume['feature_value_map']['drone'][new_alt]['val']
        # self.map_volume['img'][local_coordinates[1] + delta_y, local_coordinates[2] + delta_x] = self.map_volume['feature_value_map']['drone'][new_alt]['color']
        # for i in range(4,-1,-1):
        #     if self.map_volume['vol'][i][local_coordinates[1],local_coordinates[2]]:
        #         self.map_volume['flat'][int(local_coordinates[1]),int(local_coordinates[2])] = float(self.map_volume['vol'][i][int(local_coordinates[1]),int(local_coordinates[2])])
        #         break
        self.altitude = new_alt
        self.heading = new_heading

        if self.real_actions: # ONLY WHEN MAVSIM IS ACTIVE
            drone_position = np.where(
                self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])

            success = self.mavsimhandler.fly_path(coordinates=[self.reference_coordinates[0] + int(drone_position[1]),
                                                               self.reference_coordinates[1] + int(drone_position[2])],
                                                  altitude=self.altitude)

        return 1

    def check_for_hiker(self):
        drone_position = np.where(
            self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
        # hiker_position = np.where(self.map_volume['vol'] == self.map_volume['feature_value_map']['hiker'][0])
        # print("drone",drone_position)
        # print("hiker",self.hiker_position)
        # drone or hiker coords format (alt,x,y)
        if (drone_position[1], drone_position[2]) == (self.hiker_position[1], self.hiker_position[2]):
            return 1
        return 0
        # return int(self.map_volume[0]['hiker'][int(local_coordinates[0]),int(local_coordinates[1])])

    def check_for_crash(self):
        # if drone on altitude 0, crash
        if self.altitude == 0:
            return 1

        # if len(self.map_volume[0]['drone'].nonzero()[0]):
        #     return 1
        # at any other altutidue, check for an object at the drone's position
        drone_position = np.where(
            self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
        return int(self.original_map_volume['vol'][drone_position])
        # drone_position = self.map_volume[self.altitude]['drone'].nonzero()
        # for i in range(self.altitude,4):
        #
        #     for key in self.map_volume[i]:
        #         if key == 'drone' or key == 'map':
        #             continue
        #         #just check if drone position is returns a non-zero
        #         if self.map_volume[i][key][int(drone_position[0]),int(drone_position[1])]:
        #             return 1
        # return 0

    #PREVIOUS WORKING STEP

    # def step(self, action):
    #         ''' return next observation, reward, finished, success '''
    #
    #         action = int(action)
    #         info = {}
    #         info['success'] = False
    #
    #         done = False
    #         drone = np.where(
    #             self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
    #         hiker = self.hiker_position
    #         # You should never reach as this is the state(t-1) distane. After eval you get the new distance
    #         self.dist = np.linalg.norm(np.array(drone[-2:]) - np.array(hiker[-2:]))  # we remove height from the equation so we avoid going diagonally down
    #
    #         # Here the action takes place
    #         x = eval(self.actionvalue_heading_action[action][self.heading])
    #         # Here you should have the distance, after the drone has moved and the map has been updated
    #         # A new observation is generated which we do not see cauz we reset() and render in the step function
    #         observation = self.generate_observation()
    #
    #         crash = self.check_for_crash()
    #         info['success'] = not crash
    #         #self.render()
    #         self.crash = crash
    #         if crash:
    #             reward = -1
    #             done = True
    #             print("CRASH")
    #             if self.restart_once_done:  # HAVE IT ALWAYS TRUE!!! It learned the first time WITHOUT RESETING FROM CRASH
    #                 #observation = self.reset()
    #                 return (observation, reward, done, info)
    #             # return (self.generate_observation(), reward, done, info)
    #         # if self.dist < self.dist_old:
    #         #     reward = 1 / self.dist  # Put it here to avoid dividing by zero when you crash on the hiker
    #         # else:
    #         #     reward = -1 / self.dist
    #         if self.check_for_hiker():
    #             done = True
    #             reward = 1# + self.alt_rewards[self.altitude]
    #             # reward = 1 + 1 / self.dist
    #             print('SUCCESS!!!')
    #             if self.restart_once_done:  # HAVE IT ALWAYS TRUE!!!
    #                 #observation = self.reset()
    #                 return (observation, reward, done, info)
    #         # print("state", [ self.observation[self.altitude]['drone'].nonzero()[0][0],self.observation[self.altitude]['drone'].nonzero()[1][0]] )
    #         self.dist_old = self.dist
    #         #reward = (self.alt_rewards[self.altitude] * 0.1) * ( 1/((self.dist** 2) + 1e-7) )  # -0.01 + # previous reward = (self.alt_rewards[self.altitude] * 0.1) * ( 1 / self.dist** 2 + 1e-7 )  # -0.01 + #
    #         reward = -0.01 # If you put -0.1 then it prefers to go down and crash all the time for (n-step=32)!!!
    #         return (observation, reward, done, info)

    # TRAINING WITH THIS TILL May 3, 2019
    # def step(self, action):
    #     ''' return next observation, reward, finished, success '''
    #
    #     action = int(action)
    #     info = {}
    #     info['success'] = False
    #
    #     done = False
    #     drone_old = np.where(
    #         self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
    #     hiker = self.hiker_position
    #     # Do the action (drone is moving)
    #     x = eval(self.actionvalue_heading_action[action][self.heading])
    #
    #     observation = self.generate_observation()
    #     drone = np.where(
    #         self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
    #     self.dist = np.linalg.norm(np.array(drone[-2:]) - np.array(hiker[-2:])) # we remove height from the equation so we avoid going diagonally down
    #
    #     crash = self.check_for_crash()
    #     info['success'] = not crash
    #     self.crash = crash
    #     # BELOW WAS WORKING FINE FOR FINDING HIKER
    #     # reward = (self.alt_rewards[self.altitude]*0.1)*(1/self.dist**2+1e-7)# + self.drop*self.reward (and comment out the reward when you drop and terminate episode
    #     #reward = (self.alt_rewards[self.altitude]*0.1)*((1/(self.dist**2)+1e-7)) # -0.01 + # The closer we are to the hiker the more important is to be close to its altitude
    #     if crash:
    #         reward = -1
    #         done = True
    #         print("CRASH")
    #         if self.restart_once_done: # HAVE IT ALWAYS TRUE!!! It learned the first time WITHOUT RESETING FROM CRASH
    #             return (observation, reward, done, info)
    #         #return (self.generate_observation(), reward, done, info)
    #     if self.no_action_flag == True:
    #         reward = 0
    #         done = True
    #         if self.restart_once_done: # HAVE IT ALWAYS TRUE!!!
    #             return (observation, reward, done, info)
    #     # if self.dist < self.dist_old:
    #     #     reward = 1 / self.dist  # Put it here to avoid dividing by zero when you crash on the hiker
    #     # else:
    #     #     reward = -1 / self.dist
    #     if self.drop:#self.check_for_hiker():
    #         done = True
    #         #reward = 1 + self.alt_rewards[self.altitude] # THIS WORKS FOR FINDING THE HIKER
    #         if self.check_for_hiker(): # If you are on top of the hiker you get additional 1 point
    #             reward = 1 + self.reward + self.alt_rewards[self.altitude]
    #         else:
    #             reward = self.reward + self.alt_rewards[self.altitude] # (try to multiply them and see if it makes a difference!!! Here tho u reward for dropping low alt
    #         print('DROP!!!', 'self.reward=', self.reward, 'alt_reward=', self.alt_rewards[self.altitude])
    #         if self.restart_once_done: # HAVE IT ALWAYS TRUE!!!
    #             return (observation, reward, done, info)
    #     # print("state", [ self.observation[self.altitude]['drone'].nonzero()[0][0],self.observation[self.altitude]['drone'].nonzero()[1][0]] )
    #     self.dist_old = self.dist
    #     # HERE YOU SHOULD HAVE THE REWARD IN CASE IT CRASHES AT ALT=0 OR IN GENERAL AFTER ALL CASES HAVE BEEN CHECKED!!!
    #     if self.check_for_hiker(): # On top of the hiker avoiding infinity with distance depenedent reward function
    #         #print("hiker found:", self.check_for_hiker())
    #         # reward = (self.alt_rewards[self.altitude]*0.1)*(1/self.dist**2+1e-7) + self.drop*self.reward (and comment out the reward when you drop and terminate episode
    #         reward = 0 #1 + self.alt_rewards[self.altitude]
    #     else:
    #         # We don't want the drone to wonder around away from the hiker so we keep it close
    #         # The reward below though with PPO will make the drone just going close and around the hiker forever as it gather reward all the time
    #         reward = (self.alt_rewards[self.altitude]*0.1)*((1/((self.dist**2)+1e-7))) # -0.01 + # The closer we are to the hiker the more important is to be close to its altitude
    #         #print("scale:",(1/((self.dist**2+1e-7))), "dist=",self.dist+1e-7, "alt=", self.altitude, "drone:",drone, "hiker:", hiker,"found:", self.check_for_hiker())
    #     return (self.generate_observation(), reward, done, info)

    # def step(self, action):
    #     ''' return next observation, reward, finished, success '''
    #
    #     action = int(action)
    #     info = {}
    #     info['success'] = False
    #
    #     done = False
    #     drone_old = np.where(
    #         self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
    #     hiker = self.hiker_position
    #     # Do the action (drone is moving)
    #     x = eval(self.actionvalue_heading_action[action][self.heading])
    #
    #     observation = self.generate_observation()
    #     drone = np.where(
    #         self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
    #     self.dist = np.linalg.norm(np.array(drone[-2:]) - np.array(hiker[-2:])) # we remove height from the equation so we avoid going diagonally down
    #
    #     crash = self.check_for_crash()
    #     info['success'] = not crash
    #
    #     if crash:
    #         reward = -1
    #         done = True
    #         print("CRASH")
    #         if self.restart_once_done: # HAVE IT ALWAYS TRUE!!! It learned the first time WITHOUT RESETING FROM CRASH
    #             return (observation, reward, done, info)
    #
    #     if self.no_action_flag == True:
    #         reward = 0
    #         done = True
    #         if self.restart_once_done: # HAVE IT ALWAYS TRUE!!!
    #             return (observation, reward, done, info)
    #
    #     if self.drop:
    #         done = True
    #         reward = self.reward + self.alt_rewards[self.altitude] # (try to multiply them and see if it makes a difference!!! Here tho u reward for dropping low alt
    #         print('DROP!!!', 'self.reward=', self.reward, 'alt_reward=', self.alt_rewards[self.altitude])
    #         if self.restart_once_done: # HAVE IT ALWAYS TRUE!!!
    #             return (observation, reward, done, info)
    #     # print("state", [ self.observation[self.altitude]['drone'].nonzero()[0][0],self.observation[self.altitude]['drone'].nonzero()[1][0]] )
    #     self.dist_old = self.dist
    #     reward = -0.01#(self.alt_rewards[self.altitude]*0.1)*((1/((self.dist**2)+1e-7))) # -0.01 + # The closer we are to the hiker the more important is to be close to its altitude
    #     #print("scale:",(1/((self.dist**2+1e-7))), "dist=",self.dist+1e-7, "alt=", self.altitude, "drone:",drone, "hiker:", hiker,"found:", self.check_for_hiker())
    #     return (self.generate_observation(), reward, done, info)

    def step(self, action):
        ''' return next observation, reward, finished, success '''

        action = int(action)
        info = {}
        info['success'] = False

        done = False

        crash = self.check_for_crash()
        info['success'] = not crash

        if crash:
            reward = -1
            done = True
            print("CRASH")
            if self.restart_once_done: # HAVE IT ALWAYS TRUE!!! It learned the first time WITHOUT RESETING FROM CRASH
                return (self.generate_observation(), reward, done, info) # You should get the previous obs so no change here, or return obs=None

        # Do the action (drone is moving). If we crash we dont perform an action so no new observation
        x = eval(self.actionvalue_heading_action[action][self.heading]) # actionvalue dict contains take_action function given the arguments.
        if self.no_action_flag == True:
            reward = self.reward#-1#TODO: it was 0 in all successful training session with PPO. TRY -1 so it avoids dropping at the edges!!!! ahouls be = self.reward and fix self. reward in the drop package function
            done = True
            if self.restart_once_done: # HAVE IT ALWAYS TRUE!!!
                return (self.generate_observation(), reward, done, info)

        # Multiple packages
        if self.drop:
            self.drop = 0
            reward = self.reward
            print('DROP!!!', 'self.reward=', self.reward, 'alt_reward=', self.alt_rewards[self.altitude],
                  'hiker-package distance=', self.pack_dist)
            self.countdrop = self.countdrop + 1
            if self.countdrop >= 1: # 3 drops
                done = True
                if self.restart_once_done:  # HAVE IT ALWAYS TRUE!!!
                    return (self.generate_observation(), reward, done, info)
            # reward = self.reward + self.alt_rewards[self.altitude] # (try to multiply them and see if it makes a difference!!! Here tho u reward for dropping low alt

            # print('DROP!!!', 'self.reward=', self.reward, 'alt_reward=', self.alt_rewards[self.altitude], 'hiker-package distance=', self.pack_dist)
            # Below you should commented out so you just continue without the return. You put this inside the if self.countdrop>2
            # if self.restart_once_done: # HAVE IT ALWAYS TRUE!!!
            #     return (observation, reward, done, info)

        # # One package
        # if self.drop:
        #     done = True
        #     # reward = self.reward + self.alt_rewards[self.altitude] # (try to multiply them and see if it makes a difference!!! Here tho u reward for dropping low alt
        #     reward = self.reward
        #     print('DROP!!!', 'self.reward=', self.reward, 'alt_reward=', self.alt_rewards[self.altitude], 'hiker-package distance=', self.pack_dist)
        #     if self.restart_once_done: # HAVE IT ALWAYS TRUE!!!
        #         return (observation, reward, done, info)

        # print("state", [ self.observation[self.altitude]['drone'].nonzero()[0][0],self.observation[self.altitude]['drone'].nonzero()[1][0]] )
        # self.dist_old = self.dist
        reward = -0.01#-0.01#(self.alt_rewards[self.altitude]*0.1)*((1/((self.dist**2)+1e-7))) # -0.01 + # The closer we are to the hiker the more important is to be close to its altitude
        #print("scale:",(1/((self.dist**2+1e-7))), "dist=",self.dist+1e-7, "alt=", self.altitude, "drone:",drone, "hiker:", hiker,"found:", self.check_for_hiker())
        return (self.generate_observation(), reward, done, info)

    def reset(self):
        self.dist_old = 1000
        self.drop = False
        self.countdrop = 0
        self.no_action_flag = False
        self.heading = 1#random.randint(1, 8)
        self.altitude = 2#random.randint(1,3)
        self.reward = 0
        self.crash = 0
        self.package_dropped = 0
        self.package_position = ()

        """ start DRAWN world (Un)comment BELOW this part if you want a custom map """
        # drawn_map = \
        #     [[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 26, 2, 2, 2, 2, 2, 2, 2],
        #      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 26, 26, 2, 2, 2, 2, 2, 2, 2, 2],
        #      [2, 2, 2, 2, 2, 2, 2, 2, 2, 26, 26, 25, 2, 2, 2, 2, 2, 2, 2, 2],
        #      [2, 2, 2, 2, 2, 2, 2, 2, 26, 26, 25, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #      [2, 2, 2, 2, 2, 2, 2, 2, 26, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]]
        # drawn_map = np.array(drawn_map)
        # self.map_volume = CNP.create_custom_map(drawn_map)
        # self.hiker = (3, 3)
        # self.drone = (7, 7)
        # self.altitude = 1
        """ end DRAWN world """

        #####START COMMMENT OUT
        # #Random generated map
        # # start = random.choice([1,1,1,1,1,1,1,1,1,1])
        # # stop = random.choice([13,13,13,13,13,13,13,13,13,13])
        # # random_integers = np.random.random_integers(start,stop,(20,20))
        # # flag = bn.rvs(p=0.99, size=(20,20))
        # # #add 10% (1-p) of any value
        # # other_features = np.full((20,20),33)
        # # random_integers[flag==0] = other_features[flag==0]
        # # self.map_volume = CNP.create_custom_map(random_integers)#CNP.create_custom_map(np.random.random_integers(start,stop,(self.mapw,self.maph)))#CNP.map_to_volume_dict(self._map[0],self._map[1], self.mapw, self.maph)#CNP.create_custom_map(np.random.random_integers(start,stop,(self.mapw,self.maph))) #CNP.create_custom_map(random.choice(self.custom_maps))
        # # # Set hiker's and drone's locations
        # # #hiker = (random.randint(2, self.map_volume['vol'].shape[1] - 1), random.randint(2, self.map_volume['vol'].shape[1] - 2)) #(8,8) #
        # # #if self.dropping:
        # # hiker = (random.randint(2, self.map_volume['vol'].shape[1] - 2), random.randint(2, self.map_volume['vol'].shape[1] - 2))#(10,10)#(random.randint(2, self.map_volume['vol'].shape[1] - 2), random.randint(2, self.map_volume['vol'].shape[1] - 2))  #random.choice([(4,5),(5,5),(5,4),(4,4)]) (7,8) #
        # all_no_goes = [] # all points hiker is not allowed to go (e.g. water)
        # # better random map
        # just_grass = np.full((20, 20), 2)
        # # add some trail, trees
        # updated_map = self.add_blob(just_grass, 15, 5)[0]
        # for i in range(random.randint(1, 10)):
        #     updated_map = self.add_blob(updated_map, 50, random.choice([1, 3]))[0]
        #
        # # add some water (maybe)
        # if random.randint(0, 1):
        #     updated_map, no_go_points = self.add_blob(updated_map, 100, 15)
        #     all_no_goes.append(no_go_points)
        # # add some mountain ridges
        # updated_map, no_go_points = self.add_blob(updated_map, 75, 26)
        # all_no_goes.append(no_go_points)
        # # a few small mountain ridges
        # for i in range(random.randint(1, 5)):
        #     updated_map, no_go_points = self.add_blob(updated_map, random.randint(1, 10), 25)
        #     all_no_goes.append(no_go_points)
        # # add some bushes
        # # small clusters, 5 times
        # for i in range(random.randint(1, 8)):
        #     updated_map = self.add_blob(updated_map, random.randint(1, 5), 4)[0]
        # # add one campfire
        # updated_map, no_go_points = self.add_blob(updated_map, 0, 33)
        # all_no_goes.append(no_go_points)
        #
        # self.map_volume = CNP.create_custom_map(updated_map)
        #
        # # self.map_volume = CNP.create_custom_map(np.random.random_integers(start, stop, (20, 20)))#CNP.create_custom_map(np.random.random_integers(start, stop, (20, 20)))#CNP.map_to_volume_dict(self._map[0],self._map[1], self.mapw, self.maph)#CNP.create_custom_map(self.custom_map)#CNP.create_custom_map(np.random.random_integers(start, stop, (20, 20)))#CNP.map_to_volume_dict(self._map[0],self._map[1], self.mapw, self.maph)#CNP.create_custom_map(np.random.random_integers(start, stop, (10, 10)))#CNP.create_custom_map(self.custom_map)#CNP.create_custom_map(np.random.random_integers(start, stop, (10, 10)))
        #
        # # Set hiker's and drone's location
        # # hiker = (random.randint(2, self.map_volume['vol'].shape[1] - 1), random.randint(2, self.map_volume['vol'].shape[1] - 2)) #(8,8) #
        # # (8, 1)  # (6,3)#
        # hiker = (
        #     random.randint(3, self.map_volume['vol'].shape[1] - 3),
        #     random.randint(3, self.map_volume['vol'].shape[1] - 3))
        # while self.hiker_in_no_go_list(hiker, all_no_goes): Place the hiker but check if is placed on a no go feature
        #     hiker = (random.randint(3, self.map_volume['vol'].shape[1] - 3),
        #              random.randint(3, self.map_volume['vol'].shape[1] - 3))
        ##################

        ####actual maps### (Un)comment below if you DONT want to use custom maps
        #######################################
        self._map = random.choice(self.maps)
        if self._map[0]==1:
            path = './gym_gridworld/'
            filename = '{}-{}.mp'.format(self._map[0], self._map[1])
            # Create custom map needs a numpy array
            cust_map = pickle.load(open(path + 'maps/' + filename, 'rb'))
            self.map_volume = cust_map#CNP.create_custom_map(cust_map)
        else:
            self.map_volume = CNP.map_to_volume_dict(self._map[0], self._map[1], self.mapw, self.maph)

        map_ = self.map_volume['flat']
        # place the hiker
        hiker_safe_points = []
        for val in self.masks['hiker']:
            where_array = np.where(map_ == val)
            hiker_safe_points = hiker_safe_points + [(x, y) for x, y in zip(where_array[0], where_array[1]) if
                                                     x >= 3 and y >= 3 and x <= self.map_volume['vol'].shape[
                                                         1] - 3 and y <= self.map_volume['vol'].shape[1] - 3]
        """ Specify Hiker location"""
        # self.hiker = random.choice(hiker_safe_points)
        self.hiker = (12,12)#(11,8) #(18, 16)

        # int(self.original_map_volume['vol'][hiker])
        # place the drone
        drone_safe_points = []
        for val in self.masks[self.altitude]:
            where_array = np.where(map_ == val)
            drone_safe_points = drone_safe_points + [(x, y) for x, y in zip(where_array[0], where_array[1]) if
                                                     x >= 3 and y >= 3 and x <= self.map_volume['vol'].shape[
                                                         1] - 3 and y <= self.map_volume['vol'].shape[1] - 3]
        """ Drone around the hiker """
        # D = distance.cdist([self.hiker], drone_safe_points, 'chebyshev').astype(int) # Distances from hiker to all drone safe points
        #
        # # print('Distance:',D[0])
        # # print('Hiker',hiker)
        # # print('safe_drone',drone_safe_points)
        # # print('safe_hiker', hiker_safe_points)
        #
        # k = 50 # k closest. There might be cases in which you have very few drone safe points (e.g. 3) and only one will be really close
        # if k> np.array(drone_safe_points).shape[0]:
        #     k = np.array(drone_safe_points).shape[0] - 1 # Cauz we index from 0 but shape starts from 1 to max shape
        # indx = np.argpartition(D[0],k) # Return the indices of the k closest distances to the hiker. The [0] is VITAL!!!
        # # # Use the index to retrieve the k closest safe coords to the hiker
        # closest_neighs = np.array(drone_safe_points)[indx[:k]] # You need to have the safe points as array and not list
        # self.drone = tuple(random.choice(closest_neighs))

        """ DONT MIND """
        # NOTES: The first element in the array of safe points might be the hiker position
        # To move away from hiker increase k and define h=k/2 and discard the h first closest_neighs - 9 suppose to be the max of the closest in an open area. So just use dividends of 9 to discard
        # drone = (hiker[0]-2, hiker[1]-3)
        # drone = random.choice([(hiker[0] - 1, hiker[1] - 1), (hiker[0] - 1, hiker[1] ), (hiker[0], hiker[1] - 1 )])

        """Random away location + safe check"""
        # drone = random.choice([(hiker[0] - 5, hiker[1] - 3), (hiker[0] - 6, hiker[1]), (hiker[0], hiker[1] - 4), (hiker[0] - 6, hiker[1] - 7)])
        # self.drone = random.choice([(self.hiker[0] - 8, self.hiker[1] - 3), (self.hiker[0] - 10, self.hiker[1]), (self.hiker[0], self.hiker[1] - 9),
        #                        (self.hiker[0] - 12, self.hiker[1] - 7)])
        # self.drone = random.choice([(self.hiker[0] - 12, self.hiker[1] ), (self.hiker[0] - 10, self.hiker[1]), (self.hiker[0]-11, self.hiker[1]),
        #                        (self.hiker[0] - 12, self.hiker[1] - 12), (self.hiker[0] - 10, self.hiker[1] - 10), (self.hiker[0] - 11, self.hiker[1] - 11),
        #                        (self.hiker[0], self.hiker[1] - 12), (self.hiker[0], self.hiker[1] - 10), (self.hiker[0], self.hiker[1] - 11),
        #                             ])
        # times = 0
        # while self.drone not in drone_safe_points:
        #     # self.drone = random.choice([(self.hiker[0] - 5, self.hiker[1] - 3), (self.hiker[0] - 6, self.hiker[1]), (self.hiker[0], self.hiker[1] - 4),
        #     #                        (self.hiker[0] - 6, self.hiker[1] - 7)])
        #     self.drone = random.choice([(self.hiker[0] - 12, self.hiker[1]), (self.hiker[0] - 10, self.hiker[1]),
        #                                 (self.hiker[0] - 11, self.hiker[1]), (self.hiker[0] - 12, self.hiker[1] - 12),
        #                                 (self.hiker[0] - 10, self.hiker[1] - 10), (self.hiker[0] - 11, self.hiker[1] - 11),
        #                                 (self.hiker[0], self.hiker[1] - 12), (self.hiker[0], self.hiker[1] - 10),
        #                                 (self.hiker[0], self.hiker[1] - 11),
        #                                 ])
        #     # print('non safe reset drone pos')
        #     if times==10:
        #         print('max reps reached so reset hiker')
        #         self.hiker = random.choice(hiker_safe_points)
        #         # self.altitude = random.randint(1, 3) # NO cauz then you have to recalculate drone safe points
        #         times = 0
        #     times = times + 1

        """ All safe points included for final training """
        # self.drone = random.choice(drone_safe_points)
        """ Custom location """
        self.drone = (6,6)#(18,11)

        self.original_map_volume = copy.deepcopy(self.map_volume)
        self.hiker_drone_dist = max(abs(np.array(self.hiker) - np.array(self.drone)))
        print('>>>> hiker-drone initial distance = ',self.hiker_drone_dist)

        self.real_actions = False
        # put the drone in
        self.map_volume['vol'][self.altitude][self.drone[0], self.drone[1]] = \
        self.map_volume['feature_value_map']['drone'][self.altitude]['val']
        # self.map_volume['flat'][drone[0], drone[1]] = self.map_volume['feature_value_map']['drone'][self.altitude][
        #     'val']
        self.map_volume['img'][self.drone[0], self.drone[1]] = self.map_volume['feature_value_map']['drone'][self.altitude][
            'color']
        # self.map_volume[altitude]['drone'][local_y, local_x] = 1.0
        # put the hiker in@ altitude 0
        self.map_volume['vol'][0][self.hiker[0], self.hiker[1]] = self.map_volume['feature_value_map']['hiker']['val']
        # self.map_volume['flat'][hiker[0], hiker[1]] = self.map_volume['feature_value_map']['hiker']['val']
        self.map_volume['img'][self.hiker[0], self.hiker[1]] = self.map_volume['feature_value_map']['hiker']['color']
        self.hiker_position = np.where(self.map_volume['vol'] == self.map_volume['feature_value_map']['hiker']['val'])

        self.image_layers[0] = self.create_image_from_volume(0)
        self.image_layers[1] = self.create_image_from_volume(1)
        self.image_layers[2] = self.create_image_from_volume(2)
        self.image_layers[3] = self.create_image_from_volume(3)
        self.image_layers[4] = self.create_image_from_volume(4)

        observation = self.generate_observation()
        self.render()
        return observation

    def add_blob(self, map_array, n_cycles, value):
        points = []
        random_point = np.random.randint(0, map_array.shape[0], (1, 2))#assumes a square
        points.append(random_point)
        for i in range(n_cycles):
            a_point = random.choice(points)
            pertubation = np.random.randint(-1, 1, (1, 2))
            added_point = a_point + pertubation
            if not self.arreq_in_list(added_point,points):
                points.append(a_point + pertubation)
        return_array = np.copy(map_array)
        for point in points:
            return_array[point[0][0], point[0][1]] = value
        return (return_array,points)

    def plane_image(self, heading, color):
        '''Returns a 5x5 image as np array'''
        for point in self.planes[heading][0]:
            self.planes[heading][1][point[0], point[1]] = color
        return self.planes[heading][1]

    def create_image_from_volume(self, altitude):
        canvas = np.zeros((self.map_volume['vol'].shape[1], self.map_volume['vol'].shape[1], 3), dtype=np.uint8)
        og_vol = self.original_map_volume
        combinations = list(itertools.product(range(0, canvas.shape[0]), range(0, canvas.shape[0])))
        for x, y in combinations:
            if og_vol['vol'][altitude][x, y] == 0.0:
                canvas[x, y, :] = [255, 255, 255]
            else:
                canvas[x, y, :] = og_vol['value_feature_map'][og_vol['vol'][altitude][x, y]]['color']

        return imresize(canvas, self.factor * 100, interp='nearest')

    def create_nextstep_image(self):
        canvas = np.zeros((5, 5, 3), dtype=np.uint8)
        slice = np.zeros((5, 5))
        drone_position = np.where(
            self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
        drone_position_flat = [int(drone_position[1]), int(drone_position[2])]
        # hiker_found = False
        # hiker_point = [0, 0]
        # hiker_background_color = None
        column_number = 0
        for xy in self.possible_actions_map[self.heading]:
            if drone_position_flat[0] + xy[0] >= 0 and drone_position_flat[1] + xy[1] >= 0 and drone_position_flat[0] + \
                    xy[0] <= self.map_volume['vol'].shape[1] - 1 and drone_position_flat[1] + xy[1] <= \
                    self.map_volume['vol'].shape[2] - 1:

                # try:
                # no hiker if using original
                column = self.map_volume['vol'][:, drone_position_flat[0] + xy[0], drone_position_flat[1] + xy[1]]

            # except IndexError:
            else:
                column = [1., 1., 1., 1., 1.]
            slice[:, column_number] = column
            column_number += 1
            #print("ok")
        # put the drone in
        # cheat
        slice[self.altitude, 2] = int(self.map_volume['vol'][drone_position])
        combinations = list(itertools.product(range(0, canvas.shape[0]), range(0, canvas.shape[0])))
        for x, y in combinations:
            if slice[x, y] == 0.0:
                canvas[x, y, :] = [255, 255, 255]

            else:
                canvas[x, y, :] = self.map_volume['value_feature_map'][slice[x, y]]['color']

        # increase the image size, then put the hiker in
        canvas = imresize(canvas, self.factor * 100, interp='nearest')
        self.ego = np.flip(slice,0)
        return imresize(np.flip(canvas, 0), 20*self.map_volume['vol'].shape[2], interp='nearest')

    def generate_observation(self):
        obs = {}
        obs['volume'] = self.map_volume
        image_layers = copy.deepcopy(self.image_layers)
        map = copy.deepcopy(self.original_map_volume['img'])

        # put the drone in the image layer # we need to use the self.drone no need for np.where
        drone_position = np.where(
            self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
        drone_position = (int(drone_position[1]) * self.factor, int(drone_position[2]) * self.factor)
        for point in self.planes[self.heading][0]:
            image_layers[self.altitude][drone_position[0] + point[0], drone_position[1] + point[1], :] = \
            self.map_volume['feature_value_map']['drone'][self.altitude]['color']

        # put the hiker in the image layers
        hiker_position = (int(self.hiker_position[1] * self.factor), int(self.hiker_position[2]) * self.factor)
        for point in self.hikers[0][0]:
            image_layers[0][hiker_position[0] + point[0], hiker_position[1] + point[1], :] = \
            self.map_volume['feature_value_map']['hiker']['color']

        # map = self.original_map_volume['img']
        map = imresize(map, self.factor * 100, interp='nearest')  # resize by factor of 5
        # add the hiker
        hiker_position = (int(self.hiker_position[1] * 5), int(self.hiker_position[2]) * 5)
        # map[hiker_position[0]:hiker_position[0]+5,hiker_position[1]:hiker_position[1]+5,:] = self.hiker_image
        for point in self.hikers[0][0]:
            map[hiker_position[0] + point[0], hiker_position[1] + point[1], :] = \
            self.map_volume['feature_value_map']['hiker']['color']
        # add the drone
        drone_position = np.where(
            self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
        drone_position = (int(drone_position[1]) * 5, int(drone_position[2]) * 5)
        for point in self.planes[self.heading][0]:
            map[drone_position[0] + point[0], drone_position[1] + point[1], :] = \
                self.map_volume['feature_value_map']['drone'][self.altitude]['color']

        # maybe put the package in
        # print('pack drop flag',self.package_dropped)
        if self.package_dropped:
            self.package_dropped = 0
            package_position = (int(self.package_position[0] * 5), int(self.package_position[1]) * 5)
            for point in self.package[self.package_state][0]:
                # print(point, package_position)
                map[package_position[0] + point[0], package_position[1] + point[1], :] = [94, 249, 242]

        # map[drone_position[0]:drone_position[0] + 5,drone_position[1]:drone_position[1] + 5] = self.plane_image(self.heading,self.map_volume['feature_value_map']['drone'][self.altitude]['color'])

        # map = imresize(map, (1000,1000), interp='nearest')

        '''vertical slices at drone's position'''
        # drone_position = np.where(
        #     self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])

        nextstepimage = self.create_nextstep_image()
        obs['nextstepimage'] = nextstepimage
        obs['img'] = map
        obs['image_layers'] = image_layers
        obs['altitude'] = self.altitude
        obs['image_volume'] = np.stack(im for im in obs['image_layers'].values())#.reshape((100,100,5,3))
        obs['joined'] = np.concatenate([obs['img'],obs['nextstepimage']],axis=1)
        return obs

    def render(self, mode='human', close=False):

        # return
        if self.verbose == False:
           return
        # img = self.observation
        # map = self.original_map_volume['img']
        obs = self.generate_observation()
        self.map_image = obs['img']
        self.alt_view = obs['nextstepimage']
        # fig = plt.figure(self.this_fig_num)
        # img = np.zeros((20,20,3))
        # img[10,10,0] = 200
        # img[10,10,1] = 153
        # img[10,10,2] = 255

        #fig = plt.figure(0)
        #fig1 = plt.figure(1)
        #plt.clf()
        # plt.subplot(211)
        # plt.imshow(self.map_image)
        # plt.subplot(212)
        # plt.imshow(self.alt_view)
        # #fig.canvas.draw()
        # #plt.show()
        # plt.pause(0.00001)
        return

    def _close_env(self):
        plt.close(1)
        return


# a = GridworldEnv(map_x=70, map_y=50, local_x=2, local_y=2, hiker_x=10, heading=1, altitude=3)
# a.reset()
# # a.step(12)
# #
# # def show_img():
#
# for i in range(10000):
#     a.step(random.randint(1, 14))
#     # local_coordinates = a.map_volume[a.altitude]['drone'].nonzero()
#     # print("coordinates", local_coordinates, a.heading)
#     if a.check_for_crash():
#         print("crash at altitude", a.altitude)
#         a.reset()
#         time.sleep(0.5)
#     if a.check_for_hiker():
#         print("hiker after", i)
#         break

# print(a.check_for_crash())
print('complete')