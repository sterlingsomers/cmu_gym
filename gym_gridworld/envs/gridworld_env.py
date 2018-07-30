import gym
import sys
import os
import time
import copy
import math
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from PIL import Image as Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import threading
import random
import pygame
from scipy.misc import imresize

import datetime

import create_np_map as CNP

from mavsim_server import MavsimHandler

# define colors
# 0: black; 1 : gray; 2 : blue; 3 : green; 4 : red
COLORS = {0:[0.0,0.0,0.0], 1:[0.5,0.5,0.5], \
          2:[0.0,0.0,1.0], 3:[0.0,1.0,0.0], \
          4:[1.0,0.0,0.0], 6:[1.0,0.0,1.0], \
          7:[1.0,1.0,0.0]}

class GridworldEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    num_env = 0

    def __init__(self,map_x=0,map_y=0,local_x=0,local_y=0,heading=1,altitude=2,hiker_x=5,hiker_y=5,width=20,height=20):
        self.maps = [(70,50),(400,35),(86,266)]
        #self.map_volume = CNP.map_to_volume_dict(map_x,map_y,width,height)
        self.drop_package_grid_size_by_alt = {1:3,2:5,3:7}
        #self.original_map_volume = copy.deepcopy(self.map_volume)
        self.factor = 5
        #self.local_coordinates = [local_x,local_y]
        #self.world_coordinates = [70,50]
        self.reference_coordinates = [70,50]
        self.actions = list(range(15))
        #self.heading = heading
        #self.altitude = altitude
        self.action_space = spaces.Discrete(15)
        self.real_actions = False
        # put the drone in
        #self.map_volume['vol'][altitude][local_y,local_x] = self.map_volume['feature_value_map']['drone'][altitude]['val']
        #self.map_volume['flat'][local_y,local_x] = self.map_volume['feature_value_map']['drone'][altitude]['val']
        #self.map_volume['img'][local_y,local_x] = self.map_volume['feature_value_map']['drone'][altitude]['color']
        #self.map_volume[altitude]['drone'][local_y, local_x] = 1.0
        #put the hiker in@ altitude 0
        #self.map_volume['vol'][0][hiker_y,hiker_x] = self.map_volume['feature_value_map']['hiker']['val']
        #self.map_volume['flat'][hiker_y,hiker_x] = self.map_volume['feature_value_map']['hiker']['val']
        #self.map_volume['img'][hiker_y,hiker_x] = self.map_volume['feature_value_map']['hiker']['color']
        #self.hiker_position = np.where(self.map_volume['vol'] == self.map_volume['feature_value_map']['hiker']['val'])
        #self.map_volume[0]['hiker'][hiker_y,hiker_x] = 1.0

        if self.real_actions:
            self.mavsimhandler = MavsimHandler()
            stateThread = threading.Thread(target=self.mavsimhandler.read_state)
            stateThread.start()
            time.sleep(0.4)

        #5x5 plane descriptions
        self.planes = {}
        self.planes[1] = [[(0, 2), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4)], np.zeros((5, 5, 3))]
        self.planes[2] = [[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (0, 4), (1, 3), (2, 3), (1, 2)], np.zeros((5, 5, 3))]
        self.planes[3] = [[(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (1, 3), (2, 3), (3, 3), (2, 4)], np.zeros((5, 5, 3))]
        self.planes[4] = [[(0,4),(1,3),(2,3),(3,3),(4,4),(2,2),(3,2),(3,1),(4,0)],np.zeros((5,5,3))]
        self.planes[5] = [[(2,0),(2,1),(2,2),(2,3),(2,4),(3,1),(3,2),(3,3),(4,2)],np.zeros((5,5,3))]
        self.planes[6] = [[(0,0),(1,1),(2,2),(3,3),(4,4),(2,1),(3,1),(3,2),(4,0)],np.zeros((5,5,3))]
        self.planes[7] = [[(2,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(4,2)],np.zeros((5,5,3))]
        self.planes[8] = [[(0,0),(4,0),(1,1),(2,1),(3,1),(1,2),(2,2),(1,3),(0,4)],np.zeros((5,5,3))]


        self.hikers = {}
        self.hikers[0] = [[(0,2),(1,2),(2,2),(3,2),(4,2),(2,0),(2,1),(2,2),(2,3),(2,4)],np.zeros((5,5,3))]
        self.hiker_image = np.zeros((5,5,3))
        #self.hiker_image[:,:,:] = self.map_volume['feature_value_map']['hiker']['color']

        self.drop_probabilities = {"damage_probability": {0: 0.00, 1: 0.01, 2: 0.40, 3: 0.80},
        "stuck_probability": {"pine trees": 0.50, "pine tree": 0.25, "cabin": 0.50, "flight tower": 0.15, "firewatch tower": 0.20},
        "sunk_probability": {"water": 0.50}
        }
        self.drop_rewards = {"OK": 10,
        "OK_STUCK": 5,
        "OK_SUNK": 5,
        "DAMAGED": -10,
        "DAMAGED_STUCK": -15,
        "DAMAGED_SUNK": -15,
        "CRASHED": -30
        }


        self.actionvalue_heading_action = {
            0: {1:'self.take_action(delta_alt=-1,delta_x=-1,delta_y=0,new_heading=7)',
                2:'self.take_action(delta_alt=-1,delta_x=-1,delta_y=-1,new_heading=8)',
                3:'self.take_action(delta_alt=-1,delta_x=0,delta_y=-1,new_heading=1)',
                4:'self.take_action(delta_alt=-1,delta_x=1,delta_y=-1,new_heading=2)',
                5:'self.take_action(delta_alt=-1,delta_x=1,delta_y=0,new_heading=3)',
                6:'self.take_action(delta_alt=-1,delta_x=1,delta_y=1,new_heading=4)',
                7:'self.take_action(delta_alt=-1,delta_x=0,delta_y=1,new_heading=5)',
                8:'self.take_action(delta_alt=-1,delta_x=-1,delta_y=1,new_heading=6)'},
            1: {1:'self.take_action(delta_alt=-1,delta_x=-1,delta_y=-1,new_heading=8)',
                2:'self.take_action(delta_alt=-1,delta_x=0,delta_y=-1,new_heading=1)',
                3:'self.take_action(delta_alt=-1,delta_x=1,delta_y=-1,new_heading=2)',
                4:'self.take_action(delta_alt=-1,delta_x=1,delta_y=0,new_heading=3)',
                5:'self.take_action(delta_alt=-1,delta_x=1,delta_y=1,new_heading=4)',
                6:'self.take_action(delta_alt=-1,delta_x=0,delta_y=1,new_heading=5)',
                7:'self.take_action(delta_alt=-1,delta_x=-1,delta_y=1,new_heading=6)',
                8:'self.take_action(delta_alt=-1,delta_x=-1,delta_y=0,new_heading=7)'},
            2: {1:'self.take_action(delta_alt=-1,delta_x=0,delta_y=-1,new_heading=1)',
                2:'self.take_action(delta_alt=-1,delta_x=1,delta_y=-1,new_heading=2)',
                3:'self.take_action(delta_alt=-1,delta_x=1,delta_y=0,new_heading=3)',
                4:'self.take_action(delta_alt=-1,delta_x=1,delta_y=1,new_heading=4)',
                5:'self.take_action(delta_alt=-1,delta_x=0,delta_y=1,new_heading=5)',
                6:'self.take_action(delta_alt=-1,delta_x=-1,delta_y=1,new_heading=6)',
                7:'self.take_action(delta_alt=-1,delta_x=-1,delta_y=0,new_heading=7)',
                8:'self.take_action(delta_alt=-1,delta_x=-1,delta_y=-1,new_heading=8)'},
            3: {1:'self.take_action(delta_alt=-1,delta_x=1,delta_y=-1,new_heading=2)',
                2:'self.take_action(delta_alt=-1,delta_x=1,delta_y=0,new_heading=3)',
                3:'self.take_action(delta_alt=-1,delta_x=1,delta_y=1,new_heading=4)',
                4:'self.take_action(delta_alt=-1,delta_x=0,delta_y=1,new_heading=5)',
                5:'self.take_action(delta_alt=-1,delta_x=-1,delta_y=1,new_heading=6)',
                6:'self.take_action(delta_alt=-1,delta_x=-1,delta_y=0,new_heading=7)',
                7:'self.take_action(delta_alt=-1,delta_x=-1,delta_y=-1,new_heading=8)',
                8:'self.take_action(delta_alt=-1,delta_x=-0,delta_y=-1,new_heading=1)'},
            4: {1:'self.take_action(delta_alt=-1,delta_x=1,delta_y=0,new_heading=3)',
                2:'self.take_action(delta_alt=-1,delta_x=1,delta_y=1,new_heading=4)',
                3:'self.take_action(delta_alt=-1,delta_x=0,delta_y=1,new_heading=5)',
                4:'self.take_action(delta_alt=-1,delta_x=-1,delta_y=1,new_heading=6)',
                5:'self.take_action(delta_alt=-1,delta_x=-1,delta_y=0,new_heading=7)',
                6:'self.take_action(delta_alt=-1,delta_x=-1,delta_y=-1,new_heading=8)',
                7:'self.take_action(delta_alt=-1,delta_x=0,delta_y=-1,new_heading=1)',
                8:'self.take_action(delta_alt=-1,delta_x=1,delta_y=-1,new_heading=2)'},
            5: {1:'self.take_action(delta_alt=0, delta_x=-1, delta_y=0, new_heading=7)',
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
                 8: 'self.drop_package()',}
        


        }


        print("here")


        # self.action_map = {
        #     0: self.take_action(self.heading,self.altitude,-1,self.heading-1,self.heading-2), #turn left, down 1
        #
        # }

    # def __init__(self):
    #     self.actions = list(range(15))
    #     self.inv_actions = [0, 2, 1, 4, 3]
    #     self.action_space = spaces.Discrete(15)
    #     self.action_pos_dict = {0: [0,0], 1:[-1, 0], 2:[1,0], 3:[0,-1], 4:[0,1]}
    #
    #     ''' set observation space '''
    #     self.obs_shape = [128, 128, 3]  # observation space shape
    #     self.observation_space = spaces.Box(low=0, high=1, shape=self.obs_shape)
    #
    #     ''' initialize system state '''
    #     this_file_path = os.path.dirname(os.path.realpath(__file__))
    #     self.grid_map_path = os.path.join(this_file_path, 'plan5.txt')
    #     self.start_grid_map = self._read_grid_map(self.grid_map_path) # initial grid map
    #     self.current_grid_map = copy.deepcopy(self.start_grid_map)  # current grid map
    #     self.observation = self._gridmap_to_observation(self.start_grid_map)
    #     self.grid_map_shape = self.start_grid_map.shape
    #
    #     ''' agent state: start, target, current state '''
    #     self.agent_start_state, _ = self._get_agent_start_target_state(
    #                                 self.start_grid_map)
    #     _, self.agent_target_state = self._get_agent_start_target_state(
    #                                 self.start_grid_map)
    #     self.agent_state = copy.deepcopy(self.agent_start_state)
    #
    #     ''' set other parameters '''
    #     self.restart_once_done = False  # restart or not once done
    #     self.verbose = False # to show the environment or not
    #
    #     GridworldEnv.num_env += 1
    #     self.this_fig_num = GridworldEnv.num_env
    #     if self.verbose == True:
    #         self.fig = plt.figure(self.this_fig_num)
    #         plt.show(block=False)
    #         plt.axis('off')
    #         self._render()

    def neighbors(self,arr, x, y, N):
        #https://stackoverflow.com/questions/32604856/slicing-outside-numpy-array
        #new_arr = np.zeros((N,N))

        # Ap = np.lib.pad(arr.astype(int),1, 'constant',constant_values=(np.nan,np.nan))
        # nx = np.arange(N) + x
        # ny = np.arange(N) + y
        # Acut = Ap[np.ix_(np.arange(N) + x, np.arange(N) + y)]
        # Acut[np.isnan(Acut)] = np.nanmean(Acut)
        #
        # return Acut
        # minx = min(x - N,0)
        # miny = min(0,y - N)
        # maxx = max()


        # print(arr[x-N//2:x+N//2,y-N//2:y+N//2])
        # return arr[x-N//2:x+N//2,y-N//2:y+N//2]


        left = max(0, x - N//2)
        right = min(arr.shape[0], x + N//2)
        top = max(0, y - N//2)
        bottom = min(arr.shape[1], y + N//2)

        window = arr[left:right + 1, top:bottom + 1]
        return window

        # fillval = window.mean()
        #
        # result = np.empty((2 * N + 1, 2 * N + 1))
        # result[:] = fillval
        #
        # ll = N - x
        # tt = N - y
        # result[ll + left:ll + right + 1, tt + top:tt + bottom + 1] = window
        #
        # return result

    def position_value(self, terrain, altitude, reward_dict, probability_dict):
        damage_probability = probability_dict['damage_probability'][altitude]
        if terrain in probability_dict['stuck_probability'].keys():
            stuck_probability = probability_dict['stuck_probability'][terrain]
        else:
            stuck_probability = 0.0
        if terrain in probability_dict['sunk_probability'].keys():
            sunk_probability = probability_dict['sunk_probability'][terrain]
        else:
            sunk_probability = 0.0
        damaged = np.random.random() < damage_probability
        stuck = np.random.random() < stuck_probability
        sunk = np.random.random() < sunk_probability
        package_state = 'DAMAGED' if damaged else 'OK'
        package_state += '_STUCK' if stuck else ''
        package_state += '_SUNK' if sunk else ''
        print("Package state:", package_state)
        reward = reward_dict[package_state]
        return reward

    def drop_package(self):
        #value = -5
        #while value < 0 and value < self.original_map_volume['vol'][0].shape[0]:

        alt = self.altitude
        drone_position =  np.where(self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
        hiker_position = self.hiker_position
        region = self.drop_package_grid_size_by_alt[self.altitude]
        neighbors = self.neighbors(self.original_map_volume['vol'][0],int(drone_position[1]),int(drone_position[2]),region)
        print("neigh:")
        print(neighbors)
        x = np.random.randint(0,neighbors.shape[0])
        y = np.random.randint(0,neighbors.shape[1])
        print(x,y)
        value = neighbors[x,y]
        terrain = self.original_map_volume['value_feature_map'][value]['feature']
        reward = self.position_value(terrain, alt, self.drop_rewards, self.drop_probabilities)
        print(terrain, reward)




    def take_action(self,delta_alt=0,delta_x=0,delta_y=0,new_heading=1):
        #print("stop")
        vol_shape = self.map_volume['vol'].shape

        local_coordinates = np.where(self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
        if int(local_coordinates[1]) + delta_y < 0 or  \
            int(local_coordinates[2]) + delta_x < 0 or \
            int(local_coordinates[1] + delta_y > vol_shape[1] - 1) or \
            int(local_coordinates[2] + delta_x > vol_shape[2] - 1):

            return 0
        #todo update with shape below
        forbidden = [(0,0),(vol_shape[1]-1,0),
                     (vol_shape[1]-1,vol_shape[1]-1),(0,vol_shape[1]-1)]
        print((int(local_coordinates[1]) + delta_y, int(local_coordinates[2]) + delta_x), forbidden)
        if (int(local_coordinates[1]) + delta_y, int(local_coordinates[2]) + delta_x) in forbidden:
            return 0


        new_alt = self.altitude + delta_alt if self.altitude + delta_alt < 4 else 3
        print("new_alt", new_alt)
        if new_alt < 0:
            return 0


        #put back the original
        self.map_volume['vol'][self.altitude][local_coordinates[1],local_coordinates[2]] = float(self.original_map_volume['vol'][local_coordinates])

        self.map_volume['flat'][local_coordinates[1],local_coordinates[2]] = float(self.original_map_volume['flat'][local_coordinates[1],local_coordinates[2]])
        #self.map_volume['img'][local_coordinates[1],local_coordinates[2]] = self.original_map_volume['img'][local_coordinates[1],local_coordinates[2]]
        # put the hiker back
        self.map_volume['vol'][self.hiker_position] = self.map_volume['feature_value_map']['hiker']['val']
        self.map_volume['flat'][self.hiker_position[1],self.hiker_position[2]] = self.map_volume['feature_value_map']['hiker']['val']
        #self.map_volume['img'][self.hiker_position[1],self.hiker_position[2]] = self.map_volume['feature_value_map']['hiker']['color']
        #put the drone in
        self.map_volume['flat'][local_coordinates[1]+delta_y,local_coordinates[2]+delta_x] = self.map_volume['feature_value_map']['drone'][new_alt]['val']
        self.map_volume['vol'][new_alt][local_coordinates[1] + delta_y, local_coordinates[2] + delta_x] = self.map_volume['feature_value_map']['drone'][new_alt]['val']
        #self.map_volume['img'][local_coordinates[1] + delta_y, local_coordinates[2] + delta_x] = self.map_volume['feature_value_map']['drone'][new_alt]['color']
        # for i in range(4,-1,-1):
        #     if self.map_volume['vol'][i][local_coordinates[1],local_coordinates[2]]:
        #         self.map_volume['flat'][int(local_coordinates[1]),int(local_coordinates[2])] = float(self.map_volume['vol'][i][int(local_coordinates[1]),int(local_coordinates[2])])
        #         break
        self.altitude = new_alt
        self.heading = new_heading


        if self.real_actions:
            drone_position = np.where(self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
            coordinates = [self.reference_coordinates[0] + int(drone_position[1]),
                           self.reference_coordinates[1] + int(drone_position[2])]
            print("drone new position", drone_position)
            print("new altitude", self.altitude)
            print("sending coordinates", coordinates)
            #assume the drone is the right spot, right heading
            success = self.mavsimhandler.head_to(new_heading,self.altitude)
            #success = self.mavsimhandler.fly_path(coordinates=coordinates,altitude=self.altitude)


        return 1

    # def take_action(self,delta_alt=0,delta_x=0,delta_y=0,new_heading=1):
    #     #print("take action called",delta_alt,delta_x,delta_y,new_heading)
    #     local_coordinates = self.map_volume[self.altitude]['drone'].nonzero()
    #     if int(local_coordinates[0]) + delta_y < 0 or  \
    #         int(local_coordinates[1]) + delta_x < 0 or \
    #         int(local_coordinates[0] + delta_y > 19) or \
    #         int(local_coordinates[1] + delta_x > 19):
    #         #print('take_action returning 0')
    #         return 0
    #     #print("this happened")
    #     new_alt = self.altitude + delta_alt if self.altitude + delta_alt < 4 else 3
    #     self.map_volume[self.altitude]['drone'][local_coordinates[0],local_coordinates[1]] = 0.0
    #     self.map_volume[new_alt]['drone'][local_coordinates[0]+delta_y,local_coordinates[1]+delta_x] = 1.0
    #     self.altitude = new_alt
    #     self.heading = new_heading
    #     return 1
    def available_action(self,action):
        drone_position = np.where(self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
        vol_shape = self.map_volume['vol'].shape



    def check_for_hiker(self):
        drone_position = np.where(self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
        #hiker_position = np.where(self.map_volume['vol'] == self.map_volume['feature_value_map']['hiker'][0])
        #print("drone",drone_position)
        #print("hiker",self.hiker_position)
        if (drone_position[1],drone_position[2]) == (self.hiker_position[1],self.hiker_position[2]):
            return 1
        return 0
        #return int(self.map_volume[0]['hiker'][int(local_coordinates[0]),int(local_coordinates[1])])


    def check_for_crash(self):
        #if drone on altitude 0, crash
        if self.altitude == 0:
            return 1

        # if len(self.map_volume[0]['drone'].nonzero()[0]):
        #     return 1
        #at any other altutidue, check for an object at the drone's position
        drone_position = np.where(self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
        return int(self.original_map_volume['vol'][drone_position])
        #drone_position = self.map_volume[self.altitude]['drone'].nonzero()
        # for i in range(self.altitude,4):
        #
        #     for key in self.map_volume[i]:
        #         if key == 'drone' or key == 'map':
        #             continue
        #         #just check if drone position is returns a non-zero
        #         if self.map_volume[i][key][int(drone_position[0]),int(drone_position[1])]:
        #             return 1
        # return 0




    def step(self, action):
        ''' return next observation, reward, finished, success '''

        action = int(action)
        x = eval(self.actionvalue_heading_action[action][self.heading])
        crash = self.check_for_crash()

        #return (self.map_volume, 0, True, crash)
        self._render()


        return 0



        # action = int(action)
        # info = {}
        # info['success'] = False
        # nxt_agent_state = (self.agent_state[0] + self.action_pos_dict[action][0],
        #                     self.agent_state[1] + self.action_pos_dict[action][1])
        # if action == 0: # stay in place
        #     info['success'] = True
        #     return (self.observation, 0, False, info)
        # if nxt_agent_state[0] < 0 or nxt_agent_state[0] >= self.grid_map_shape[0]:
        #     info['success'] = False
        #     return (self.observation, 0, False, info)
        # if nxt_agent_state[1] < 0 or nxt_agent_state[1] >= self.grid_map_shape[1]:
        #     info['success'] = False
        #     return (self.observation, 0, False, info)
        # # successful behavior
        # org_color = self.current_grid_map[self.agent_state[0], self.agent_state[1]]
        # new_color = self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]]
        # if new_color == 0:
        #     if org_color == 4:
        #         self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 0
        #         self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = 4
        #     elif org_color == 6 or org_color == 7:
        #         self.current_grid_map[self.agent_state[0], self.agent_state[1]] = org_color-4
        #         self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = 4
        #     self.agent_state = copy.deepcopy(nxt_agent_state)
        # elif new_color == 1: # gray
        #     info['success'] = False
        #     return (self.observation, 0, False, info)
        # elif new_color == 2 or new_color == 3:
        #     self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 0
        #     self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = new_color+4
        #     self.agent_state = copy.deepcopy(nxt_agent_state)
        # self.observation = self._gridmap_to_observation(self.current_grid_map)
        # self._render()
        # if nxt_agent_state[0] == self.agent_target_state[0] and nxt_agent_state[1] == self.agent_target_state[1] :
        #     target_observation = copy.deepcopy(self.observation)
        #     if self.restart_once_done:
        #         self.observation = self._reset()
        #         info['success'] = True
        #         return (self.observation, 1, True, info)
        #     else:
        #         info['success'] = True
        #         return (target_observation, 1, True, info)
        # else:
        #     info['success'] = True
        #     return (self.observation, 0, False, info)



    def reset(self):

        self.heading = random.randint(1,8)
        self.altitude = 3
        _map = random.choice(self.maps)
        self.map_volume = CNP.map_to_volume_dict(_map[0], _map[1], 10, 10)
        hiker = (random.randint(2,self.map_volume['vol'].shape[1]-1),random.randint(2,self.map_volume['vol'].shape[1]-2))
        drone = (random.randint(2,self.map_volume['vol'].shape[1]-1),random.randint(2,self.map_volume['vol'].shape[1]-2))
        while drone == hiker:
            drone = (random.randint(2, self.map_volume['vol'].shape[1]-1), random.randint(2, self.map_volume['vol'].shape[1]-2))



        self.original_map_volume = copy.deepcopy(self.map_volume)

        # self.local_coordinates = [local_x,local_y]
        # self.world_coordinates = [70,50]
        self.reference_coordinates = [_map[0], _map[1]]
        self.actions = list(range(15))


        self.action_space = spaces.Discrete(15)
        #self.real_actions = False
        # put the drone in
        self.map_volume['vol'][self.altitude][drone[0], drone[1]] = self.map_volume['feature_value_map']['drone'][self.altitude]['val']
        self.map_volume['flat'][drone[0], drone[1]] = self.map_volume['feature_value_map']['drone'][self.altitude]['val']
        self.map_volume['img'][drone[0], drone[1]] = self.map_volume['feature_value_map']['drone'][self.altitude]['color']
        # self.map_volume[altitude]['drone'][local_y, local_x] = 1.0
        # put the hiker in@ altitude 0
        self.map_volume['vol'][0][hiker[0], hiker[1]] = self.map_volume['feature_value_map']['hiker']['val']
        self.map_volume['flat'][hiker[0], hiker[1]] = self.map_volume['feature_value_map']['hiker']['val']
        self.map_volume['img'][hiker[0], hiker[1]] = self.map_volume['feature_value_map']['hiker']['color']
        self.hiker_position = np.where(self.map_volume['vol'] == self.map_volume['feature_value_map']['hiker']['val'])

        observation = self.generate_observation()
        return observation



    def _reset(self):
        self.agent_state = copy.deepcopy(self.agent_start_state)
        self.current_grid_map = copy.deepcopy(self.start_grid_map)
        self.observation = self._gridmap_to_observation(self.start_grid_map)
        self._render()
        return self.observation

    def _read_grid_map(self, grid_map_path):
        grid_map = open(grid_map_path, 'r').readlines()
        grid_map_array = []
        for k1 in grid_map:
            k1s = k1.split(' ')
            tmp_arr = []
            for k2 in k1s:
                try:
                    tmp_arr.append(int(k2))
                except:
                    pass
            grid_map_array.append(tmp_arr)
        grid_map_array = np.array(grid_map_array)
        return grid_map_array

    def _get_agent_start_target_state(self, start_grid_map):
        start_state = None
        target_state = None
        for i in range(start_grid_map.shape[0]):
            for j in range(start_grid_map.shape[1]):
                this_value = start_grid_map[i,j]
                if this_value == 4:
                    start_state = [i,j]
                if this_value == 3:
                    target_state = [i,j]
        if start_state is None or target_state is None:
            sys.exit('Start or target state not specified')
        return start_state, target_state

    def _gridmap_to_observation(self, grid_map, obs_shape=None):
        if obs_shape is None:
            obs_shape = self.obs_shape
        observation = np.random.randn(*obs_shape)*0.0
        gs0 = int(observation.shape[0]/grid_map.shape[0])
        gs1 = int(observation.shape[1]/grid_map.shape[1])
        for i in range(grid_map.shape[0]):
            for j in range(grid_map.shape[1]):
                for k in range(3):
                    this_value = COLORS[grid_map[i,j]][k]
                    observation[i*gs0:(i+1)*gs0, j*gs1:(j+1)*gs1, k] = this_value
        return observation

    def plane_image(self,heading, color):
        '''Returns a 5x5 image as np array'''
        for point in self.planes[heading][0]:
            self.planes[heading][1][point[0], point[1]] = color
        return self.planes[heading][1]

    def generate_observation(self):
        map = self.original_map_volume['img']
        map = imresize(map, self.factor * 100, interp='nearest') #resize by factor of 5
        #add the hiker
        hiker_position = (int(self.hiker_position[1]* 5), int(self.hiker_position[2]) * 5)
        #map[hiker_position[0]:hiker_position[0]+5,hiker_position[1]:hiker_position[1]+5,:] = self.hiker_image
        for point in self.hikers[0][0]:
            map[hiker_position[0]+point[0],hiker_position[1]+point[1],:] = self.map_volume['feature_value_map']['hiker']['color']
        #add the drone
        drone_position = np.where(self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
        drone_position = (int(drone_position[1]) * 5, int(drone_position[2]) * 5)
        for point in self.planes[self.heading][0]:
            map[drone_position[0] + point[0],drone_position[1] + point[1],:] = self.map_volume['feature_value_map']['drone'][self.altitude]['color']
        #map[drone_position[0]:drone_position[0] + 5,drone_position[1]:drone_position[1] + 5] = self.plane_image(self.heading,self.map_volume['feature_value_map']['drone'][self.altitude]['color'])

        #map = imresize(map, (1000,1000), interp='nearest')
        return map




    def _render(self, mode='human', close=False):
        #return
        #if self.verbose == False:
        #    return
        #img = self.observation
        #map = self.original_map_volume['img']
        map = self.generate_observation()
        #map = self.map_volume['flat'] / self.altitude
        #fig = plt.figure(self.this_fig_num)
        #img = np.zeros((20,20,3))
        #img[10,10,0] = 200
        #img[10,10,1] = 153
        #img[10,10,2] = 255
        #planes should be self.planes, in iinit
        #
        #drone_position = np.where(self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])
        #drone_position = (int(drone_position[1])*5,int(drone_position[2])*5)

        #map = imresize(map,(100,100),interp='nearest')
        #map[drone_position[0]:drone_position[0] + 5,drone_position[1]:drone_position[1] + 5] = self.plane_image(self.heading,self.map_volume['feature_value_map']['drone'][self.altitude]['color'])
        # handles = []
        # for label in self.map_volume['feature_value_map']:
        #     color = [255,255,255]
        #     if 'color' in self.map_volume['feature_value_map'][label]:
        #         color = self.map_volume['feature_value_map'][label]
        #     patch = matplotlib.patches.Patch(color=color, label=label)
        #     handles.append(patch)

        #map = imresize(map,(50,50),interp='nearest')

        fig = plt.figure(0)
        plt.clf()
        plt.imshow(map,vmax=9)
        fig.canvas.draw()
        plt.pause(0.00001)
        return 
 
    def change_start_state(self, sp):
        ''' change agent start state '''
        ''' Input: sp: new start state '''
        if self.agent_start_state[0] == sp[0] and self.agent_start_state[1] == sp[1]:
            _ = self._reset()
            return True
        elif self.start_grid_map[sp[0], sp[1]] != 0:
            return False
        else:
            s_pos = copy.deepcopy(self.agent_start_state)
            self.start_grid_map[s_pos[0], s_pos[1]] = 0
            self.start_grid_map[sp[0], sp[1]] = 4
            self.current_grid_map = copy.deepcopy(self.start_grid_map)
            self.agent_start_state = [sp[0], sp[1]]
            self.observation = self._gridmap_to_observation(self.current_grid_map)
            self.agent_state = copy.deepcopy(self.agent_start_state)
            self._reset()
            self._render()
        return True
        
    
    def change_target_state(self, tg):
        if self.agent_target_state[0] == tg[0] and self.agent_target_state[1] == tg[1]:
            _ = self._reset()
            return True
        elif self.start_grid_map[tg[0], tg[1]] != 0:
            return False
        else:
            t_pos = copy.deepcopy(self.agent_target_state)
            self.start_grid_map[t_pos[0], t_pos[1]] = 0
            self.start_grid_map[tg[0], tg[1]] = 3
            self.current_grid_map = copy.deepcopy(self.start_grid_map)
            self.agent_target_state = [tg[0], tg[1]]
            self.observation = self._gridmap_to_observation(self.current_grid_map)
            self.agent_state = copy.deepcopy(self.agent_start_state)
            self._reset()
            self._render()
        return True
    
    def get_agent_state(self):
        ''' get current agent state '''
        return self.agent_state

    def get_start_state(self):
        ''' get current start state '''
        return self.agent_start_state

    def get_target_state(self):
        ''' get current target state '''
        return self.agent_target_state

    def _jump_to_state(self, to_state):
        ''' move agent to another state '''
        info = {}
        info['success'] = True
        if self.current_grid_map[to_state[0], to_state[1]] == 0:
            if self.current_grid_map[self.agent_state[0], self.agent_state[1]] == 4:
                self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 0
                self.current_grid_map[to_state[0], to_state[1]] = 4
                self.observation = self._gridmap_to_observation(self.current_grid_map)
                self.agent_state = [to_state[0], to_state[1]]
                self._render()
                return (self.observation, 0, False, info)
            if self.current_grid_map[self.agent_state[0], self.agent_state[1]] == 6:
                self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 2
                self.current_grid_map[to_state[0], to_state[1]] = 4
                self.observation = self._gridmap_to_observation(self.current_grid_map)
                self.agent_state = [to_state[0], to_state[1]]
                self._render()
                return (self.observation, 0, False, info)
            if self.current_grid_map[self.agent_state[0], self.agent_state[1]] == 7:  
                self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 3
                self.current_grid_map[to_state[0], to_state[1]] = 4
                self.observation = self._gridmap_to_observation(self.current_grid_map)
                self.agent_state = [to_state[0], to_state[1]]
                self._render()
                return (self.observation, 0, False, info)
        elif self.current_grid_map[to_state[0], to_state[1]] == 4:
            return (self.observation, 0, False, info)
        elif self.current_grid_map[to_state[0], to_state[1]] == 1:
            info['success'] = False
            return (self.observation, 0, False, info)
        elif self.current_grid_map[to_state[0], to_state[1]] == 3:
            self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 0
            self.current_grid_map[to_state[0], to_state[1]] = 7
            self.agent_state = [to_state[0], to_state[1]]
            self.observation = self._gridmap_to_observation(self.current_grid_map)
            self._render()
            if self.restart_once_done:
                self.observation = self._reset()
                return (self.observation, 1, True, info)
            return (self.observation, 1, True, info)
        else:
            info['success'] = False
            return (self.observation, 0, False, info)

    def _close_env(self):
        plt.close(1)
        return
    
    def jump_to_state(self, to_state):
        a, b, c, d = self._jump_to_state(to_state)
        return (a, b, c, d) 

#sample code
a = GridworldEnv(map_x=70,map_y=50,local_x=2,local_y=2,hiker_x=10,heading=1,altitude=3)
a.reset()
#a.step(7)
# #a.step(12)
# #
# #def show_img():
# now = datetime.datetime.now()
# for i in range(10000):
#     a.step(random.randint(1,14))
#     #local_coordinates = a.map_volume[a.altitude]['drone'].nonzero()
#     #print("coordinates", local_coordinates, a.heading)
#     if a.check_for_crash():
#         print("crash at altitude", a.altitude)
#         a.reset()
#         #time.sleep(0.5)
#     if a.check_for_hiker():
#         print("hiker after", i)
#         a.reset()

a.step(15)
#print(a.check_for_crash())
print('complete')#, (datetime.datetime.now().second - now.second))