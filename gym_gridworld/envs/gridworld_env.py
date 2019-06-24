import gym
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

from scipy.misc import imresize
import matplotlib.pyplot as plt

from gym_gridworld.envs import create_np_map as CNP

from gym_gridworld.envs.mavsim_udp_server import MavsimUDPHandler

# define colors
# 0: black; 1 : gray; 2 : blue; 3 : green; 4 : red
COLORS = {0: [0.0, 0.0, 0.0], 1: [0.5, 0.5, 0.5], \
          2: [0.0, 0.0, 1.0], 3: [0.0, 1.0, 0.0], \
          4: [1.0, 0.0, 0.0], 6: [1.0, 0.0, 1.0], \
          7: [1.0, 1.0, 0.0]}


class HeadingEnumeration:

    """Class provides heading translation between integer constants and descriptive strings.
       There is a singleton instance called HEADING that can be used for convenience.

       current_heading = HEADING.EAST

       print( HEADING.to_string(current_heading) )

       --> "East"

       ToDo: Odd that headings are one based instead of zero based ...
       
    """

    def __init__(self):

        self.heading_to_description = {
            1:'North',        # X++
            2:'North East',
            3:'East',         # Y++
            4:'South East',   # X--
            5:'South',        # X--
            6:'South West',
            7:'West',         # Y--
            8:'North West'
        }

        self.heading_to_short_description = {
            1:'N',         # X++
            2:'NE',
            3:'E',         # Y++
            4:'SE',        # X--
            5:'S',         # X--
            6:'SW',
            7:'W',         # Y--
            8:'NW'
        }

        self.NORTH      = 1
        self.NORTH_EAST = 2
        self.EAST       = 3
        self.SOUTH_EAST = 4
        self.SOUTH      = 5
        self.SOUTH_WEST = 6
        self.WEST       = 7
        self.NORTH_WEST = 8

    def to_string(self,heading_int):
        return self.heading_to_description[heading_int]

    def to_short_string(self,heading_int):
        return self.heading_to_short_description[heading_int]

HEADING = HeadingEnumeration()



class ActionEnumeration:

    """Class provides action translation between integer constants and descriptive strings.
       There is a singleton instance called ACTION that can be used for convenience.

       current_action = ACTION.LEVEL_FORWARD

       print( ACTION.to_string(current_action) )

       --> "level forward"

    """

    def __init__(self):

        self.action_to_description = {

            0:'down left 90',
            1:'down left 45',
            2:'down forward',
            3:'down right 45',
            4:'down right 90',

            5:'level left 90',
            6:'level left 45',
            7:'level forward',
            8:'level right 45',
            9:'level right 90',

            10:'up left 90',
            11:'up left 45',
            12:'up forward',
            13:'up right 45',
            14:'up right 90',

            15:'drop'
        }

        self.action_to_short_description = {

            0:'DL90',
            1:'DL45',
            2:'DF',
            3:'DR45',
            4:'DR90',

            5:'LL90',
            6:'LL45',
            7:'LF',
            8:'LR45',
            9:'LR90',

            10:'UL90',
            11:'UL45',
            12:'UF',
            13:'UR45',
            14:'UR90',

            15:'DRP'
        }

        self.DOWN_LEFT_90  =  0
        self.DOWN_LEFT_45  =  1
        self.DOWN_FORWARD  =  2
        self.DOWN_RIGHT_45 =  3
        self.DOWN_RIGHT_90 =  4

        self.LEVEL_LEFT_90    =  5
        self.LEVEL_LEFT_45    =  6
        self.LEVEL_FORWARD    =  7
        self.LEVEL_RIGHT_45   =  8
        self.LEVEL_RIGHT_90   =  9

        self.UP_LEFT_90   = 10
        self.UP_LEFT_45   = 11
        self.UP_FORWARD   = 12
        self.UP_RIGHT_45  = 13
        self.UP_RIGHT_90  = 14

        self.DROP             = 15


    def delta_z(self,action_int):
        
        """Returns an integer in {-1,0,1} corresponding to descending, level or ascending flight respectively"""

        if action_int==self.DROP:
            return 0

        if action_int<5:
            return -1
        else:
            if action_int< 10:
                return 0
            else:
                return 1
            
    def delta_heading(self,action_int):
        
        """Returns an integer in {-2,-1,0,1,2} corresponding to left 90, left 45, forward, right 45 or right 90 turn respectively
           for flight actions and None for other actions such as DROP """
        
        if action_int > 14:
            return None
        else:
            return (action_int % 5) - 2
        
    def new_heading(self, old_heading, action_int):

        dh = self.delta_heading(action_int)

        heading = old_heading + dh
        if heading < 1:
            heading = heading + 8
        if heading > 8:
            heading = heading - 8

        return heading

    def new_altitude(self, old_altitude, action_int):

        dz = self.delta_z(action_int)
        altitude = old_altitude + dz
        if altitude>4:
            altitude = 3
        else:
            if altitude<0:
                altitude = 0

        return altitude

    def to_string(self,action_int):
        return self.action_to_description[action_int]

    def to_short_string(self,action_int):
        return self.action_to_short_description[action_int]

ACTION = ActionEnumeration()


open_map = np.array(

           [ [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [2, 2, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [2, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [2, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [2, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [2, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [2, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
           ])

            
box_canyon_map = np.array(

          [ [  1,  1,  1,  1,  2,  2,  2,  1,  2,  2,  2,  1,  2,  2,  1,  2,  2,  2,  2, 2 ],
            [  1,  1,  1,  1,  2,  2,  2, 25, 25, 25, 25,  2,  2,  2,  1,  2,  2,  1,  2, 2 ],
            [  2,  1,  2,  2,  2,  2, 25, 25, 25, 25, 25, 25, 25, 25,  2,  2,  2,  2,  2, 2 ],
            [  2,  1,  1,  1,  2, 25, 25, 25, 25, 26, 26, 25, 25, 25, 25, 25,  2,  2,  1, 2 ],
            [  2,  2,  1,  2,  2, 25, 25, 25, 25, 26, 26, 26, 25, 25, 25, 25,  2,  2,  1, 2 ],
            [  2,  2,  1,  2,  2, 25, 25, 25, 25,  2, 25, 25, 25, 25, 25, 25,  2,  1,  2, 2 ],
            [  2,  2,  1,  2,  2, 25, 26, 25, 25,  2,  2, 25, 25, 25, 25, 25,  2,  1,  2, 2 ],
            [  2,  2,  1,  2,  2, 25, 26, 25, 25,  2,  2, 25, 26, 25, 25, 25,  2,  2,  1, 2 ],
            [  1,  1,  2,  2,  2, 25, 26, 25, 25,  2,  1, 25, 26, 25, 25, 25,  2,  2,  2, 2 ],
            [  1,  2,  1,  2, 25, 25, 25, 25, 25, 24, 24, 25, 26, 25, 25, 25,  2,  2,  1, 2 ],
            [  2,  2,  1,  2, 25, 25, 25, 25, 25, 24, 24, 25, 26, 25, 25, 25,  2,  1,  1, 2 ],
            [  2,  2,  1,  2, 25, 25, 25, 25, 25, 24, 24, 25, 26, 25, 25, 25,  2,  1,  2, 2 ],
            [  1,  1,  1,  2,  2, 25, 25, 25, 25,  2,  2, 25, 25, 25, 25, 25,  2,  2,  2, 2 ],
            [  1,  1,  2,  2,  2, 25, 25, 25, 25,  2,  2, 25, 25, 25, 25, 25,  2,  2,  2, 2 ],
            [  2,  1,  1,  2,  2,  2, 25, 25, 25,  2,  2, 25, 25, 25, 25, 25,  2,  2,  1, 2 ],
            [  2,  2,  2,  2, 22, 22, 22, 22, 22,  2,  2,  2, 25, 25, 25,  2,  2,  2,  2, 2 ],
            [  2,  1,  1,  2,  2,  2,  2,  2,  2,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2, 2 ],
            [  2,  2,  2,  1,  1,  1,  2,  2,  2,  1,  1,  1,  2,  2,  1,  1,  2,  1,  2, 2 ],
            [  2,  2,  2,  2,  2,  1,  1,  1,  1,  2,  2,  1,  1,  2,  2,  2,  2,  2,  2, 2 ],
            [  2,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  1,  1, 2 ]
          ])



class GridworldEnv(gym.Env):

    metadata = {'render.modes': ['human']}
    num_envs = 1

    def __init__(self,
                 hiker_initial_position=None,
                 drone_initial_position=None, drone_initial_altitude=None, drone_initial_heading=None,
                 timestep_limit=1,
                 width=20, height=20,
                 use_mavsim=False,
                 verbose=False,
                 curriculum_radius=None,
                 goal_mode='drop',
                 episode_length=None):


        self.hiker_initial_position=hiker_initial_position
        self.drone_initial_position=drone_initial_position
        self.drone_initial_altitude=drone_initial_altitude
        self.drone_initial_heading=drone_initial_heading
        self.curriculum_radius=curriculum_radius
        self.goal_mode = goal_mode
        self.episode_length=episode_length

        # # TODO: Pass the environment with arguments

        #num_alts = 4
        self.verbose = verbose # to show the environment or not
        self.dropping = True # This is for the reset to select the proper starting locations for hiker and drone
        self.restart_once_done = True  # restart or not once done
        self.drop = False
        self.countdrop = 0
        self.no_action_flag = False
        self.submap_offsets =[(265, 308), (20, 94), (146, 456), (149, 341), (164, 90), (167, 174),
                              (224,153), (241,163), (260,241), (265,311), (291,231),
                              (308,110), (334,203), (360,112), (385,291), (330,352), (321,337)]#[(400,35), (350,90), (430,110),(390,50), (230,70)] #[(86, 266)] (70,50) # For testing, 70,50 there is no where to drop in the whole map
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
        self.observation_space = spaces.Box(low=0, high=255, shape=self.obs_shape)
        self.use_mavsim_simulator = use_mavsim
        self.crash = 0
        self.package_dropped = 0
        self.package_position = ()
        # self._max_episode_steps = 10 # Max timesteps

        self.tile_ids_compatible_with_hiker = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 18, 21, 22]
        self.tile_ids_compatible_with_drone_altitude = {
            1: [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 18, 21, 22],
            2: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 18, 21, 22, 19, 24, 30, 32],
            3: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 18, 21, 22, 19, 24, 30, 32, 13, 17, 19, 25, 28]
        }

        if self.use_mavsim_simulator:
            self.mavsimhandler = MavsimUDPHandler()

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
                                   "stuck_probability": {"pine trees": 0.50,
                                                         "pine tree": 0.25,
                                                         "cabin": 0.50,
                                                         "flight tower": 0.15,
                                                         "firewatch tower": 0.20},
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

        # Given <ActionID> and <previousHeadingID>, move agent and set new heading

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


        print("GridWorldEnv initialization complete")



    def neighbors(self, arr, x, y, N):

        """returns a bounding box [window, left, top,right,bottom] or radius N around point X,Y
           where window is the elements in this box extracted from given array 'arr'.
           Method properly truncates the neighbor window at boundaries of arr."""

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
        x = eval(self.actionvalue_heading_action[7][self.heading])


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

        if self.use_mavsim_simulator:
            drone_position = np.where(
                self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])

            #success = self.mavsimhandler.fly_path(coordinates=[self.reference_coordinates[0] + int(drone_position[1]),
            #                                                   self.reference_coordinates[1] + int(drone_position[2])],
            #                                      altitude=self.altitude)

            self.mavsimhandler.head_to(self.heading,self.altitude)

        return 1


    def check_for_drone_over_hiker(self,drone_position):

        """returns true if drone is over hiker"""

        return (drone_position[1], drone_position[2]) == (self.hiker_position[1], self.hiker_position[2])


    def check_for_crash(self,drone_position):

        """Returns non zero if drone is at altitude zero
           or there is something in the volume at the position of the drone
           as defined by the ORIGINAL map"""

        if self.altitude == 0:
            return 1

        return int(self.original_map_volume['vol'][drone_position])


    def step(self, action):

        ''' return next observation, reward, finished, success '''

        action = int(action)
        info = {}
        info['success'] = False

        done = False

        self.step_number = self.step_number + 1
        if self.episode_length!=None and self.step_number >= self.episode_length:
            done = True
            reward = -1
            info['success'] = False
            info['ex'] = 'Timeout'
            info['Rtime'] = -1

            return (self.generate_observation(), reward, done, info)

        hiker = self.hiker_position

        # observation = self.generate_observation() # You took out this one in Jun 6, 2019 and you substitute the return(observation,...) with self.generate_observation (except the last one which is the return of the whole function)
        drone_position = self.get_drone_position()
        self.dist = np.linalg.norm(np.array(drone_position[-2:]) - np.array(hiker[-2:])) # we remove height from the equation so we avoid going diagonally down

        crash = self.check_for_crash(drone_position)
        info['success'] = not crash

        if crash:
            reward = -1
            info['Rcrash']=-1
            done = True
            print("CRASH")
            info['ex'] = 'crash'
            if self.restart_once_done: # HAVE IT ALWAYS TRUE!!! It learned the first time WITHOUT RESETING FROM CRASH
                return (self.generate_observation(), reward, done, info) # You should get the previous obs so no change here, or return obs=None

        # Do the action (drone is moving). If we crash we dont perform an action so no new observation
        x = eval(self.actionvalue_heading_action[action][self.heading])
        if self.no_action_flag == True:
            reward = self.reward#-1#TODO: it was 0 in all successful training session with PPO. TRY -1 so it avoids dropping at the edges!!!! ahouls be = self.reward and fix self. reward in the drop package function
            done = True
            if self.restart_once_done: # HAVE IT ALWAYS TRUE!!!
                return (self.generate_observation(), reward, done, info)

        if self.goal_mode=='navigate':
            if self.check_for_drone_over_hiker(drone_position):
                done = True
                reward=1
                info['ex']='arrived'
                info['Rhike']=1
                return (self.generate_observation(), reward, done, info)

        # Multiple packages

        if self.goal_mode=='drop' and self.drop:
            self.drop = 0
            reward = self.reward
            info['Rdrop']=reward
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
        self.dist_old = self.dist
        reward = -0.0001#-0.01#(self.alt_rewards[self.altitude]*0.1)*((1/((self.dist**2)+1e-7))) # -0.01 + # The closer we are to the hiker the more important is to be close to its altitude
        #print("scale:",(1/((self.dist**2+1e-7))), "dist=",self.dist+1e-7, "alt=", self.altitude, "drone:",drone, "hiker:", hiker,"found:", self.check_for_hiker())
        info['Rstep']=reward
        return (self.generate_observation(), reward, done, info)


    def reset(self):

        """ goal_mode in { None, 'navigate' } """

        #print("GridworldEnv.reset\n   initial drone position={}, heading={}, altitude={}, ... "
        #      .format(drone_initial_position,drone_initial_heading, drone_initial_altitude))

        #print("   initial hiker position={} )".format( hiker_initial_position))
        #print("   curriculum_radius {}".format(curriculum_radius))

        self.step_number = 0

        self.dist_old = 1000
        self.drop = False
        self.countdrop = 0
        self.no_action_flag = False

        if self.drone_initial_heading==None:
            self.heading = random.randint(1, 8)
        else:
            self.heading=self.drone_initial_heading

        if self.drone_initial_altitude==None:
            self.altitude = random.randint(1,3)
        else:
            self.altitude = self.drone_initial_altitude

        self.reward = 0
        self.crash = 0
        self.package_dropped = 0
        self.package_position = ()

        # self.map_volume = CNP.create_custom_map(box_canyon_map)
        # self.generate_random_map()

        # hiker = (10, 10)
        # drone = (18, 10)
        # self.altitude = 1
        # end DRAWN world

        self.submap_offset = random.choice(self.submap_offsets)

        self.map_volume = CNP.map_to_volume_dict( self.submap_offset[0], self.submap_offset[1], self.mapw, self.maph )
        map_ = self.map_volume['flat']


        # place the hiker

        valid_hiker_drone_pair = False

        while not valid_hiker_drone_pair:

            if self.hiker_initial_position==None:

                # ToDo: investigate this: hiker_safe_points = np.isin(map_, self.tile_ids_compatible_with_hiker)

                hiker_safe_points = []

                for val in self.tile_ids_compatible_with_hiker:

                    where_array = np.where(map_ == val)

                    hiker_safe_points = hiker_safe_points + [ (x, y) for x, y in zip(where_array[0], where_array[1])
                                                              if     x >= 3 and y >= 3
                                                                 and x <= self.map_volume['vol'].shape[1] - 3    # ToDo: BOB - should one of these be shape[2]??
                                                                 and y <= self.map_volume['vol'].shape[1] - 3       ]  #ToDo: Bob -replace with mapw, maph?

                hiker = random.choice(hiker_safe_points)
                # int(self.original_map_volume['vol'][hiker])
            else:
                hiker = self.hiker_initial_position

            if self.drone_initial_position==None:

                drone_safe_points = []
                for val in self.tile_ids_compatible_with_drone_altitude[self.altitude]:

                    where_array = np.where(map_ == val)

                    drone_safe_points = drone_safe_points + [(x, y) for x, y in zip(where_array[0], where_array[1])
                                                              if      x >= 3 and y >= 3
                                                                  and x <= self.map_volume['vol'].shape[1] - 3    # ToDo: BOB - should one of these be shape[2]??
                                                                  and y <= self.map_volume['vol'].shape[1] - 3 ]



                D = distance.cdist([hiker], drone_safe_points, 'chebyshev').astype(int) # Distances from hiker to all drone safe points

                if self.curriculum_radius!=None:
                    close_location_idxs = D[0] < self.curriculum_radius
                    if sum(close_location_idxs)==0:
                        print("WARNING - no safe locations next to hiker location {} -- trying again ".format(hiker))
                    else:
                        valid_hiker_drone_pair=True
                        drone_safe_points = np.array(drone_safe_points)[close_location_idxs,:]
                    #print("Curriculum with radius {}".format(self.curriculum_radius))
                    #print("Satisfying points {}".format(drone_safe_points))

                drone = random.choice(drone_safe_points)
                valid_hiker_drone_pair=True

            else:
                valid_hiker_drone_pair=True
                drone = self.drone_initial_position

            #print("Chose hiker position ",hiker)
            #print("Chose drone position ",drone)
            # print('Distance:',D[0])
            # print('Hiker',hiker)
            # print('safe_drone',drone_safe_points)
            # print('safe_hiker', hiker_safe_points)

            if False:

                k = 50 # k closest. There might be cases in which you have very few drone safe points (e.g. 3) and only one will be really close
                k = 5

                if k> np.array(drone_safe_points).shape[0]:
                    k = np.array(drone_safe_points).shape[0] - 1 # Cauz we index from 0 but shape starts from 1 to max shape
                indx = np.argpartition(D[0],k) # Return the indices of the k closest distances to the hiker. The [0] is VITAL!!!
                # # Use the index to retrieve the k closest safe coords to the hiker
                closest_neighs = np.array(drone_safe_points)[indx[:k]] # You need to have the safe points as array and not list


                # drone = tuple(random.choice(closest_neighs))

                # NOTES: The first element might be the hiker position
                # To move away from hiker increase k and define h=k/2 and discard the h first closest_neighs - 9 suppose to be the max of the closest in an open area. So just use dividends of 9 to discard
                # drone = (hiker[0]-2, hiker[1]-3)
                # drone = random.choice([(hiker[0] - 1, hiker[1] - 1), (hiker[0] - 1, hiker[1] ), (hiker[0], hiker[1] - 1 )])

                # random away location + safe check
                # drone = random.choice([(hiker[0] - 5, hiker[1] - 3), (hiker[0] - 6, hiker[1]), (hiker[0], hiker[1] - 4), (hiker[0] - 6, hiker[1] - 7)])
                # drone = random.choice([(hiker[0] - 8, hiker[1] - 3), (hiker[0] - 10, hiker[1]), (hiker[0], hiker[1] - 9),
                #                        (hiker[0] - 6, hiker[1] - 7)])
                # times = 0
                # while drone not in drone_safe_points:
                #     drone = random.choice([(hiker[0] - 5, hiker[1] - 3), (hiker[0] - 6, hiker[1]), (hiker[0], hiker[1] - 4),
                #                            (hiker[0] - 6, hiker[1] - 7)])
                #     # print('non safe reset drone pos')
                #     if times==10:
                #         print('max reps reached so reset hiker')
                #         hiker = random.choice(hiker_safe_points)
                #         times = 0
                #     times = times + 1

            # all safe points included for final training
            #print("Chose drone location {}".format(drone))
            # drone = (18,18)


############################
        # # Set hiker's and drone's locations
        # #hiker = (random.randint(2, self.map_volume['vol'].shape[1] - 1), random.randint(2, self.map_volume['vol'].shape[1] - 2)) #(8,8) #
        # #if self.dropping:
        # hiker = (10,10)#(random.randint(2, self.map_volume['vol'].shape[1] - 2), random.randint(2, self.map_volume['vol'].shape[1] - 2))  # (7,8) #
        # # drone = (random.randint(2, self.map_volume['vol'].shape[1] - 2), random.randint(2, self.map_volume['vol'].shape[1] - 1))
        # drone = random.choice([(hiker[0]-1, hiker[1]-1),(hiker[0]-1, hiker[1]),(hiker[0], hiker[1]-1)])## Package drop starts close to hiker!!! #(random.randint(2, self.map_volume['vol'].shape[1] - 1), random.randint(2, self.map_volume['vol'].shape[1] - 2)) # (8,8) #
        # drone = random.choice([(hiker[0] - 5, hiker[1] - 7), (hiker[0] - 7, hiker[1]), (hiker[0], hiker[1] - 7)])
        #else:
            # hiker = (random.randint(2, self.map_volume['vol'].shape[1] - 2), random.randint(2, self.map_volume['vol'].shape[1] - 2))  # (7,8) #
            # drone = (random.randint(2, self.map_volume['vol'].shape[1] - 2), random.randint(2, self.map_volume['vol'].shape[1] - 2))

        # while drone == hiker:
        #     print('$$$$$$$$ AWAY !!! $$$$$$$')
        #     drone = (random.randint(2, self.map_volume['vol'].shape[1] - 1),
        #              random.randint(2, self.map_volume['vol'].shape[1] - 2))

        self.original_map_volume = copy.deepcopy(self.map_volume)

        # self.local_coordinates = [local_x,local_y]
        # self.world_coordinates = [70,50]
        self.reference_coordinates = [self.submap_offset[0], self.submap_offset[1]]


        # put the drone in
        self.map_volume['vol'][self.altitude][drone[0], drone[1]] = \
                   self.map_volume['feature_value_map']['drone'][self.altitude]['val']

        # self.map_volume['flat'][drone[0], drone[1]] = self.map_volume['feature_value_map']['drone'][self.altitude][
        #     'val']

        self.map_volume['img'][drone[0], drone[1]] = self.map_volume['feature_value_map']['drone'][self.altitude][
            'color']


        #print("GridworldEnv reset extracted drone position {} (Z,X,Y) ".format(self.get_drone_position()))

        # self.map_volume[altitude]['drone'][local_y, local_x] = 1.0
        # put the hiker in@ altitude 0

        self.map_volume['vol'][0][hiker[0], hiker[1]] = self.map_volume['feature_value_map']['hiker']['val']

        # self.map_volume['flat'][hiker[0], hiker[1]] = self.map_volume['feature_value_map']['hiker']['val']

        self.map_volume['img'][hiker[0], hiker[1]] = self.map_volume['feature_value_map']['hiker']['color']

        # ToDo: BOB This next line seems redundant - why calculate hiker_position from map from hiker?

        self.hiker_position = np.where(self.map_volume['vol'] == self.map_volume['feature_value_map']['hiker']['val'])

        self.image_layers[0] = self.create_image_from_volume(0)
        self.image_layers[1] = self.create_image_from_volume(1)
        self.image_layers[2] = self.create_image_from_volume(2)
        self.image_layers[3] = self.create_image_from_volume(3)
        self.image_layers[4] = self.create_image_from_volume(4)

        observation = self.generate_observation()
        self.render()
        return observation

    def generate_random_map(self):

        ####START COMMMENT OUT
        #Random generated map
        # start = random.choice([1,1,1,1,1,1,1,1,1,1])
        # stop = random.choice([13,13,13,13,13,13,13,13,13,13])
        # random_integers = np.random.random_integers(start,stop,(20,20))
        # flag = bn.rvs(p=0.99, size=(20,20))
        # #add 10% (1-p) of any value
        # other_features = np.full((20,20),33)
        # random_integers[flag==0] = other_features[flag==0]
        # self.map_volume = CNP.create_custom_map(random_integers)#CNP.create_custom_map(np.random.random_integers(start,stop,(self.mapw,self.maph)))#CNP.map_to_volume_dict(self._map[0],self._map[1], self.mapw, self.maph)#CNP.create_custom_map(np.random.random_integers(start,stop,(self.mapw,self.maph))) #CNP.create_custom_map(random.choice(self.custom_maps))
        # # Set hiker's and drone's locations
        # #hiker = (random.randint(2, self.map_volume['vol'].shape[1] - 1), random.randint(2, self.map_volume['vol'].shape[1] - 2)) #(8,8) #
        # #if self.dropping:
        # hiker = (random.randint(2, self.map_volume['vol'].shape[1] - 2), random.randint(2, self.map_volume['vol'].shape[1] - 2))#(10,10)#(random.randint(2, self.map_volume['vol'].shape[1] - 2), random.randint(2, self.map_volume['vol'].shape[1] - 2))  #random.choice([(4,5),(5,5),(5,4),(4,4)]) (7,8) #
        all_no_goes = [] # all points hiker is not allowed to go (e.g. water)
        # better random map
        just_grass = np.full((20, 20), 2)
        # add some trail, trees
        updated_map = self.add_blob(just_grass, 15, 5)[0]
        for i in range(random.randint(1, 10)):
            updated_map = self.add_blob(updated_map, 50, random.choice([1, 3]))[0]

        # add some water (maybe)
        if random.randint(0, 1):
            updated_map, no_go_points = self.add_blob(updated_map, 100, 15)
            all_no_goes.append(no_go_points)
        # add some mountain ridges
        updated_map, no_go_points = self.add_blob(updated_map, 75, 26)
        all_no_goes.append(no_go_points)
        # a few small mountain ridges
        for i in range(random.randint(1, 5)):
            updated_map, no_go_points = self.add_blob(updated_map, random.randint(1, 10), 25)
            all_no_goes.append(no_go_points)
        # add some bushes
        # small clusters, 5 times
        for i in range(random.randint(1, 8)):
            updated_map = self.add_blob(updated_map, random.randint(1, 5), 4)[0]
        # add one campfire
        updated_map, no_go_points = self.add_blob(updated_map, 0, 33)
        all_no_goes.append(no_go_points)

        self.map_volume = CNP.create_custom_map(updated_map)

        # self.map_volume = CNP.create_custom_map(np.random.random_integers(start, stop, (20, 20)))#CNP.create_custom_map(np.random.random_integers(start, stop, (20, 20)))#CNP.map_to_volume_dict(self._map[0],self._map[1], self.mapw, self.maph)#CNP.create_custom_map(self.custom_map)#CNP.create_custom_map(np.random.random_integers(start, stop, (20, 20)))#CNP.map_to_volume_dict(self._map[0],self._map[1], self.mapw, self.maph)#CNP.create_custom_map(np.random.random_integers(start, stop, (10, 10)))#CNP.create_custom_map(self.custom_map)#CNP.create_custom_map(np.random.random_integers(start, stop, (10, 10)))

        # Set hiker's and drone's location
        # hiker = (random.randint(2, self.map_volume['vol'].shape[1] - 1), random.randint(2, self.map_volume['vol'].shape[1] - 2)) #(8,8) #
        # (8, 1)  # (6,3)#
        hiker = (
            random.randint(3, self.map_volume['vol'].shape[1] - 3),
            random.randint(3, self.map_volume['vol'].shape[1] - 3))
        
        while self.hiker_in_no_go_list(hiker, all_no_goes): # Place the hiker but check if is placed on a no go feature
            hiker = (random.randint(3, self.map_volume['vol'].shape[1] - 3),
                     random.randint(3, self.map_volume['vol'].shape[1] - 3))

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

        """Returns a single 3-channel RGB image corresponding to a horizontal slice through volume at given altitude"""

        canvas = np.zeros( ( self.map_volume['vol'].shape[1],
                             self.map_volume['vol'].shape[1],
                             3),
                            dtype=np.uint8)

        og_vol = self.original_map_volume
        combinations = list( itertools.product( range(0, canvas.shape[0]),
                                                range(0, canvas.shape[0])  ))

        for x, y in combinations:
            if og_vol['vol'][altitude][x, y] == 0.0:
                canvas[x, y, :] = [255, 255, 255]
            else:
                canvas[x, y, :] = og_vol['value_feature_map'][ og_vol['vol'][altitude][x, y] ]['color']

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

        # put the drone in the image layer
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
        drone_position = np.where(
            self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.altitude]['val'])

        nextstepimage = self.create_nextstep_image()

        obs['nextstepimage'] = nextstepimage
        obs['img'] = map
        obs['image_layers'] = image_layers

        return obs


    def get_drone_position(self):

        """returns the position of the drone as a tuple with altitude first and then x and y positions (Z,X,Y)"""

        # The appearance (tile type) of the drone is altitude dependent so we need to look this up every time

        drone_tile_id = self.map_volume['feature_value_map']['drone'][self.altitude]['val']

        # Find tile that looks like the drone

        drone_zxy = np.where( self.map_volume['vol'] == drone_tile_id  )

        return drone_zxy


    def get_hiker_position(self):

        """returns the position of the hiker as a tuple with altitude first and then x and y positions (Z,X,Y)"""
        # Here the appearance of the drone is altitude dependent so we need to look this up every time

        drone_tile_id = self.map_volume['feature_value_map']['hiker'][self.altitude]['val']

        # Find tile that looks like the drone

        drone_zxy = np.where( self.map_volume['vol'] == drone_tile_id  )

        return drone_zxy


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


print('complete')