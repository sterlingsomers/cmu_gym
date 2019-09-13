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
import pickle
from util import deep_update, one_based_normalized_circular_loss

from scipy.misc import imresize
import matplotlib.pyplot as plt

from gym_gridworld.envs import create_np_map as CNP

from gym_gridworld.envs.mavsim_lib_server import MavsimLibHandler

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
            1:'North',        # row++
            2:'North East',
            3:'East',         # col++
            4:'South East',   # row--
            5:'South',        # row--
            6:'South West',
            7:'West',         # col --
            8:'North West'
        }

        self.heading_to_short_description = {
            1:'N',         # row ++
            2:'NE',
            3:'E',         # col ++
            4:'SE',        # row --
            5:'S',         # row --
            6:'SW',
            7:'W',         # col --
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

        if action_int==None or action_int == self.DROP:
            return old_heading

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

        label =  self.action_to_short_description[action_int]
        return label

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

nixel_sample_map = np.array(

    [[15., 15.,  1.,  2.,  2.,  2.,  2.,  2., 25., 25.,  2.,  2., 25., 25., 24., 26.,  2.,  2., 25., 24.],
     [15., 15.,  2.,  2.,  2.,  2.,  2.,  2.,  2., 31., 25.,  2.,  2.,  2.,  2., 24.,  2., 24., 24., 25.],
     [15., 15.,  1.,  2.,  2.,  2.,  2., 24., 25., 24.,  2., 26.,  2.,  2.,  2.,  2.,  2.,  2.,  2., 24.],
     [15., 15.,  2., 25.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2., 24.,  2.,  2.,  2.,  2., 24., 24., 25.],
     [15., 15.,  1., 31.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2., 24.,  2.,  2.,  2., 31., 25., 24.,  2.],
     [15., 15.,  1., 26.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2., 25., 25., 24., 31., 31., 24.,  2.,  2.],
     [15., 15.,  2., 25.,  2., 24.,  2.,  2.,  2.,  2.,  2.,  2., 24., 25., 25.,  2.,  2., 24.,  2.,  2.],
     [15., 15.,  2.,  1.,  1.,  2., 24.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2., 24.,  2.],
     [15., 15.,  2.,  2., 17.,  2., 24., 24., 24.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.],
     [15., 15.,  2.,  1.,  2.,  1., 26., 25.,  2.,  2.,  2.,  2.,  2.,  2.,  2., 25., 26.,  2.,  2.,  2.],
     [15., 15., 15.,  1.,  2.,  2., 25., 25., 24.,  2., 25.,  2.,  2.,  2.,  2., 25., 26.,  2.,  2.,  2.],
     [15., 15., 15.,  1.,  2.,  2., 25.,  2.,  2.,  2., 31., 24.,  2.,  2.,  2., 26.,  2.,  2.,  2.,  2.],
     [15., 15., 15., 15., 15.,  2.,  2., 25.,  2.,  2., 24., 26.,  2.,  2.,  2.,  2.,  2.,  2.,  2., 24.],
     [ 1., 15., 15., 15., 15., 15.,  2.,  1.,  1.,  1.,  1.,  1.,  1.,  2., 10.,  1.,  2., 24.,  2., 26.],
     [ 1.,  1., 15., 15., 15., 15., 15.,  1.,  2.,  1.,  1.,  1.,  1.,  1., 10.,  1.,  2., 24., 24., 26.],
     [ 1.,  1.,  1., 15., 15., 15., 15., 15.,  2.,  1.,  1.,  1.,  1.,  1., 10.,  1.,  2., 31., 26., 25.],
     [ 1.,  1.,  1.,  2., 15., 15., 15., 15.,  1.,  1.,  1.,  2.,  2., 10.,  2.,  1.,  2., 26., 25.,  2.],
     [ 1.,  1.,  1.,  1.,  1., 15., 15., 15., 15., 15.,  1.,  2.,  2., 10.,  1.,  1., 26., 26.,  2.,  2.],
     [ 1.,  1.,  2.,  1.,  1.,  1., 15., 15., 15., 15., 15.,  2.,  2.,  1., 10.,  1., 24., 25.,  2.,  2.],
     [ 1.,  1.,  1.,  1.,  1.,  1.,  1., 15., 15., 15., 15., 15.,  2.,  1., 10.,  2., 25.,  2.,  2.,  2.]]



)

def list_ij_equal_to(A,x):

    """"Given a 2D array A and a scalar value x,
        return a list of row-colum i,j pairs where A is equal to x"""

    i_and_j_lists = np.where(A==x)

    ij_pairs = [ p for p in zip( i_and_j_lists[0], i_and_j_lists[1] ) ]

    return ij_pairs


class GridworldEnv(gym.Env):

    metadata = {'render.modes': ['human']}
    num_envs = 1

    def __init__(self, params={} ):


        print("gridworld_env: NEW GRIDWORLD CREATED")

        self.params = params


        # # TODO: Pass the environment with arguments

        #num_alts = 4
        self.verbose = params['verbose'] # to show the environment or not
        self.dropping = True # This is for the reset to select the proper starting locations for hiker and drone
        self.restart_once_done = True  # restart or not once done
        self.drop = False
        self.countdrop = 0
        self.no_action_flag = False


        self.mapw = 20
        self.maph = 20
        self.dist_old = 1000
        self.drop_package_grid_size_by_alt = {1: 3, 2: 5, 3: 7}
        self.factor = 5
        self.reward = 0

        self.agent_action_to_sim_action = \
                       [ ACTION.DOWN_LEFT_45,  ACTION.DOWN_FORWARD,  ACTION.DOWN_RIGHT_45,
                         ACTION.LEVEL_LEFT_45, ACTION.LEVEL_FORWARD, ACTION.LEVEL_RIGHT_45,
                         ACTION.UP_LEFT_45,    ACTION.UP_FORWARD,    ACTION.UP_RIGHT_45 ]

        self.action_space = spaces.Discrete(len(self.agent_action_to_sim_action))

        self.obs_shape = [100,100,3]
        self.observation_space = spaces.Box(low=0, high=255, shape=self.obs_shape)

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

        if self.params['use_mavsim_simulator']:
            self.mavsimhandler = MavsimLibHandler(self.params['mavsim'])


        #self.image_layers = {}


        # The drone stencil gives the subset of pixel locations that need to be overwritten
        # in the drone color scheme over top of the map cell.
        #
        # The drone stencil is heading dependent so that the appearance of the drone changes with heading

        self.drone_stencil = {}

        self.drone_stencil[1] = [[(0, 2), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4)], np.zeros((5, 5, 3))]
        self.drone_stencil[2] = [[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (0, 4), (1, 3), (2, 3), (1, 2)], np.zeros((5, 5, 3))]
        self.drone_stencil[3] = [[(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (1, 3), (2, 3), (3, 3), (2, 4)], np.zeros((5, 5, 3))]
        self.drone_stencil[4] = [[(0, 4), (1, 3), (2, 3), (3, 3), (4, 4), (2, 2), (3, 2), (3, 1), (4, 0)], np.zeros((5, 5, 3))]
        self.drone_stencil[5] = [[(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (4, 2)], np.zeros((5, 5, 3))]
        self.drone_stencil[6] = [[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (2, 1), (3, 1), (3, 2), (4, 0)], np.zeros((5, 5, 3))]
        self.drone_stencil[7] = [[(2, 0), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)], np.zeros((5, 5, 3))]
        self.drone_stencil[8] = [[(0, 0), (4, 0), (1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (1, 3), (0, 4)], np.zeros((5, 5, 3))]


        # Hiker glyphs (supposed to look like a person with arms and legs outstretched)
        # We describe an orthogonal and diagonal version of the hiker and then just rotate these to get full 8 headings

        self.hiker_glyph_ortho = np.array([ [ 0,0,1,0,0 ],
                                            [ 1,1,1,1,1 ],
                                            [ 0,0,1,0,0 ],
                                            [ 0,1,0,1,0 ],
                                            [ 1,0,0,0,1 ], ])

        self.hiker_glyph_diag  = np.array([ [ 0,0,1,0,1 ],
                                            [ 0,0,0,1,0 ],
                                            [ 1,1,1,0,1 ],
                                            [ 0,0,1,0,0 ],
                                            [ 0,0,1,0,0 ], ])

        self.hiker_stencil = {}
        self.hiker_stencil[1] = [ list_ij_equal_to( self.hiker_glyph_ortho, 1), np.zeros((5, 5, 3)) ]
        self.hiker_stencil[2] = [ list_ij_equal_to( self.hiker_glyph_diag, 1), np.zeros((5, 5, 3)) ]

        self.hiker_stencil[3] = [ list_ij_equal_to( np.rot90(self.hiker_glyph_ortho, k=1, axes=(1, 0)), 1), np.zeros((5, 5, 3)) ]
        self.hiker_stencil[4] = [ list_ij_equal_to( np.rot90(self.hiker_glyph_diag, k=1, axes=(1, 0)), 1), np.zeros((5, 5, 3)) ]

        self.hiker_stencil[5] = [ list_ij_equal_to( np.rot90(self.hiker_glyph_ortho, k=2, axes=(1, 0)), 1), np.zeros((5, 5, 3)) ]
        self.hiker_stencil[6] = [ list_ij_equal_to( np.rot90(self.hiker_glyph_diag, k=2, axes=(1, 0)), 1), np.zeros((5, 5, 3)) ]

        self.hiker_stencil[7] = [ list_ij_equal_to( np.rot90(self.hiker_glyph_ortho, k=3, axes=(1, 0)), 1), np.zeros((5, 5, 3)) ]
        self.hiker_stencil[8] = [ list_ij_equal_to( np.rot90(self.hiker_glyph_diag, k=3, axes=(1, 0)), 1), np.zeros((5, 5, 3)) ]

        self.hiker_image = np.zeros((5, 5, 3))

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

        # Altitude rewards
        # self.alt_rewards = {0:-1, 1:1, 2:-0.5, 3:-0.8} # This is bad!
        self.alt_rewards = {0:0, 1:1, 2:0.5, 3:0.08 }


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
            self.map_dictionary['vol'] == self.map_dictionary['feature_value_map']['drone'][self.drone_altitude]['val'])
        self.no_action_flag = False

        if local_coordinates[1] == 0 or \
            local_coordinates[2] == 0 or \
            local_coordinates[1] == self.map_dictionary['vol'].shape[1] - 1 or \
            local_coordinates[2] == self.map_dictionary['vol'].shape[1] - 1:
            print("NOACTION")
            self.no_action_flag = True
            self.reward = 0#-1 # might be redundant cauz u have a reward = 0 in the step function if the no action flag is true. Also this returns 0
            # self.package_state = 'OOB' # You might need it when you drop out of bounds
            #self.drop = True
            return 0
        self.drop = True
        alt = self.drone_altitude
        drone_position = np.where(self.map_dictionary['vol'] == self.map_dictionary['feature_value_map']['drone'][self.drone_altitude]['val'])
        region = self.drop_package_grid_size_by_alt[self.drone_altitude]
        neighbors, left, top, right, bottom = self.neighbors(self.original_map_volume['vol'][0], int(drone_position[1]),
                                              int(drone_position[2]), region)
        w = self.map_dictionary['vol'][0][left:right, top:bottom]
        is_hiker_in_neighbors = np.any(w == self.map_dictionary['feature_value_map']['hiker']['val'])

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
                self.reward = (reward + self.alt_rewards[self.drone_altitude]) / ((self.pack_dist ** 2) + 1e-7)#reward + self.alt_rewards[self.altitude] + 1/((self.pack_dist** 2) + 1e-7)#

        self.package_position = pack_world_coords
        self.package_dropped = True
        x = eval(self.actionvalue_heading_action[7][self.drone_heading])


    def take_action(self, delta_alt=0, delta_x=0, delta_y=0, new_heading=1):

        """Take the action and update the world with its effects.
           In the case of mavsim, it updates old heading to new_heading and then takes a step in this direction.

           Returns 0 if action was OK and non-zero if action resulted in a failure of some kind - likely a crash"""

        action_ok = 0   #

        if self.params['verbose']:
            print("gridworld_env take_action current heading {} new_heading {}".format( self.get_drone_heading(), new_heading))

        if self.params['use_mavsim_simulator']:

            #print("take_action mavsim heading {} {}".format(new_heading, HEADING.heading_to_short_description[new_heading]))

            vol_shape = self.map_dictionary['vol'].shape
            distance = 1

            self.drone_altitude = self.mavsimhandler.get_drone_position()[0]

            new_alt = self.drone_altitude + delta_alt if self.drone_altitude + delta_alt < 4 else 3

            # Ma
            self.mavsimhandler.head_to( new_heading , distance, new_alt )

            self.drone_heading = self.mavsimhandler.get_drone_heading()
            drone_position = self.mavsimhandler.get_drone_position()
            self.drone_altitude = drone_position[0]

            if drone_position[1] < 0 or \
               drone_position[2] < 0 or \
               drone_position[1] > vol_shape[1] - 1 or \
               drone_position[2] > vol_shape[2] - 1:

               action_ok = 1  # We crashed because the return is non-zero

        else:


            # print("stop")
            vol_shape = self.map_dictionary['vol'].shape

            local_coordinates = np.where(
                self.map_dictionary['vol'] == self.map_dictionary['feature_value_map']['drone'][self.drone_altitude]['val'])

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

            new_alt = self.drone_altitude + delta_alt if self.drone_altitude + delta_alt < 4 else 3
            if new_alt < 0:
                return 0

            # put back the original
            self.map_dictionary['vol'][self.drone_altitude][local_coordinates[1], local_coordinates[2]] = float(
                self.original_map_volume['vol'][local_coordinates])

            # put the hiker back
            self.map_dictionary['vol'][self.hiker_position] = self.map_dictionary['feature_value_map']['hiker']['val']

            # put the drone in
            # self.map_volume['tile_typeid_array'][local_coordinates[1]+delta_y,local_coordinates[2]+delta_x] = self.map_volume['feature_value_map']['drone'][new_alt]['val']
            self.map_dictionary['vol'][new_alt][local_coordinates[1] + delta_y, local_coordinates[2] + delta_x] = \
            self.map_dictionary['feature_value_map']['drone'][new_alt]['val']

            self.drone_altitude = new_alt
            self.drone_heading = new_heading

        return 1


    def check_for_drone_over_hiker(self,drone_position):

        """returns true if drone is over hiker"""

        # First coordinate is altitude, second is x and third is y

        return (drone_position[1], drone_position[2]) == (self.hiker_position[1], self.hiker_position[2])


    def check_for_crash(self,drone_position):

        """Returns zero if drone is OK, -1 if drone descended to altitude zero
           and returns tile_id of the item crashed into if crashed into obstacle. """

        #print("Checking for crash at drone_position {}".format(drone_position))

        if self.params['use_mavsim_simulator']:

            if self.params['verbose']:
                pos = self.get_drone_position()
                print("gridworld_env check_for_crash mavsim cmu_pos={} local_driver_crash_state {}".format(pos, self.mavsimhandler.crashed))
                #v = self.mavsimhandler._command("('FLIGHT', 'MS_QUERY_TERRAIN', %d, %d)" % (pos[2], pos[1]))
                #print("gridworld_env check_for_crash mavsim terrain query {}".format(v))

            if self.mavsimhandler.crashed:
                return 1 # int(self.original_map_volume['vol'][tuple(drone_position)])
            else:
                return 0
        else:
            if self.drone_altitude == 0:
                return -1

            return int(self.original_map_volume['vol'][tuple(drone_position)])


    def step(self, action):

        ''' return next observation, reward, finished, success '''

        #alts, types = self.mavsimhandler._pull_map_for_location_size(0,0,20,20)
        #plt.imshow(types, cmap=plt.get_cmap('hsv'))
        #plt.show()

        action = self.agent_action_to_sim_action[int(action)]  # Convert from action index, to one of the gridworld_domain actions

        if self.params['verbose']:
            print("gridworld_env.step(action={} {}) step# {} drone_pos {} drone_hdg {}".format(
                action,
                ACTION.to_short_string(action),
                self.step_number,
                self.get_drone_position(),
                self.get_drone_heading())    )



        info = {}
        info['success'] = False
        done = False

        if self.params['verbose']:
            print("   gridworld_env.step params[episode_length] {}".format(self.params['episode_length']))

        self.step_number = self.step_number + 1
        if self.step_number > self.params['episode_length']:
            if self.params['verbose']:
                print("   gridworld_env.step exceeded episode timeout {}".format(self.params['episode_length']))
            done = True
            reward = -1
            info['success'] = False
            info['ex'] = 'Timeout'
            info['Rtime'] = -1

            return (self.generate_observation(), reward, done, info)


        drone_position = self.get_drone_position()
        #print("Step getting drone position: {}".format(drone_position))

        # Euclidean distance on the ground (ignore height difference)
        self.dist = np.linalg.norm(  np.array(drone_position[-2:]) - np.array(self.hiker_position[-2:])  )

        crash = self.check_for_crash(drone_position)

        if crash!=0:
            if self.params['verbose']:
                print('   gridworld_env.step crashed')

            reward = -1
            info['Rcrash']=-1
            done = True
            print("CRASH")
            info['ex'] = 'crash'
            info['crash_obj']=crash
            info['success']= False

            if self.restart_once_done: # HAVE IT ALWAYS TRUE!!! It learned the first time WITHOUT RESETING FROM CRASH
                return (self.generate_observation(), reward, done, info) # You should get the previous obs so no change here, or return obs=None

        # Do the action (drone is moving). If we crash we dont perform an action so no new observation

        x = eval(self.actionvalue_heading_action[action][ self.get_drone_heading() ])

        if self.no_action_flag == True:
            reward = self.reward
            done = True
            if self.restart_once_done: # HAVE IT ALWAYS TRUE!!!
                return (self.generate_observation(), reward, done, info)


        if self.params['goal_mode']=='navigate':

            if self.check_for_drone_over_hiker(drone_position):

                if self.params['verbose']:
                    print('gridworld_env.step reached position over hiker')

                reward = 5 #1
                done = True
                info['ex']='arrived'
                info['Rhike']=1


                if 'align_drone_and_hiker_heading' in self.params and self.params['align_drone_and_hiker_heading']:

                    if self.params['verbose']:
                        print("gridworld_env.step checking heading alignment {} =? {}".format(
                            HEADING.heading_to_short_description[self.get_drone_heading()],
                            HEADING.heading_to_short_description[self.hiker_heading]  )  )


                    # We get partial credit the closer the headings match

                    R = one_based_normalized_circular_loss( self.drone_heading, self.hiker_heading, 8)

                    if self.params['verbose']:
                        print("gridworld_env.step reward for heading alignment is {}".format(R))

                    info['Rhead']=R
                    reward=reward+R

                    if self.drone_heading == self.hiker_heading:
                        info['OKhead']=True

                if 'align_drone_and_hiker_altitude' in self.params and self.params['align_drone_and_hiker_altitude']:

                    if self.params['verbose']:
                        print("gridworld_env.step checking altitude alignment {} =? {}".format(self.drone_altitude, self.hiker_altitude))

                    R = ( 4 - abs(self.drone_altitude-self.hiker_altitude) ) / 4
                    reward=reward+R
                    info['Ralt']=R

                    if self.params['verbose']:
                        print("gridworld_env.step reward for altitude alignment is {}".format(R))

                    if self.drone_altitude == self.hiker_altitude:
                        info['OKalt']=True

                return (self.generate_observation(), reward, done, info)

        # Multiple packages

        if self.params['goal_mode']=='drop' and self.drop:
            self.drop = 0
            reward = self.reward
            info['Rdrop']=reward
            print('DROP!!!', 'self.reward=', self.reward, 'alt_reward=', self.alt_rewards[self.drone_altitude],
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

        if self.get_drone_heading() in [ HEADING.NORTH, HEADING.EAST, HEADING.SOUTH, HEADING.WEST ]:  # orthogonal moves cost less
            reward = -0.01
        else:  # Diagonal moves in space cost more
            reward = -0.015


        #-0.01#(self.alt_rewards[self.altitude]*0.1)*((1/((self.dist**2)+1e-7))) # -0.01 + # The closer we are to the hiker the more important is to be close to its altitude
        #print("scale:",(1/((self.dist**2+1e-7))), "dist=",self.dist+1e-7, "alt=", self.altitude, "drone:",drone, "hiker:", hiker,"found:", self.check_for_hiker())

        info['Rstep']=reward
        obs = self.generate_observation()

        return (obs, reward, done, info)


    def get_tile_typeid_at(self, row,col):

        return self.map_dictionary['tile_typeid_array'][row][col]


    def reset(self, param_updates={} ):

        """ goal_mode in { None, 'navigate' } """

        deep_update( self.params, param_updates )

        if self.params['verbose']:

            if 'curriculum_radius' in self.params:
                print("gridworld_env.reset params[curriculum_radius] {}".format(self.params['curriculum_radius']))

            if 'drone_initial_position' in self.params:
                print("gridworld_env.reset params[drone_initial_position] {}".format(self.params['drone_initial_position']))


        self.step_number = 0

        self.dist_old = 1000
        self.drop = False
        self.countdrop = 0
        self.no_action_flag = False

        if 'drone_initial_heading' in self.params:
            self.drone_heading=self.params['drone_initial_heading']
        else:
            self.drone_heading = random.randint(1, 8)


        if 'drone_initial_altitude' in self.params:
            self.drone_altitude = self.params['drone_initial_altitude']
        else:
            self.drone_altitude = random.randint(1, 3)

        if self.params['verbose']:
            print('gridworld_env reset setting drone heading to {} {}'.format(
                self.drone_heading,
                HEADING.to_short_string(self.drone_heading)))
            print('gridworld_env reset setting altitude to {} '.format(self.drone_altitude))

        self.reward = 0
        self.crash = 0
        self.package_dropped = 0
        self.package_position = ()


        #---------------------------------------------
        # Get a submap to fly on
        #---------------------------------------------

        if 'use_custom_map' in self.params:

            self.map_dictionary = CNP.create_custom_map(box_canyon_map)

        else:

            self.submap_offset = random.choice(self.params['submap_offsets'])


            # In the mavsim paradigm, we request to fly in a submap on the larger global map

            if self.params['use_mavsim_simulator']:

                self.mavsimhandler.reset( self.submap_offset,(20,20) )
                self.map_dictionary   = self.mavsimhandler.get_submap()


            # In the case of randomly generated maps, we directly load a 20x20 slice from disk

            else:

                map_filename = '{}-{}.mp'.format(self.submap_offset[0],self.submap_offset[1])
                self.map_dictionary = pickle.load(open(self.params['map_path'] + '/' + map_filename, 'rb'))



        # place the hiker

        valid_hiker_drone_pair = False

        while not valid_hiker_drone_pair:

            if self.params['verbose']:
                print("gridworld_env reset trying to place hiker")

            if not 'hiker_initial_position' in self.params:

                # ToDo: investigate this: hiker_safe_points = np.isin(map_, self.tile_ids_compatible_with_hiker)

                hiker_safe_points = []

                for val in self.tile_ids_compatible_with_hiker:

                    where_array = np.where(self.map_dictionary['tile_typeid_array'] == val)

                    hiker_safe_points = hiker_safe_points + [(x, y) for x, y in zip(where_array[0], where_array[1])
                                                             if x >= 3 and y >= 3
                                                             and x <= self.map_dictionary['vol'].shape[1] - 3  # ToDo: BOB - should one of these be shape[2]??
                                                             and y <= self.map_dictionary['vol'].shape[1] - 3]  #ToDo: Bob -replace with mapw, maph?

                if len(hiker_safe_points)<1:
                    print("gridworld_env.reset WARNING Could not find safe hiker location on map. Setting to (5,5)")
                    hiker = (5,5)
                else:
                    hiker = random.choice(hiker_safe_points)

            else:
                hiker = self.params['hiker_initial_position']

            if self.params['use_mavsim_simulator']:
                print("gridworld_env.reset setting mavsim hiker position to {}".format(hiker))
                self.mavsimhandler.set_hiker_position(hiker)



            if not 'hiker_initial_heading' in self.params:

                self.hiker_heading= random.choice(range(1,8+1))
            else:
                self.hiker_heading = self.params['hiker_initial_heading']

            if not 'hiker_initial_altitude' in self.params:
                self.hiker_altitude = random.choice(range(1,3+1))
            else:
                self.hiker_altitude = self.params['hiker_initial_altitude']



            if not 'drone_initial_position' in self.params:

                drone_safe_points = []
                for val in self.tile_ids_compatible_with_drone_altitude[self.drone_altitude]:

                    where_array = np.where(self.map_dictionary['tile_typeid_array'] == val)

                    drone_safe_points = drone_safe_points + [(x, y) for x, y in zip(where_array[0], where_array[1])
                                                             if x >= 3 and y >= 3
                                                             and x <= self.map_dictionary['vol'].shape[1] - 3  # ToDo: BOB - should one of these be shape[2]??
                                                             and y <= self.map_dictionary['vol'].shape[1] - 3]

                if len(drone_safe_points)<1:
                    print("gridworld_env.reset WARNING Could not find safe drone location on map, setting to (1,1).")
                    drone_safe_points=[(1,1)]

                D = distance.cdist([hiker], drone_safe_points, 'chebyshev').astype(int) # Distances from hiker to all drone safe points

                if 'curriculum_radius' in self.params:
                    if self.params['verbose']:
                        print("gridworld_env.step Using curriculum mode with radius:{}".format(self.params['curriculum_radius']))
                    close_location_idxs = D[0] < self.params['curriculum_radius']
                    if sum(close_location_idxs)==0:
                        print("gridworld_env.reset WARNING - no safe locations next to hiker location {} -- trying again ".format(hiker))
                    else:
                        valid_hiker_drone_pair=True
                        drone_safe_points = np.array(drone_safe_points)[close_location_idxs,:]
                    #print("Curriculum with radius {}".format(self.curriculum_radius))
                    #print("Satisfying points {}".format(drone_safe_points))

                drone = random.choice(drone_safe_points)
                valid_hiker_drone_pair=True

            else:
                valid_hiker_drone_pair=True
                drone = self.params['drone_initial_position']


            if self.params['verbose']:
                print('gridworld_env.reset hiker_position {} drone_position {} '.format(hiker,drone))

            if self.params['use_mavsim_simulator']:
                self.mavsimhandler.set_drone_position( [ self.drone_altitude, drone[0], drone[1] ],
                                                         self.drone_heading )


        self.original_map_volume = copy.deepcopy(self.map_dictionary)

        # self.local_coordinates = [local_x,local_y]
        # self.world_coordinates = [70,50]
        self.reference_coordinates = [self.submap_offset[0], self.submap_offset[1]]


        print("gridworld_env.reset print drone location {},{} ".format(drone[0],drone[1]))

        # put the drone in
        self.map_dictionary['vol'][self.drone_altitude][drone[0], drone[1]] = \
                   self.map_dictionary['feature_value_map']['drone'][self.drone_altitude]['val']

        # self.map_volume['tile_typeid_array'][drone[0], drone[1]] = self.map_volume['feature_value_map']['drone'][self.altitude][
        #     'val']

        self.map_dictionary['rgb_image_with_actors'][drone[0], drone[1]] = self.map_dictionary['feature_value_map']['drone'][self.drone_altitude][
            'color']


        #print("GridworldEnv reset extracted drone position {} (Z,X,Y) ".format(self.get_drone_position()))

        # self.map_volume[altitude]['drone'][local_y, local_x] = 1.0
        # put the hiker in@ altitude 0

        self.map_dictionary['vol'][0][hiker[0], hiker[1]] = self.map_dictionary['feature_value_map']['hiker']['val']

        # self.map_volume['tile_typeid_array'][hiker[0], hiker[1]] = self.map_volume['feature_value_map']['hiker']['val']

        self.map_dictionary['rgb_image_with_actors'][hiker[0], hiker[1]] = self.map_dictionary['feature_value_map']['hiker']['color']

        # ToDo: BOB This next line seems redundant - why calculate hiker_position from map from hiker?

        self.hiker_position = [self.hiker_altitude, hiker[0], hiker[1]]

        #self.image_layers[0] = self.create_image_from_volume(0)
        #self.image_layers[1] = self.create_image_from_volume(1)
        #self.image_layers[2] = self.create_image_from_volume(2)
        #self.image_layers[3] = self.create_image_from_volume(3)
        #self.image_layers[4] = self.create_image_from_volume(4)

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

        self.map_dictionary = CNP.create_custom_map(updated_map)

        # self.map_volume = CNP.create_custom_map(np.random.random_integers(start, stop, (20, 20)))#CNP.create_custom_map(np.random.random_integers(start, stop, (20, 20)))#CNP.map_to_volume_dict(self._map[0],self._map[1], self.mapw, self.maph)#CNP.create_custom_map(self.custom_map)#CNP.create_custom_map(np.random.random_integers(start, stop, (20, 20)))#CNP.map_to_volume_dict(self._map[0],self._map[1], self.mapw, self.maph)#CNP.create_custom_map(np.random.random_integers(start, stop, (10, 10)))#CNP.create_custom_map(self.custom_map)#CNP.create_custom_map(np.random.random_integers(start, stop, (10, 10)))

        # Set hiker's and drone's location
        # hiker = (random.randint(2, self.map_volume['vol'].shape[1] - 1), random.randint(2, self.map_volume['vol'].shape[1] - 2)) #(8,8) #
        # (8, 1)  # (6,3)#
        hiker = (
            random.randint(3, self.map_dictionary['vol'].shape[1] - 3),
            random.randint(3, self.map_dictionary['vol'].shape[1] - 3))
        
        while self.hiker_in_no_go_list(hiker, all_no_goes): # Place the hiker but check if is placed on a no go feature
            hiker = (random.randint(3, self.map_dictionary['vol'].shape[1] - 3),
                     random.randint(3, self.map_dictionary['vol'].shape[1] - 3))

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

        for point in self.drone_stencil[heading][0]:
            self.drone_stencil[heading][1][point[0], point[1]] = color

        return self.drone_stencil[heading][1]


    def create_image_from_volume(self, altitude):

        """Returns a single 3-channel RGB image corresponding to a horizontal slice through volume at given altitude"""

        canvas = np.zeros( ( self.map_dictionary['vol'].shape[1],
                             self.map_dictionary['vol'].shape[1],
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

        drone_position = self.get_drone_position()

        drone_position_flat = [int(drone_position[1]), int(drone_position[2])]
        # hiker_found = False
        # hiker_point = [0, 0]
        # hiker_background_color = None
        column_number = 0
        for xy in self.possible_actions_map[self.drone_heading]:
            if drone_position_flat[0] + xy[0] >= 0 and drone_position_flat[1] + xy[1] >= 0 and drone_position_flat[0] + \
                    xy[0] <= self.map_dictionary['vol'].shape[1] - 1 and drone_position_flat[1] + xy[1] <= \
                    self.map_dictionary['vol'].shape[2] - 1:

                # try:
                # no hiker if using original
                column = self.map_dictionary['vol'][:, drone_position_flat[0] + xy[0], drone_position_flat[1] + xy[1]]

            # except IndexError:
            else:
                column = [1., 1., 1., 1., 1.]
            slice[:, column_number] = column
            column_number += 1
            #print("ok")

        # put the drone in

        drone_tile_id = self.map_dictionary['feature_value_map']['drone'][self.drone_altitude]['val']
        slice[self.drone_altitude, 2] = drone_tile_id


        combinations = list(itertools.product(range(0, canvas.shape[0]), range(0, canvas.shape[0])))
        for x, y in combinations:
            if slice[x, y] == 0.0:
                canvas[x, y, :] = [255, 255, 255]
            else:
                canvas[x, y, :] = self.map_dictionary['value_feature_map'][slice[x, y]]['color']

        # increase the image size, then put the hiker in
        canvas = imresize(canvas, self.factor * 100, interp='nearest')
        self.ego = np.flip(slice,0)
        return imresize(np.flip(canvas, 0), 20 * self.map_dictionary['vol'].shape[2], interp='nearest')


    def generate_observation(self):

        obs = {}
        obs['volume'] = self.map_dictionary
        #image_layers = copy.deepcopy(self.image_layers)

        map_image_with_actors = copy.deepcopy(self.original_map_volume['rgb_image_with_actors'])

        # put the drone in the image layer

        drone_position = self.get_drone_position()

        if self.params['verbose']:
            print("gridworld_env generate_observation drone_position {}".format(drone_position))

        #print("generate_observation drone_position {} ".format(drone_position))

        #drone_position = np.where(
        #    self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.drone_altitude]['val'])

        drone_image_offset = (int(drone_position[1]) * self.factor, int(drone_position[2]) * self.factor)

        drone_color = self.map_dictionary['feature_value_map']['drone'][ self.drone_altitude]['color']
        #for pixel in self.drone_stencil[self.drone_heading][0]:
        #    image_layers[ self.drone_altitude ][ drone_image_offset[0] + pixel[0], drone_image_offset[1] + pixel[1], :] = drone_color


        # put the hiker in the image layers
        if self.params['verbose']:
            print("gridworld_env.generate_observation self.hiker_position {}".format(self.hiker_position))
        hiker_image_offset = (int(self.hiker_position[1] * self.factor), int(self.hiker_position[2]) * self.factor)
        hiker_color = self.map_dictionary['feature_value_map']['hiker']['color']

        if 'render_hiker_altitude' in self.params and self.params['render_hiker_altitude']:
            hiker_color_adjusted = [ c*( self.hiker_altitude )/3 for c in hiker_color]
        else:
            hiker_color_adjusted= hiker_color

        #for pixel in self.hiker_stencil[self.hiker_heading][0]:
        #    image_layers[0][hiker_image_offset[0] + pixel[0], hiker_image_offset[1] + pixel[1], :] = hiker_color_adjusted

        # map = self.original_map_volume['rgb_image_with_actors']
        map_image_with_actors = imresize(map_image_with_actors, self.factor * 100, interp='nearest')  # resize by factor of 5
        # add the hiker
        hiker_image_offset = (int(self.hiker_position[1] * 5), int(self.hiker_position[2]) * 5)
        # map[hiker_position[0]:hiker_position[0]+5,hiker_position[1]:hiker_position[1]+5,:] = self.hiker_image
        for pixel in self.hiker_stencil[self.hiker_heading][0]:
            map_image_with_actors[hiker_image_offset[0] + pixel[0], hiker_image_offset[1] + pixel[1], :] = hiker_color_adjusted

        # add the drone
        #drone_image_offset = np.where(
        #    self.map_volume['vol'] == self.map_volume['feature_value_map']['drone'][self.drone_altitude]['val'])
        #drone_image_offset = (int(drone_image_offset[1]) * 5, int(drone_image_offset[2]) * 5)

        for pixel in self.drone_stencil[self.drone_heading][0]:
            map_image_with_actors[drone_image_offset[0] + pixel[0], drone_image_offset[1] + pixel[1], :] = \
                self.map_dictionary['feature_value_map']['drone'][self.drone_altitude]['color']

        # maybe put the package in
        # print('pack drop flag',self.package_dropped)

        if self.package_dropped:
            self.package_dropped = 0

            package_position = (int(self.package_position[0] * 5), int(self.package_position[1]) * 5)
            for pixel in self.package[self.package_state][0]:
                # print(point, package_position)
                map_image_with_actors[package_position[0] + pixel[0], package_position[1] + pixel[1], :] = [94, 249, 242]

        # map[drone_position[0]:drone_position[0] + 5,drone_position[1]:drone_position[1] + 5] = self.plane_image(self.heading,self.map_volume['feature_value_map']['drone'][self.altitude]['color'])

        # map = imresize(map, (1000,1000), interp='nearest')

        '''vertical slices at drone's position'''
        drone_image_offset = np.where(
            self.map_dictionary['vol'] == self.map_dictionary['feature_value_map']['drone'][self.drone_altitude]['val'])

        nextstepimage = self.create_nextstep_image()

        obs['nextstepimage'] = nextstepimage
        obs['rgb_image_with_actors']           = map_image_with_actors
        #obs['image_layers']  = image_layers

        return obs


    def get_drone_heading(self):

        if self.params['use_mavsim_simulator']:
            return self.mavsimhandler.get_drone_heading()
        else:
            return self.drone_heading


    def get_drone_position(self):


        """returns the position of the drone as a tuple
           with altitude, row, col or (Z,Y,X)"""

        # The appearance (tile type) of the drone is altitude dependent so we need to look this up every time

        if self.params['use_mavsim_simulator']:

            return self.mavsimhandler.get_drone_position()

        else:
            drone_tile_id = self.map_dictionary['feature_value_map']['drone'][self.drone_altitude]['val']

            # Find tile that looks like the drone

            drone_zxy = np.where(self.map_dictionary['vol'] == drone_tile_id)

            return drone_zxy


    def get_hiker_position(self):

        """returns the position of the hiker as a tuple with altitude first and then x and y positions (Z,X,Y)"""
        # Here the appearance of the drone is altitude dependent so we need to look this up every time

        drone_tile_id = self.map_dictionary['feature_value_map']['hiker'][self.drone_altitude]['val']

        # Find tile that looks like the drone

        drone_zxy = np.where(self.map_dictionary['vol'] == drone_tile_id)

        return drone_zxy


    def render(self, mode='human', close=False):

        # return
        #if not self.params['verbose']:
        #   return
        # img = self.observation
        # map = self.original_map_volume['rgb_image_with_actors']
        obs = self.generate_observation()
        self.map_image = obs['rgb_image_with_actors']
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