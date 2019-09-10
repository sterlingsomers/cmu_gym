# Fully Convolutional Network (the 2nd in DeepMind's paper)


from run_agent import Simulation
from run_agent import analyze_result

from gym_gridworld.envs.gridworld_env import HEADING
from gym_gridworld.envs.gridworld_env import ACTION

import pandas as pd
from run_agent import default_params
from util import deep_update
import glob
from convert_nparray_pickles_to_cmu_maps import pathname_to_offset
import itertools

"""First test of 1000 generated maps from Jacobs collection of ~2700"""

if __name__ == "__main__":

    map_path = 'gym_gridworld/maps/nixel_maps_2'  # Used for cmu_drone gridworld_env engine, not used for mavsim

    step = 20
    map_ordinates = [ i for i in range(0,490,step) ]
    all_maps = [ p for p in itertools.product(map_ordinates,map_ordinates) ]
    num_maps = len(all_maps)
    print("Total number of maps {}".format(num_maps))

    # Because we are using a step of size 20, there will be zero overlap between train and test sets.
    # This should lead to more accurate generalization predictions.

    # Use 3/4=75% of data for training and 1/4=25% for testing.
    # We should get (480/20+1)^2 = 625 maps with 450 for training and 175 for testing


    train_maps = [ (x,y) for (x,y) in all_maps if (x//step)%4 > 0  ]
    test_maps  = [ (x,y) for (x,y) in all_maps if (x//step)%4 == 0 ]

    num_train = len(train_maps)
    num_test = len(test_maps)

    print("Number of training maps: {} ".format(num_train))
    print("Number of testing maps: {} ".format(num_test))

    params =     {

                    'run': {
                        'model_name':'parc_2019-09-05-A',
                        'training': False,
                        'verbose': False,
                        'K_batches': 500,
                        'n_envs':1,
                        'policy_type':'DeepDensePolicy',
                        'sleep_time': 0.0,#5,
                        'use_keyboard_control':True,
                    },

                    'env': {

                        'submap_offsets':test_maps,
                        'map_path': map_path,  # Used by cmu gridworld_env to access sampled maps, not used by mavsim
                        'episode_length':40,
                        'verbose':True,
                        'align_drone_and_hiker_heading':True,
                        'align_drone_and_hiker_altitude':True,
                        'render_hiker_altitude':True,
                        'use_mavsim_simulator':True,
                        'mavsim':{
                            # https://gitlab.com/COGLEProject/mavsim/tree/develop
                            # git rev-parse origin/develop -> 7bc21639c19d6561906d7a65882406092f179b89
                            'verbose': True,
                            'show_global_map':False,
                            'halt_on_error': True
                        },
                    },

                    'agent' : {
                        'action_neg_entropy_weight': 0.01,
                        'use_egocentric':False,
                        'use_additional_fully_connected':False
                    }

                }



    sim = Simulation( params )

    params = { 'env': {
                      'submap_offsets': [ (160,180) ], # Row/Y, Col

                      'drone_initial_position': (7,3),  # Row/Y, column/X in local coordinates
                      #'drone_initial_heading': HEADING.NORTH_EAST,
                      'drone_initial_altitude': 3,

                      'hiker_initial_position': (7,7),  # Row/Y, column/X
                      'hiker_initial_altitude': 3
                     }
            }


    result = sim.run( params   )

    actions,rewards,n,statistics = analyze_result(result)

    print("Statistics:")
    print(statistics)

    print("Actions:")
    print(actions)

    print("Rewards")
    print(rewards)
