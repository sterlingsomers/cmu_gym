# Fully Convolutional Network (the 2nd in DeepMind's paper)


from run_agent import Simulation
from run_agent import extract_episode_trajectory_as_dataframe

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

    deep_update( default_params,
                 {

                    'run': {
                        'model_name':'parc_2019-08-19-A',
                        'training': True,
                        'verbose': False,
                        'K_batches': 100, # Was 2000
                        'n_envs':1 # was 10
                    },

                    'env': {
                        'submap_offsets':train_maps,
                        'map_path': map_path,  # Used by cmu gridworld_env to access sampled maps, not used by mavsim
                        'episode_length':30,
                        'verbose':False,
                        'align_drone_and_hiker_heading':True,
                        'align_drone_and_hiker_altitude':True,
                        'render_hiker_altitude':True,
                        'use_mavsim_simulator':True,
                        'mavsim':{
                            'verbose': False,
                            'halt_on_error': False
                        },
                    },

                    'agent' : {
                        'action_neg_entropy_weight': 0.01
                    }

                } )


    sim = Simulation(default_params)

    print("Training")

    for i in range(2,23):
        print("---------------------------------------------------------------------------------")
        print("run_agent_parc_* starting new outer loop with curriculm_radius {}".format(i))
        print("---------------------------------------------------------------------------------")
        result = sim.run( param_updates={ 'env':{'curriculum_radius':i+1} } )

    sim.close()

    print("Training complete")