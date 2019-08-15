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

"""First test of 1000 generated maps from Jacobs collection of ~2700"""

if __name__ == "__main__":

    map_path = 'gym_gridworld/maps/nixel_maps_2'
    map_filenames = glob.glob(map_path + '/*.mp')
    offsets = [pathname_to_offset(filename) for filename in map_filenames]

    # This is repeatable for debugging purposes whereas a random choice would not be

    train_maps = offsets[::2]   # Even maps
    test_maps  = offsets[1::2]  # Odd maps


    deep_update( default_params,
                 {

                    'run': {
                        'model_name':'parc_2019-07-31-B',
                        'training': True,
                        'verbose': False,
                        'K_batches': 2000,
                        'n_envs':10
                    },

                    'env': {
                        'map_path': map_path,
                        'submap_offsets':train_maps,
                        'episode_length':30,
                        'verbose':False,
                        'align_drone_and_hiker_heading':True,
                        'align_drone_and_hiker_altitude':True,
                        'render_hiker_altitude':True,
                    },

                    'agent' : {
                        'action_neg_entropy_weight': 0.01
                    }

                } )


    sim = Simulation(default_params)

    print("Training")

    for i in range(2,23):
        print("run_agent_parc_* starting new outer loop with curriculm_radius {}".format(i))
        result = sim.run( param_updates={ 'env':{'curriculum_radius':i+1} } )

    sim.close()

    print("Training complete")