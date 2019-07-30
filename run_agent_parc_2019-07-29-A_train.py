# Fully Convolutional Network (the 2nd in DeepMind's paper)


from run_agent import Simulation
from run_agent import extract_episode_trajectory_as_dataframe

from gym_gridworld.envs.gridworld_env import HEADING
from gym_gridworld.envs.gridworld_env import ACTION

import pandas as pd
from run_agent import default_params
from util import deep_update

"""First test of 1000 generated maps from Jacobs collection of ~2700"""

if __name__ == "__main__":


    train_maps = [ ( 999, i   ) for i in range( 0, 2700, 2 ) ]
    test_maps  = [ ( 999, i+1 ) for i in range( 0, 2700, 2 ) ]


    deep_update( default_params,
                 {

                    'run': {
                        'model_name':'parc_2019-07-29-D',
                        'training': True,
                        'verbose': False,
                        'K_batches': 1001,
                        'n_envs':10
                    },

                    'env': {
                        'map_path': 'gym_gridworld/maps/nixel_maps',
                        'submap_offsets':train_maps,
                        'episode_length':25,
                        'verbose':False
                    },

                    'agent' : {
                        'action_neg_entropy_weight': 0.01
                    }

                } )


    sim = Simulation(default_params)

    print("Training")

    for i in range(2,21):
        print("run_agent_parc_* starting new outer loop with curriculm_radius {}".format(i))
        result = sim.run( param_updates={ 'env':{'curriculum_radius':i+1} } )

    sim.close()

    print("Training complete")