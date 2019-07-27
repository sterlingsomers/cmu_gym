# Fully Convolutional Network (the 2nd in DeepMind's paper)


from run_agent import Simulation
from run_agent import extract_episode_trajectory_as_dataframe

from gym_gridworld.envs.gridworld_env import HEADING
from gym_gridworld.envs.gridworld_env import ACTION

import pandas as pd
from run_agent import default_params

"""First test of 1000 generated maps from Jacobs collection of ~2700"""

if __name__ == "__main__":


    train_maps = [ ( 999, i   ) for i in range( 0, 2700, 2 ) ]
    test_maps  = [ ( 999, i+1 ) for i in range( 0, 2700, 2 ) ]


    run = default_params['run']

    run['model_name'] = 'parc_2019-07-26-A'
    run['training']   = True
    run['verbose']    = False

    env = default_params['env']

    env['map_path']       = 'gym_gridworld/maps/nixel_maps'
    env['submap_offsets'] = train_maps


    sim = Simulation(params=default_params)

    print("Training")

    for i in range(2,21):
        default_params['env']['curriculum_radius'] = i
        default_params['env']['episode_length']    = 5+i
        result = sim.run()

    sim.close()

    print("Training complete")