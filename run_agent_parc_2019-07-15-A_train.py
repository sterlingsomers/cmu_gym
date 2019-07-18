# Fully Convolutional Network (the 2nd in DeepMind's paper)


from run_agent import Simulation
from run_agent import extract_episode_trajectory_as_dataframe

from gym_gridworld.envs.gridworld_env import HEADING
from gym_gridworld.envs.gridworld_env import ACTION

import pandas as pd

"""First test of 1000 generated maps from Jacobs collection of ~2700"""

if __name__ == "__main__":


    train_maps = [ (999,i  ) for i in range( 0, 2700, 2 ) ]

    sim = Simulation(
        training=True,
        verbose=False,
        model_name='parc_2019-07-15-A',
        goal_mode='navigate',
        K_batches=1001,
        episode_length=25,
        policy_type='DeepFullyConv',
        map_path='gym_gridworld/maps/nixel_maps/',
        submap_offsets=train_maps
    )


    print("Training")

    for i in range(2,21):
        result = sim.run(curriculum_radius=i)

    sim.close()

    print("Training complete")