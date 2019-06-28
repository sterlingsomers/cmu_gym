# Fully Convolutional Network (the 2nd in DeepMind's paper)


from run_agent import Simulation
from run_agent import extract_episode_trajectory_as_dataframe

from gym_gridworld.envs.gridworld_env import HEADING
from gym_gridworld.envs.gridworld_env import ACTION

import pandas as pd



if __name__ == "__main__":

    sim = Simulation(
        training=True,
        verbose=False,
        model_name='parc_2019-06-27-C',
        goal_mode='navigate',
        K_batches=19001,
        episode_length=25,
        curriculum_radius=20,
        policy_type='FullyConv'
    )

    print("Training")
    sim.run()


    sim.close()


    print("Training complete")