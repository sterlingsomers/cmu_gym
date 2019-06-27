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
        model_name='parc_2019-06-25-C',
        curriculum_radius=2,
        goal_mode='navigate',
        K_batches=1003,
        episode_length=17
    )

    print("Training")


    sim.curriculum_radius=2
    result = sim.run()

    #sim.curriculum_radius=3
    #result = sim.run()

    #sim.curriculum_radius=4
    #result = sim.run()


    sim.close()


    print("Training complete")