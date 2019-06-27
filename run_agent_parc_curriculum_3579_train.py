# Fully Convolutional Network (the 2nd in DeepMind's paper)


from run_agent import Simulation
from run_agent import extract_episode_trajectory_as_dataframe

from gym_gridworld.envs.gridworld_env import HEADING
from gym_gridworld.envs.gridworld_env import ACTION

import pandas as pd



if __name__ == "__main__":


    drone_initial_position = (7,7)
    drone_initial_heading  = HEADING.EAST
    drone_initial_altitude = 2

    hiker_initial_position = (3,3)

    sim = Simulation(
        training=True,
        verbose=False,
        model_name='parc_curriculum_1000_at_3-5-7-9',
        curriculum_radius=3,
        goal_mode='navigate',
        K_batches=1500,
        episode_length=10
    )

    print("Initial run radius 3")

    sim.curriculum_radius=3
    episode_length=10

    result = sim.run(
        sleep_time=0,
    )

    print("Run radius 5")

    sim.curriculum_radius=5
    episode_length=13

    result = sim.run(
        sleep_time=0,
    )


    print("Run radius 7")
    sim.curriculum_radius=7
    episode_length=15


    result = sim.run(
        sleep_time=0,
    )

    print("Run radius 9")
    sim.curriculum_radius=9
    episode_length=17


    result = sim.run(
        sleep_time=0,
    )

    sim.close()


    print("Training complete")