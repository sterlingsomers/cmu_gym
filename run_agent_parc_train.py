# Fully Convolutional Network (the 2nd in DeepMind's paper)


from run_agent import Simulation
from run_agent import extract_trajectory
from run_agent import list_of_tuples_to_dataframe

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
        visualize=False,
        curriculum_radius=3,
        goal_mode='navigate',
        episode_length=9

    )

    result = sim.run(
        episodes_to_run = 1,
        sleep_time=0,
    )

    print("Drone initial position {} Hiker Initial Position {}".format(drone_initial_position,hiker_initial_position))
    print("Trajectory")

    key_columns = extract_trajectory(result)
    data_frame = list_of_tuples_to_dataframe(key_columns)
    pd.options.display.width = 0
    print(data_frame)

