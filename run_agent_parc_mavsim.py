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

    hiker_initial_position = (6,7)

    sim = Simulation(        training=False,
                             verbose=True,
                             model_name='parc_curriculum',
                             curriculum_radius=3,
                             goal_mode='navigate',
                             episode_length=9,
                             use_mavsim=True)



    result = sim.run(
        episodes_to_run = 1,
        sleep_time=1,
    )

    #print("Drone initial position {} Hiker Initial Position {}".format(drone_initial_position,hiker_initial_position))
    print("Trajectory")

    data_frame = extract_episode_trajectory_as_dataframe(result)
    pd.options.display.width = 0
    print(data_frame)

