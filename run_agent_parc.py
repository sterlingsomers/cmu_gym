# Fully Convolutional Network (the 2nd in DeepMind's paper)

import pandas as pd

from run_agent import Simulation

def extract_trajectory(result):

    traj_all_columns = result[0]['nav']
    traj_key_columns = [ (  ( step['drone'][0][0] ,step['drone'][1][0]),
                            step['altitude'],
                            step['action'],
                            step['reward']   )
                         for step in traj_all_columns]

    return traj_key_columns

def list_of_tuples_to_dataframe(list_of_tuples):

    df  = pd.DataFrame(list_of_tuples,columns=['location','altitude','action','reward'])
    return df


if __name__ == "__main__":

    sim = Simulation()

    drone_initial_position = (5,5)
    hiker_initial_position = (3,3)

    result = sim.run(
        episodes_to_run = 1,
        drone_initial_position = drone_initial_position,
        hiker_initial_position = hiker_initial_position,
        sleep_time=0
    )

    print("Drone initial position {} Hiker Initial Position {}")
    print("Trajectory")

    key_columns = extract_trajectory(result)
    data_frame = list_of_tuples_to_dataframe(key_columns)

    print(data_frame)

