# Fully Convolutional Network (the 2nd in DeepMind's paper)


from run_agent import Simulation
from run_agent import extract_all_episodes, augment_dataframe_with_reward_detail, aggregate_rewards
from run_agent import to_mavsim_actions
from run_agent import to_mavsim_rewards
from gym_gridworld.envs.gridworld_env import HEADING
from gym_gridworld.envs.gridworld_env import ACTION

import pandas as pd



if __name__ == "__main__":


    episodes_to_run = 500

    print("Evaluation")

    sim = Simulation(
        training=False,
        verbose=True,
        model_name='parc_2019-06-25-A',
        curriculum_radius=9,
        goal_mode='navigate',
        K_batches=200,
        episode_length=17
    )


    result = sim.run(
        episodes_to_run = episodes_to_run,
        sleep_time=0,
    )


    #print("Drone initial position {} Hiker Initial Position {}".format(drone_initial_position,hiker_initial_position))
    print("Evaluation on {} episodes".format(episodes_to_run))

    all_data_df = extract_all_episodes(result)

    all_data_with_reward_df = augment_dataframe_with_reward_detail(all_data_df)


    pd.options.display.width = 0
    #print(all_data_with_reward_df)

    episode_reward_sums = aggregate_rewards(all_data_with_reward_df)
    print(episode_reward_sums)

    reward_stats = episode_reward_sums.mean(axis=0).to_dict()
    print("Average accumulated reward per episode")
    print(reward_stats)


    actions = to_mavsim_actions(all_data_df)
    print("Mavsim actions")
    print(actions)

    rewards = to_mavsim_rewards(all_data_df)
    print("Mavsim rewards")
    print(rewards)