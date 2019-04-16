from gym.envs.registration import register

# You have to register ALL env-arg combos!!! Also each name should end with v[number]
for verbose in [True, False]:
    register(
        id='gridworld{}-v0'.format('visualize' if verbose else ''),
        kwargs={'verbose':verbose},
        entry_point='gym_gridworld.envs:GridworldEnv',
    )
