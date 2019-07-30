from gym.envs.registration import register

register(
    id='gridworld-v0',
    kwargs={},
    entry_point='gym_gridworld.envs:GridworldEnv',
)
