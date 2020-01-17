from gym.envs.registration import register

for i in range(10):
    for stochastic in [True, False]:
        register(
            id='Gridworld{}{}-v1'.format(i, 'Stochastic' if stochastic else ''), # The v0 or v1 cannot be smth else e.g. -image, has to be a version
            kwargs={'plan': i, 'stochastic': stochastic},
            entry_point='gym_grid_im.envs.gridworld_env:GridworldEnv', # folder/folder/file:class
        )
# e.g. for the plan1.txt and plan1s.txt you gym.make(Gridworld1-v0) or Gridworld1s-v0

#(MINE)
# register(
#     id='Grid-v0',
#     entry_point='gym_gridworld_custom.gym_grid.envs.gridworld_env:GridworldEnv',
#     kwargs={'plan': 1}, # Run env with arguments (e.g. for vizualization)
# )

# register(
#     id='SequentialGridworld{}-v0'.format('Stochastic'),
#     kwargs={'plans': list(range(1, 6)), 'stochastic': True},
#     entry_point='gym_gridworld.envs:SequentialGridworldEnv',
# )