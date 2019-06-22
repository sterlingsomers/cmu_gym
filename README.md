# gym-gridworld

Basic implementation of gridworld game 
for reinforcement learning research. 

## Install gym-gridworld

install virtual environment for gridworld

    cd gym-gridworld
    conda env create -f environment.yml
    source gridworld
    pip install -e .

## Use gym-gridworld
    
    import gym
    import gym_gridworld
    env = gym.make('gridworld-v0')
    _ = env.reset()
    _ = env.step(env.action_space.sample())
    
## Visualize gym-gridworld
In order to visualize the gridworld, you need to set `env.verbose` to `True`

    env.verbose = True
    _ = env.reset()


## Architectural Notes

* The Simulation class creates an abstraction over an experimental context. What is the goal, learning rate, number of steps, etc.
  It embodies the driving loop and visualization.
  
* The GridworldEnv class manages the world dynamics, episode termination and the reward function. 
   * The world dynamics can be split into two depending on whether they come from the MAVSim simulator or the CMU gridworld.
   * The reward always comes from the CMU gridworld.
   * The episode control always comes from the CMU gridworld.
