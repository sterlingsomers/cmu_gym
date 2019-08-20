# cmu_drone

Includes A2C actor critic implementation, internal gridworld simulation 
and an interface to external PARC MAVSim simulator.
It also comes with 2700 sample worlds for training using the 
in built gridworld simulator.

# MAVSim integration

The MAVSim world allows the generation of a wide variety of interesting
highly textured worlds to support training on diverse inputs.
This should help generalization considerably.

To use mavsim integration you need to checkout the mavsim library.
Currently we are using the 'develop' branch.
Wihtin your chosen environment (anaconda, virtualenv)
simply do a 

    python setup.py install
    
CMU gridworld will then be able to talk to MAVsim directly.

# Setting up experiments and using models

Control of the cmu gridworld system is done through nested dictionaries.
The default dictionary is in run_agent.py.

Sample experiments such as run_agent_parc_2019-07-31-B.test.py
send in partial dictionaries whose entries overwrite overlapping 
entries in the default dictionary.


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
