# Fully Convolutional Network (the 2nd in DeepMind's paper)


from run_agent import Simulation
from run_agent import analyze_result

from run_agent import default_params,to_mavsim_actions,to_mavsim_rewards
from util import deep_update
from gym_gridworld.envs.gridworld_env import HEADING

"""First test of 1350 generated maps from Jacobs collection of ~2700"""

if __name__ == "__main__":


    test_map  = [ ( 460, 370 ) ]

    deep_update(
        default_params,
        {
            'run': {
                'training':False,
                'model_name':'parc_2019-07-29-F',
                'K_batches': 1,
                'sleep_time': 0.3,
                'verbose':False
            },

            'env': {
                'verbose':False,
                'map_path':'gym_gridworld/maps/nixel_maps_2',
                'submap_offsets':test_map,
                'episode_length':25,
                'render_hiker_altitude':False,
            },

            'agent' : {
                'action_neg_entropy_weight': 0.01
            }
            
        } )

    sim = Simulation( params = default_params )

    result = sim.run( { 'env': {'drone_initial_position':(6,3),
                                'drone_initial_heading': HEADING.WEST,
                                'drone_initial_altitude':2,

                                'hiker_initial_position':(6,6),
                                'hiker_initial_altitude':1 }     } )

    actions,rewards,n,statistics = analyze_result(result)

    print("Actions:")
    print(actions)

    print("Rewards")
    print(rewards)
