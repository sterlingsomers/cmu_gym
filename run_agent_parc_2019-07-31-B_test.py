# Fully Convolutional Network (the 2nd in DeepMind's paper)


from run_agent import Simulation
from run_agent import analyze_result

from run_agent import default_params,to_mavsim_actions,to_mavsim_rewards
from util import deep_update
from gym_gridworld.envs.gridworld_env import HEADING

import glob
from convert_nparray_pickles_to_cmu_maps import pathname_to_offset

"""First test of 1350 generated maps from Jacobs collection of ~2700"""

if __name__ == "__main__":



    map_path = 'gym_gridworld/maps/nixel_maps_2'
    map_filenames = glob.glob(map_path + '/*.mp')
    offsets = [pathname_to_offset(filename) for filename in map_filenames]

    # This is repeatable for debugging purposes whereas a random choice would not be

    train_maps = offsets[::2]   # Even maps
    test_maps  = offsets[1::2]  # Odd maps
    #test_maps  = [ ( 460, 370 ) ]

    deep_update(
        default_params,
        {
            'run': {
                'training':False,
                'model_name':'parc_2019-07-31-B',
                'K_batches': 20,
                'verbose':False,
                'show_pygame_display': True,
                'sleep_time': 0.3,
            },

            'env': {
                'verbose':False,
                'map_path':'gym_gridworld/maps/nixel_maps_2',
                'submap_offsets':test_maps,
                'episode_length':25,
                'render_hiker_altitude':True,
            },

            'agent' : {
                'action_neg_entropy_weight': 0.01
            }
            
        } )

    sim = Simulation( params = default_params )\

    parms = { 'env': {'drone_initial_position':(6,3),
                      'drone_initial_heading': HEADING.WEST,
                      'drone_initial_altitude':2,

                      'hiker_initial_position':(4,17),
                      'hiker_initial_altitude':1 }
            }

    #result = sim.run( parms  )
    result = sim.run(  )

    actions,rewards,n,statistics = analyze_result(result)

    print("Actions:")
    print(actions)

    print("Rewards")
    print(rewards)
