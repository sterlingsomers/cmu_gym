# Fully Convolutional Network (the 2nd in DeepMind's paper)


from run_agent import Simulation
from run_agent import analyze_result

from run_agent import default_params
from util import deep_update

"""First test of 1350 generated maps from Jacobs collection of ~2700"""

if __name__ == "__main__":


    train_maps = [ ( 999, i   ) for i in range( 0, 2700, 2 ) ]
    test_maps  = [ ( 999, i+1 ) for i in range( 0, 2700, 2 ) ]

    deep_update(
        default_params,

        {
            'run': {
                'training': False,
                'model_name': 'parc_2019-07-29-F',
                'K_batches': 500,

                'show_pygame_display': False,
                'sleep_time': 0.0, # 0.3 is good for watching
            },

            'env': {
                'verbose':False,
                'map_path':'gym_gridworld/maps/nixel_maps',
                'submap_offsets':test_maps,
                'episode_length':25,
                'align_drone_and_hiker_heading':True,
                'align_drone_and_hiker_altitude':True,
                'render_hiker_altitude':False,
            },

            'agent' : {
                'action_neg_entropy_weight': 0.01
            }
            
        } )

    sim = Simulation( params = default_params )

    result = sim.run()

    actions,rewards,n,statistics = analyze_result(result)

    print("Statistics on {} episodes".format(n))
    print(statistics)


