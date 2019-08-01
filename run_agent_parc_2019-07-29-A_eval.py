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
                'training':False,
                'model_name':'parc_2019-07-29-D',
                'episodes_to_run': 500,
                'sleep_time': 0.3
            },

            'env': {
                'verbose':True,
                'map_path':'gym_gridworld/maps/nixel_maps',
                'submap_offsets':test_maps,
                'episode_length':25
            },

            'agent' : {
                'action_neg_entropy_weight': 0.01
            }
            
        } )

    sim = Simulation( params = default_params )

    result = sim.run()

    analysis = analyze_result(result)