# Fully Convolutional Network (the 2nd in DeepMind's paper)


from run_agent import Simulation
from run_agent import analyze_result

from run_agent import default_params


"""First test of 1000 generated maps from Jacobs collection of ~2700"""

if __name__ == "__main__":


    train_maps = [ ( 999, i   ) for i in range( 0, 2700, 2 ) ]
    test_maps  = [ ( 999, i+1 ) for i in range( 0, 2700, 2 ) ]


    default_params['run']['model_name']='parc_2019-07-26-A'
    env = default_params['env']
    env['map_path']= 'gym_gridworld/maps/nixel_maps'
    env['submap_offsets']=test_maps
    env['episode_length']=25
    env['curriculum_radius']=25

    episodes_to_run = 500


    sim = Simulation(
        params = default_params,
    )

    default_params['env']['curriculum_radius']=20

    result = sim.run(
        episodes_to_run = episodes_to_run,
        sleep_time=0,
    )

    analysis = analyze_result(result)