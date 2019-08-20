# Fully Convolutional Network (the 2nd in DeepMind's paper)


from run_agent import Simulation
from run_agent import analyze_result

from run_agent import default_params,to_mavsim_actions,to_mavsim_rewards
from util import deep_update
from gym_gridworld.envs.gridworld_env import HEADING
import itertools

"""First test of 1350 generated maps from Jacobs collection of ~2700"""

if __name__ == "__main__":


    map_ordinates = [ i for i in range(0,490,10) ]
    test_map = [ p for p in itertools.product(map_ordinates,map_ordinates) ]

    num_maps = len(test_map)

    deep_update(
        default_params,
        {
            'run': {
                'training':False,
                'model_name':'parc_2019-07-31-B',
                'K_batches': 5,
                'sleep_time': 0,
                'verbose':False
            },

            'env': {
                'verbose':False,
                'submap_offsets':test_map,
                'episode_length':25,
                'render_hiker_altitude':True,
                'use_mavsim_simulator':True,
                #'mavsim_scenario':"['COGLE_0:stubland_1:512_2:512_3:256_4:7_5:24|-0.1426885426044464/Terrain_0:0_1:100_2:0.05_3:0.5_4:0.05_5:0.5_6:0.05_7:0.5_8:0.5_9:0.5_10:0.7_11:0.3_12:0.5_13:0.5_14:True/', '0.36023542284965515/Ocean_0:60/', '-0.43587446212768555/River_0:0.01_1:100/', '-0.3501245081424713/Tree_0:500_1:20.0_2:4.0_3:0.01_4:2.0_5:0.1_6:1.9_7:3.0_8:2.2_9:3.5/', '0.6151155829429626/Airport_0:15.0_1:25_2:35_3:1000_4:[]/', '0.34627288579940796/Building_0:150_1:10.0_2:[]_3:1/', '0.31582069396972656/Road_0:3_1:500/', '-0.061891376972198486/DropPackageMission_0:1_3:Find the hiker last located at (88, 186, 41)_4:Provision the hiker with Food_5:Return and report to Southeast International Airport (SEI) airport_6:Southeast Regional Airport_7:Southeast International Airport_8:0_9:20.0_10:20.0_11:40.0/', '-0.25830233097076416/Stub_0:0.8_1:1.0_2:1.0_3:1.0_4:1.0_5:1.0/']"
            },

            'agent' : {
                'action_neg_entropy_weight': 0.01
            }
            
        } )

    sim = Simulation( params = default_params )

    param = { 'env': {'drone_initial_position':(5,10),
                      'drone_initial_heading': HEADING.WEST,
                      'drone_initial_altitude':2,

                      'hiker_initial_position':(6,6),
                      'hiker_initial_altitude':1 }     }


    result = sim.run(  )

    actions,rewards,n,statistics = analyze_result(result)

    print("Statistics:")
    print(statistics)

    print("Actions:")
    print(actions)

    print("Rewards")
    print(rewards)
