# Fully Convolutional Network (the 2nd in DeepMind's paper)
import logging
import os
import shutil
import sys
from datetime import datetime
import time
from time import sleep
import numpy as np
import pickle
#from functools import partial

from absl import flags
from actorcritic.agent import ActorCriticAgent, ACMode
from actorcritic.runner import Runner, PPORunParams
# from common.multienv import SubprocVecEnv, make_sc2env, SingleEnv, make_env

import tensorflow as tf
# import tensorflow.contrib.eager as tfe
# tf.enable_eager_execution()

#import argparse
from baselines import logger
from baselines.bench import Monitor
#from baselines.common.misc_util import boolean_flag
from baselines.common import set_global_seeds
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
#from baselines.common.vec_env.vec_normalize import VecNormalize
import gym
#from gym_grid.envs import GridEnv
# import gym_gridworld
from gridworld_v2 import gameEnv

FLAGS = flags.FLAGS
flags.DEFINE_bool("visualize", False, "Whether to render with pygame.")
flags.DEFINE_float("sleep_time", 0.0, "Time-delay in the demo")
flags.DEFINE_integer("resolution", 8, "Resolution for screen and minimap feature layers.")
flags.DEFINE_integer("step_mul", 100, "Game steps per agent step.")
flags.DEFINE_integer("step2save", 500, "Game step to save the model.") #A2C every 1000, PPO 250
flags.DEFINE_integer("n_envs", 80, "Number of environments to run in parallel")
flags.DEFINE_integer("episodes", 500, "Number of complete episodes")
flags.DEFINE_integer("n_steps_per_batch", 32,
    "Number of steps per batch, if None use 8 for a2c and 128 for ppo")  # (MINE) TIMESTEPS HERE!!! You need them cauz you dont want to run till it finds the beacon especially at first episodes - will take forever
flags.DEFINE_integer("all_summary_freq", 10, "Record all summaries every n batch")
flags.DEFINE_integer("scalar_summary_freq", 5, "Record scalar summaries every n batch")
flags.DEFINE_string("checkpoint_path", "_files/models", "Path for agent checkpoints")
flags.DEFINE_string("summary_path", "_files/summaries", "Path for tensorboard summaries") #A2C_custom_maps#A2C-science-allmaps - BEST here for one policy
flags.DEFINE_string("model_name", "dokimib", "Name for checkpoints and tensorboard summaries") # DONT touch TESTING is the best (take out normalization layer in order to work! -- check which parts exist in the restore session if needed)
flags.DEFINE_integer("K_batches", 4000, # Batch is like a training epoch!
    "Number of training batches to run in thousands, use -1 to run forever") #(MINE) not for now
flags.DEFINE_string("map_name", "DefeatRoaches", "Name of a map to use.")
flags.DEFINE_float("discount", 0.95, "Reward-discount for the agent")
flags.DEFINE_boolean("training", True,
    "if should train the model, if false then save only episode score summaries"
)
flags.DEFINE_enum("if_output_exists", "continue", ["fail", "overwrite", "continue"],
    "What to do if summary and model output exists, only for training, is ignored if notraining")
flags.DEFINE_float("max_gradient_norm", 10.0, "good value might depend on the environment") # orig: 1000
flags.DEFINE_float("loss_value_weight", 0.5, "good value might depend on the environment") # orig:1.0, good value: 0.5
flags.DEFINE_float("entropy_weight_spatial", 0.00000001,
    "entropy of spatial action distribution loss weight") # orig:1e-6
flags.DEFINE_float("entropy_weight_action", 0.01, "entropy of action-id distribution loss weight") # orig:1e-6
flags.DEFINE_float("ppo_lambda", 0.95, "lambda parameter for ppo")
flags.DEFINE_integer("ppo_batch_size", None, "batch size for ppo, if None use n_steps_per_batch")
flags.DEFINE_integer("ppo_epochs", 3, "epochs per update")
flags.DEFINE_enum("policy_type", "FactoredPostTraining", ["MetaPolicy", "FullyConv", "FactoredPolicy", "FactoredPostTraining", "Relational", "AlloAndAlt", "FullyConv3D"], "Which type of Policy to use")
flags.DEFINE_enum("agent_mode", ACMode.A2C, [ACMode.A2C, ACMode.PPO], "if should use A2C or PPO")

### NEW FLAGS ####
flags.DEFINE_integer("rgb_screen_size", 128,
                        "Resolution for rendered screen.") # type None if you want only features

flags.DEFINE_integer("max_agent_steps", 0, "Total agent steps.")
flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
#flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")

#flags.DEFINE_string("map", None, "Name of a map to use.")


FLAGS(sys.argv)

#TODO this runner is maybe too long and too messy..
full_chekcpoint_path = os.path.join(FLAGS.checkpoint_path, FLAGS.model_name)

if FLAGS.training:
    full_summary_path = os.path.join(FLAGS.summary_path, FLAGS.model_name)
else:
    full_summary_path = os.path.join(FLAGS.summary_path, "no_training", FLAGS.model_name)


def check_and_handle_existing_folder(f):
    if os.path.exists(f):
        if FLAGS.if_output_exists == "overwrite":
            shutil.rmtree(f)
            print("removed old folder in %s" % f)
        elif FLAGS.if_output_exists == "fail":
            raise Exception("folder %s already exists" % f)


def _print(i):
    print(datetime.now())
    print("# batch %d" % i)
    sys.stdout.flush()


def _save_if_training(agent):
    agent.save(full_chekcpoint_path)
    agent.flush_summaries()
    sys.stdout.flush()

def make_custom_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            # env = gym.make(env_id)
            env = gym.wrappers.TimeLimit(gameEnv(partial=False,size=9))#,goal_color=[np.random.uniform(), np.random.uniform(), np.random.uniform()])
            env._max_episode_steps = 500 # ONLY FOR GYM ENVS
            env.seed(seed + rank)
            # Monitor should take care of reset!
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True) # SUBPROC NEEDS 4 OUTPUS FROM STEP FUNCTION
            return env
        return _thunk
    #set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

def main():
    if FLAGS.training:
        check_and_handle_existing_folder(full_chekcpoint_path)
        check_and_handle_existing_folder(full_summary_path)

    #(MINE) Create multiple parallel environements (or a single instance for testing agent)
    if FLAGS.training and FLAGS.visualize==False:
        #envs = SubprocVecEnv((partial(make_sc2env, **env_args),) * FLAGS.n_envs)
        #envs = SubprocVecEnv([make_env(i,**env_args) for i in range(FLAGS.n_envs)])
        # envs = make_custom_env('gridworld{}-v3'.format('visualize' if FLAGS.visualize else ''), FLAGS.n_envs, 1)
        envs = make_custom_env('MsPacman-v4', FLAGS.n_envs, 1)
    elif FLAGS.training==False:
        #envs = make_custom_env('gridworld-v0', 1, 1)
        envs = gym.make('gridworld{}-v0'.format('visualize' if FLAGS.visualize else ''))
    else:
        print('Wrong choices in FLAGS training and visualization')
        return

    print("Requested environments created successfully")

    tf.reset_default_graph()
    # The following lines fix the problem with using more than 2 envs!!!
    # config = tf.ConfigProto(allow_soft_placement=True,
    #                                     log_device_placement=True) # first option allows to use with device:cpu so some operations go there and second option shows all operations where do they go
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    #sess = tf.Session()


    agent = ActorCriticAgent(
        mode=FLAGS.agent_mode,
        sess=sess,
        spatial_dim=FLAGS.resolution, # Here you pass the resolution which is used in the step for the output probabilities map
        unit_type_emb_dim=5,
        loss_value_weight=FLAGS.loss_value_weight,
        entropy_weight_action_id=FLAGS.entropy_weight_action,
        entropy_weight_spatial=FLAGS.entropy_weight_spatial,
        scalar_summary_freq=FLAGS.scalar_summary_freq,
        all_summary_freq=FLAGS.all_summary_freq,
        summary_path=full_summary_path,
        max_gradient_norm=FLAGS.max_gradient_norm,
        num_actions=envs.action_space.n,
        num_envs= FLAGS.n_envs,
        nsteps= FLAGS.n_steps_per_batch,
        obs_dim= envs.observation_space.shape,
        policy=FLAGS.policy_type
    )
    # Build Agent
    agent.build_model()
    if os.path.exists(full_chekcpoint_path):
        if FLAGS.policy_type == 'FactoredPostTraining':
            agent.init() # Initialize all the variables, so the ones that are not loaded are getting initialized.
        agent.load(full_chekcpoint_path) #We load the variables so hopefully the initialisation is substituted.
        # TODO: You can run a self.sess(run) with a custom saver s = tf.train.saver(vars) and then use the code of agent.load with the vars you want.
        # Initializing specific variables
        # head_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "theta/heads")
        # unit_vs = tf.initializers.variables(head_train_vars)
        # agent.sess.run(unit_vs)
    else:
        agent.init()
# (MINE) Define TIMESTEPS per episode (batch as each worker has its own episodes -- different timelines)
    # If it is not training you don't need that many steps. You need one to take the decision...Actually seem to be game steps
    if FLAGS.n_steps_per_batch is None:
        n_steps_per_batch = 128 if FLAGS.agent_mode == ACMode.PPO else 8
    else:
        n_steps_per_batch = FLAGS.n_steps_per_batch

    if FLAGS.agent_mode == ACMode.PPO:
        ppo_par = PPORunParams(
            FLAGS.ppo_lambda,
            batch_size=FLAGS.ppo_batch_size or n_steps_per_batch,
            n_epochs=FLAGS.ppo_epochs
        )
    else:
        ppo_par = None

    runner = Runner(
        envs=envs,
        agent=agent,
        discount=FLAGS.discount,
        n_steps=n_steps_per_batch,
        do_training=FLAGS.training,
        ppo_par=ppo_par,
        policy_type = FLAGS.policy_type,
        n_envs= FLAGS.n_envs
    )

    # runner.reset() # Reset env which means you get first observation. You need reset if you run episodic tasks!!! SC2 is not episodic task!!!

    if FLAGS.K_batches >= 0:
        n_batches = FLAGS.K_batches  # (MINE) commented here so no need for thousands * 1000
    else:
        n_batches = -1


    if FLAGS.training:
        i = 0

        try:
            runner.reset()
            t=0
            start_time = time.time()
            while True:
                # For below you have to enable monitor's early reset. You might want to take out Monitor
                # if t==390:
                #     runner.reset()
                #     t=0

                if i % 500 == 0:
                    _print(i)
                if i % FLAGS.step2save == 0:
                    _save_if_training(agent)
                if FLAGS.policy_type == 'MetaPolicy' or FLAGS.policy_type == 'LSTM':
                    runner.run_meta_batch()
                elif (FLAGS.policy_type == 'FactoredPolicy') or (FLAGS.policy_type == 'FactoredPostTraining'):
                    runner.run_factored_batch()
                else:
                    runner.run_batch()  # (MINE) HERE WE RUN MAIN LOOP for while true
                #runner.run_batch_solo_env()
                i += 1
                if 0 <= n_batches <= i: #when you reach the certain amount of batches break
                    break
                # t=t+1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 60:
                print("--- %s minutes ---" % (np.round(elapsed_time/60,2))  )
            else:
                print("--- %s seconds ---" % (np.round(elapsed_time, 2)))
        except KeyboardInterrupt:
            pass
    else: # Test the agent
        try:
            import pygame
            # import time
            import random
            # pygame.font.get_fonts() # Run it to get a list of all system fonts
            display_w = 1200
            display_h = 720

            BLUE = (128, 128, 255)
            DARK_BLUE = (1, 50, 130)
            RED = (255, 192, 192)
            BLACK = (0, 0, 0)
            WHITE = (255, 255, 255)

            sleep_time = FLAGS.sleep_time

            pygame.init()
            gameDisplay = pygame.display.set_mode((display_w, display_h))
            gameDisplay.fill(DARK_BLUE)
            pygame.display.set_caption('Neural Introspection')
            clock = pygame.time.Clock()

            def screen_mssg_variable(text, variable, area):
                font = pygame.font.SysFont('arial', 16)
                txt = font.render(text + str(variable), True, WHITE)
                gameDisplay.blit(txt, area)


            def process_img(img, x,y):
                # swap the axes else the image will come not the same as the matplotlib one
                img = img.transpose(1,0,2)
                surf = pygame.surfarray.make_surface(img)
                surf = pygame.transform.scale(surf, (300, 300))
                gameDisplay.blit(surf, (x, y))

            #all_data = [{'nav':[],'drop':[]}] * FLAGS.episodes #each entry is an episode, sorted into nav or drop steps
            all_data = [{'nav':[],'stuck':False} for x in range(FLAGS.episodes)]
            step_data = {}
            dictionary = {}
            running = True
            while runner.episode_counter <= (FLAGS.episodes - 1) and running==True:
                print('Episode: ', runner.episode_counter)

                # Init storage structures
                dictionary[runner.episode_counter] = {}
                mb_obs = []
                mb_actions = []
                mb_action_probs = []
                mb_flag = []
                mb_fc = []
                mb_rewards = []
                mb_values = []
                mb_drone_pos = []
                mb_heading = []
                mb_crash = []
                mb_map_volume = [] # obs[0]['volume']==envs.map_volume
                mb_ego = []

                runner.reset_demo()  # Cauz of differences in the arrangement of the dictionaries
                # STORE THE INIT HEADING AND ALT FOR NAMING THE FILE (yes it will ovewrite it num_epis times withe the same values)
                drone_head_alt = str(runner.envs.heading) + '-' + str(runner.envs.altitude)
                map_xy = runner.envs.map_image
                map_alt = runner.envs.alt_view
                process_img(map_xy, 20, 20)
                process_img(map_alt, 20, 400)
                pygame.display.update()

                dictionary[runner.episode_counter]['hiker_pos'] = runner.envs.hiker_position
                dictionary[runner.episode_counter]['map_name'] = str(runner.envs._map[0]) + '-' + str(runner.envs._map[1])
                # dictionary[nav_runner.episode_counter]['map_volume'] = map_xy

                # Quit pygame if the (X) button is pressed on the top left of the window
                # Seems that without this for event quit doesnt show anything!!!
                # Also it seems that the pygame.event.get() is responsible to REALLY updating the screen contents
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                sleep(sleep_time)
                # Timestep counter
                t=0

                drop_flag = 0
                stuck_flag = 0
                done = 0

                while done==0:

                    mb_obs.append(runner.latest_obs)
                    # mb_flag.append(drop_flag)
                    mb_heading.append(runner.envs.heading)

                    drone_pos = np.where(runner.envs.map_volume['vol'] == runner.envs.map_volume['feature_value_map']['drone'][runner.envs.altitude]['val'])
                    mb_drone_pos.append(drone_pos)
                    mb_map_volume.append(runner.envs.map_volume) # what is contained here?
                    mb_ego.append(runner.envs.ego)

                    step_data = {'stuck':False}
                    #I need the egocentric view + hiker's position
                    #then drone steps, need action
                    step_data['volume'] = np.array(runner.envs.map_volume['vol'],copy=True)
                    step_data['heading'] = runner.envs.heading
                    step_data['hiker'] = runner.envs.hiker_position
                    step_data['altitude'] = runner.envs.altitude
                    step_data['drone'] = np.where(step_data['volume'] == runner.envs.map_volume['feature_value_map']['drone'][runner.envs.altitude]['val'])


                    # dictionary[nav_runner.episode_counter]['observations'].append(nav_runner.latest_obs)
                    # dictionary[nav_runner.episode_counter]['flag'].append(drop_flag)

                    # INTERACTION
                    obs, action, value, reward, done, info, fc, action_probs = runner.run_trained_batch()

                    if done and not info['success']:
                        mb_crash.append(runner.envs.crash)
                        print('Crash, terminate episode')
                        break # Also we prevent new data for the new time step to be saved

                    mb_actions.append(action)
                    mb_action_probs.append(action_probs)
                    mb_rewards.append(reward)
                    mb_fc.append(fc)
                    mb_values.append(value)
                    mb_crash.append(runner.envs.crash)

                    step_data['action'] = action
                    step_data['reward'] = reward
                    step_data['fc'] = fc
                    step_data['action_probs'] = action_probs

                    all_data[runner.episode_counter]['nav'].append(step_data)


                    screen_mssg_variable("Value    : ", np.round(value,3), (168, 350))
                    screen_mssg_variable("Reward: ", np.round(reward,3), (168, 372))
                    pygame.display.update()
                    pygame.event.get()
                    sleep(sleep_time)

                    if action==15:
                        drop_flag = 1
                        # dictionary[runner.episode_counter]['pack-hiker_dist'] = runner.envs.pack_dist
                        # dictionary[runner.episode_counter]['pack condition'] = runner.envs.package_state
                        screen_mssg_variable("Package state:", runner.envs.package_state, (20, 350)) # The update of the text will be at the same time with the update of state
                        pygame.display.update()
                        pygame.event.get()  # Update the screen
                        sleep(sleep_time)
                    mb_flag.append(drop_flag)

                    if done:
                        score = sum(mb_rewards)
                        print(">>>>>>>>>>>>>>>>>>>>>>>>>>> episode %d ended in %d steps. Score %f" % (runner.episode_counter, t, score))

                    # BLIT!!!
                    # First Background covering everyything from previous session
                    gameDisplay.fill(DARK_BLUE)
                    map_xy = obs[0]['img']
                    map_alt = obs[0]['nextstepimage']
                    process_img(map_xy, 20, 20)
                    process_img(map_alt, 20, 400)
                    # Update finally the screen with all the images you blitted in the run_trained_batch
                    pygame.display.update() # Updates only the blitted parts of the screen, pygame.display.flip() updates the whole screen
                    pygame.event.get() # Show the last state and then reset
                    sleep(sleep_time)

                    t += 1
                    if t == 70:
                        stuck_flag = 1
                        step_data['stuck'] = True
                        all_data[runner.episode_counter]['stuck'] = True
                        all_data[runner.episode_counter]['nav'].append(step_data)
                        break
                    # else:
                    #     stuck_flag = 0

                dictionary[runner.episode_counter]['map_volume'] = mb_map_volume # You might need to save only for epis=0 % YOU NEED THE NEW MAP AT EVERY STEP SO SAVE THE ARRAY (FLAT)
                dictionary[runner.episode_counter]['ego'] = mb_ego # SHOULD BE SAVED FOR EVERY STEP (THE 5x5 array)
                dictionary[runner.episode_counter]['flag'] = mb_flag
                dictionary[runner.episode_counter]['actions'] = mb_actions
                dictionary[runner.episode_counter]['action_probs'] = mb_action_probs
                dictionary[runner.episode_counter]['rewards'] = mb_rewards
                dictionary[runner.episode_counter]['fc'] = mb_fc
                dictionary[runner.episode_counter]['values'] = mb_values
                dictionary[runner.episode_counter]['drone_pos'] = mb_drone_pos
                dictionary[runner.episode_counter]['headings'] = mb_heading # THIS STORES THE HEADINGS
                dictionary[runner.episode_counter]['crash'] = mb_crash
                dictionary[runner.episode_counter]['pack-hiker_dist'] = runner.envs.pack_dist if drop_flag==1 else None
                dictionary[runner.episode_counter]['pack condition'] = runner.envs.package_state if drop_flag==1 else None
                dictionary[runner.episode_counter]['pack position'] = runner.envs.package_position if drop_flag==1 else None
                dictionary[runner.episode_counter]['stuck_epis'] = stuck_flag# if stuck_flag else 0

                runner.episode_counter += 1
                clock.tick(15)

            print("...saving dictionary.")
            #TODO: IMPORT THE onepolicy_analysis.py FILE AND CONVERT IT INTO PANDAS
            folder = '/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/'
            map_name = str(runner.envs._map[0]) + '-' + str(runner.envs._map[1])#'custom'#str(runner.envs._map[0]) + '-' + str(runner.envs._map[1])
            drone_init_loc = str(runner.envs.drone[0]) + '-' + str(runner.envs.drone[1])
            # drone_head_alt = str(runner.envs.heading) + '-' + str(runner.envs.altitude) # THIS IS WRONG AS IT WILL GET THE LAST HEADING AND ALT
            hiker_loc = str(runner.envs.hiker[0]) + '-' + str(runner.envs.hiker[1])
            type = '.tj'
            path = folder + 'MAP' + map_name + '_' + 'D' + drone_init_loc + '_' + 'HeadAlt' + drone_head_alt + '_' + 'H' + hiker_loc + '_' + str(FLAGS.episodes) + type
            pickle_in = open(path,'wb')
            pickle.dump(dictionary, pickle_in)# Saves a dictionary (seems smaller file)

            # with open('./data/all_data' + map_name + '_' + drone_init_loc + '_' + drone_head_alt + '_' + hiker_loc + str(FLAGS.episodes) + '.lst', 'wb') as handle:
            #     pickle.dump(all_data, handle)# Saves a list (seems larger file)
            # with open('./data/All_maps_20x20_500.tj', 'wb') as handle:
            #     pickle.dump(dictionary, handle)

        except KeyboardInterrupt:
            pass

    print("Okay. Work is done")

    envs.close()


if __name__ == "__main__":
    main()
