# Fully Convolutional Network (the 2nd in DeepMind's paper)
import logging
import os
import shutil
import sys
from datetime import datetime
from time import sleep
import numpy as np
import pickle

import pandas as pd
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
#from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from subproc_vec_env_custom import SubprocVecEnv

from baselines.common.vec_env.vec_frame_stack import VecFrameStack
#from baselines.common.vec_env.vec_normalize import VecNormalize
import gym
#import gym_gridworld
#from gym_grid.envs import GridEnv
import gym_gridworld

from gym_gridworld.envs.gridworld_env import HEADING
from gym_gridworld.envs.gridworld_env import ACTION
import json
import ast

FLAGS = flags.FLAGS
flags.DEFINE_bool("visualize", True, "Whether to render with pygame.")
flags.DEFINE_boolean("training", False,
                     "if should train the model, if false then save only episode score summaries"  )

flags.DEFINE_float("sleep_time", 0, "Time-delay in the demo")


flags.DEFINE_integer("n_envs", 10, "Number of environments to run in parallel")
flags.DEFINE_integer("episodes", 5, "Number of complete episodes")
flags.DEFINE_integer("n_steps_per_batch", 32,
                     "Number of steps per batch, if None use 8 for a2c and 128 for ppo")  # (MINE) TIMESTEPS HERE!!! You need them cauz you dont want to run till it finds the beacon especially at first episodes - will take forever

flags.DEFINE_integer("K_batches", 1003, # Batch is like a training epoch!
                     "Number of training batches to run in thousands, use -1 to run forever") #(MINE) not for now


flags.DEFINE_integer("resolution", 32, "Resolution for screen and minimap feature layers.")
flags.DEFINE_integer("step_mul", 100, "Game steps per agent step.")
flags.DEFINE_integer("step2save", 1000, "Game step to save the model.") #A2C every 1000, PPO 250


# Tensorboard Summaries

flags.DEFINE_integer("all_summary_freq", 50, "Record all summaries every n batch")
flags.DEFINE_integer("scalar_summary_freq", 5, "Record scalar summaries every n batch")
flags.DEFINE_string("checkpoint_path", "_files/models", "Path for agent checkpoints")
flags.DEFINE_string("summary_path", "_files/summaries", "Path for tensorboard summaries")
flags.DEFINE_string("model_name", "A2C_multiple_packs", "Name for checkpoints and tensorboard summaries") # DONT touch TESTING is the best (take out normalization layer in order to work! -- check which parts exist in the restore session if needed)
flags.DEFINE_enum("if_output_exists", "fail", ["fail", "overwrite", "continue"],
                  "What to do if summary and model output exists, only for training, is ignored if notraining")


flags.DEFINE_string("map_name", "DefeatRoaches", "Name of a map to use.")


# Learner Parameters

flags.DEFINE_float("discount", 0.95, "Reward-discount for the agent")

flags.DEFINE_float("max_gradient_norm", 10.0, "good value might depend on the environment") # orig: 1000
flags.DEFINE_float("loss_value_weight", 0.5, "good value might depend on the environment") # orig:1.0
flags.DEFINE_float("entropy_weight_spatial", 0.00000001,
    "entropy of spatial action distribution loss weight") # orig:1e-6
flags.DEFINE_float("entropy_weight_action", 0.001, "entropy of action-id distribution loss weight") # orig:1e-6
flags.DEFINE_float("ppo_lambda", 0.95, "lambda parameter for ppo")
flags.DEFINE_integer("ppo_batch_size", None, "batch size for ppo, if None use n_steps_per_batch")
flags.DEFINE_integer("ppo_epochs", 3, "epochs per update")
flags.DEFINE_enum("policy_type", "FullyConv", ["MetaPolicy", "FullyConv", "Relational", "DeepFullyConv"], "Which type of Policy to use")
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




def make_custom_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = gym.make(env_id, **wrapper_kwargs)
            env.seed(seed + rank)
            # Monitor should take care of reset!
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True) # SUBPROC NEEDS 4 OUTPUS FROM STEP FUNCTION
            return env
        return _thunk
    #set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


class Simulation:

    def __init__(self,

                 training=FLAGS.training,
                 verbose = FLAGS.visualize,
                 environment_id = 'gridworld-v0',
                 model_name = FLAGS.model_name,
                 policy_type = FLAGS.policy_type,

                 drone_initial_position=None,
                 drone_initial_heading = None,
                 drone_initial_altitude= None,
                 hiker_initial_position=None,

                 curriculum_radius=None,
                 goal_mode=None,
                 use_mavsim=None,
                 K_batches=FLAGS.K_batches,
                 episode_length=None  ):

        self.training = training
        self.verbose = verbose
        self.environment_id = environment_id
        self.model_name = model_name
        self.K_batches=K_batches
        self.policy_type=policy_type

        #TODO this runner is maybe too long and too messy..
        self.full_checkpoint_path = os.path.join(FLAGS.checkpoint_path, self.model_name)

        if self.training:
            self.full_summary_path = os.path.join(FLAGS.summary_path, self.model_name)
        else:
            self.full_summary_path = os.path.join(FLAGS.summary_path, "no_training", self.model_name)


        if self.training:
            check_and_handle_existing_folder(self.full_checkpoint_path)
            check_and_handle_existing_folder(self.full_summary_path)

        kwargs= { 'drone_initial_position':drone_initial_position,
                  'drone_initial_heading':drone_initial_heading,
                  'drone_initial_altitude':drone_initial_altitude,
                  'hiker_initial_position':hiker_initial_position,
                  'goal_mode':goal_mode,
                  'verbose':verbose,
                  'episode_length':episode_length,
                  'curriculum_radius':curriculum_radius,
                  'use_mavsim':use_mavsim}

        #(MINE) Create multiple parallel environements (or a single instance for testing agent)
        if self.training and self.verbose==False:
            #envs = SubprocVecEnv((partial(make_sc2env, **env_args),) * FLAGS.n_envs)
            #envs = SubprocVecEnv([make_env(i,**env_args) for i in range(FLAGS.n_envs)])
            self.envs = make_custom_env(self.environment_id, FLAGS.n_envs, 1, wrapper_kwargs=kwargs)
        elif self.training==False:
            #envs = make_custom_env('gridworld-v0', 1, 1)
            print("Making a single Environment for Testing")
            self.envs = gym.make(self.environment_id, **kwargs)
        else:
            print('Wrong choices in FLAGS training and visualization')
            return

            #envs = SingleEnv(make_sc2env(**env_args))
        #envs = gym.make('gridworld-v0')
        # envs = SubprocVecEnv([make_env(i) for i in range(FLAGS.n_envs)])
        # envs = VecNormalize(env)
        # use for debugging 'Breakout-v0', Grid-v0, gridworld-v0
        #envs = VecFrameStack(make_custom_env('gridworld-v0', FLAGS.n_envs, 1), 1) # One is number of frames to stack within each env
        #envs = make_custom_env('gridworld-v0', FLAGS.n_envs, 1)
        print("Requested environments created successfully")
        #env = gym.make('gridworld-v0')
        tf.reset_default_graph()
        # The following lines fix the problem with using more than 2 envs!!!
        # config = tf.ConfigProto(allow_soft_placement=True,
        #                                     log_device_placement=True) # first option allows to use with device:cpu so some operations go there and second option shows all operations where do they go
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config)
        #sess = tf.Session()

        self.agent = ActorCriticAgent(
            mode=FLAGS.agent_mode,
            sess=self.sess,
            spatial_dim=FLAGS.resolution, # Here you pass the resolution which is used in the step for the output probabilities map
            unit_type_emb_dim=5,
            loss_value_weight=FLAGS.loss_value_weight,
            entropy_weight_action_id=FLAGS.entropy_weight_action,
            entropy_weight_spatial=FLAGS.entropy_weight_spatial,
            scalar_summary_freq=FLAGS.scalar_summary_freq,
            all_summary_freq=FLAGS.all_summary_freq,
            summary_path=self.full_summary_path,
            max_gradient_norm=FLAGS.max_gradient_norm,
            num_actions=self.envs.action_space.n,
            policy=self.policy_type
        )
        # Build Agent
        self.agent.build_model()
        if os.path.exists(self.full_checkpoint_path):
            print("Found existing model, loading weights")
            self.agent.load(self.full_checkpoint_path) #(MINE) LOAD!!!
        else:
            print("No model, random weights")
            self.agent.init()

        # (MINE) Define TIMESTEPS per episode (batch as each worker has its own episodes -- different timelines)
        # If it is not training you don't need that many steps. You need one to take the decision...Actually seem to be game steps
        if FLAGS.n_steps_per_batch is None:
            self.n_steps_per_batch = 128 if FLAGS.agent_mode == ACMode.PPO else 8
        else:
            self.n_steps_per_batch = FLAGS.n_steps_per_batch

        if FLAGS.agent_mode == ACMode.PPO:
            ppo_par = PPORunParams(
                FLAGS.ppo_lambda,
                batch_size=FLAGS.ppo_batch_size or self.n_steps_per_batch,
                n_epochs=FLAGS.ppo_epochs
            )
        else:
            ppo_par = None

        self.runner = Runner(
            envs=self.envs,
            agent=self.agent,
            discount=FLAGS.discount,
            n_steps=self.n_steps_per_batch,
            do_training=self.training,
            ppo_par=ppo_par,
            policy_type = FLAGS.policy_type
        )



    def _save_if_training(self,agent):
        agent.save(self.full_checkpoint_path)
        agent.flush_summaries()
        sys.stdout.flush()



    def run(self,
            episodes_to_run=FLAGS.episodes,
            curriculum_radius=None,
            sleep_time = FLAGS.sleep_time):


        """Runs one or more episodes/batches and records the outcomes, optionally providing a visualization.

           episodes_to_run -- number of episodes to execute

           sleep_time -- amount of time to pause in between every simulation step - useful for visualization

           return -- a datastructure which is a list of dictionaries, one per epsisode that describe the episode

                all_data is an ordered list of episode description records. Each record is a dictionary

                all_data[episode#] = { 'stuck': Boolean,  # True if agent timed out
                                     'nav':   List }    # List of steps in the trajectory of the agent
                                     
                step = {'stuck': Boolean
                        'volume':   translation of map to 3D vector voxels
                        'heading':  drone heading integer [1,8] corresponding to N, NE, E, ...
                        'hiker':    hiker position (
                        'altitude': drone altitude
                        'drone':    drone position
                        'action':   last action
                        'reward':   last reward - real value
                        'fc':       vector given the activations of the fully connected layer of the policy
                        'action_probs':  distribution over actions
                        'info':      a dictionary giving additional information about environment events
                        'map':      map used
                }
                            
                            
                            """

        if curriculum_radius!=None:
            self.curriculum_radius=curriculum_radius
            self.runner.reset(curriculum_radius=curriculum_radius)

        self.episodes_to_run = episodes_to_run
        self.sleep_time=sleep_time
        self.trajectory = list()

        # runner.reset() # Reset env which means you get first observation. You need reset if you run episodic tasks!!! SC2 is not episodic task!!!

        if self.K_batches >= 0:
            n_batches = self.K_batches  # (MINE) commented here so no need for thousands * 1000
        else:
            n_batches = -1


        all_data = [{'nav':[],'stuck':False} for x in range(episodes_to_run)]


        if self.training:
            i = 0

            try:
                print("run_agent.py:main.training reset")
                self.runner.reset()
                t=0
                while True:
                    # For below you have to enable monitor's early reset. You might want to take out Monitor
                    # if t==390:
                    #     runner.reset()
                    #     t=0

                    if i % 500 == 0:
                        _print(i)
                    if i % FLAGS.step2save == 0:
                        self._save_if_training(self.agent)
                    if FLAGS.policy_type == 'MetaPolicy':
                        self.runner.run_meta_batch()
                    else:
                        self.runner.run_batch(i)  # (MINE) HERE WE RUN MAIN LOOP for while true
                    #runner.run_batch_solo_env()
                    i += 1
                    if 0 <= n_batches <= i: #when you reach the certain amount of batches break
                        break
                    # t=t+1
            except KeyboardInterrupt:
                pass
        else: # Test the agent

            try:
                import pygame
                import time
                import random
                # pygame.font.get_fonts() # Run it to get a list of all system fonts
                display_w = 1200
                display_h = 720

                BLUE = (128, 128, 255)
                DARK_BLUE = (1, 50, 130)
                RED = (255, 192, 192)
                BLACK = (0, 0, 0)
                WHITE = (255, 255, 255)

                sleep_time = self.sleep_time

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
                step_data = {}
                dictionary = {}
                running = True
                while self.runner.episode_counter < self.episodes_to_run and running==True:
                    print('Episode: ', self.runner.episode_counter)

                    # Init storage structures
                    dictionary[self.runner.episode_counter] = {}
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

                    self.runner.reset_demo()  # Cauz of differences in the arrangement of the dictionaries

                    map_name = str(self.runner.envs.submap_offset)
                    map_xy = self.runner.envs.map_image
                    map_alt = self.runner.envs.alt_view
                    process_img(map_xy, 20, 20)
                    process_img(map_alt, 20, 400)
                    pygame.display.update()

                    dictionary[self.runner.episode_counter]['hiker_pos'] = self.runner.envs.hiker_position
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
                    done = False

                    while not done:

                        mb_obs.append(self.runner.latest_obs)
                        # mb_flag.append(drop_flag)
                        mb_heading.append(self.runner.envs.heading)

                        drone_pos = np.where(self.runner.envs.map_volume['vol'] == self.runner.envs.map_volume['feature_value_map']['drone'][self.runner.envs.altitude]['val'])
                        mb_drone_pos.append(drone_pos)
                        mb_map_volume.append(self.runner.envs.map_volume) # what is contained here?
                        mb_ego.append(self.runner.envs.ego)

                        step_data = {'stuck':False}
                        #I need the egocentric view + hiker's position
                        #then drone steps, need action
                        step_data['volume'] = np.array(self.runner.envs.map_volume['vol'],copy=True)
                        step_data['heading'] = self.runner.envs.heading
                        step_data['hiker'] = self.runner.envs.hiker_position
                        step_data['altitude'] = self.runner.envs.altitude
                        step_data['drone'] = np.where(step_data['volume'] == self.runner.envs.map_volume['feature_value_map']['drone'][self.runner.envs.altitude]['val'])


                        # dictionary[nav_runner.episode_counter]['observations'].append(nav_runner.latest_obs)
                        # dictionary[nav_runner.episode_counter]['flag'].append(drop_flag)
                        print("Agent taking step")
                        # INTERACTION
                        obs, action, value, reward, done, info, fc, action_probs = self.runner.run_trained_batch()

                        print("Is done?? ",done)

                        step_data['action'] = action

                        step_data['reward'] = reward
                        step_data['fc'] = fc
                        step_data['action_probs'] = action_probs
                        step_data['info'] = info
                        step_data['map'] = self.runner.envs.submap_offset
                        print("run_agent.py:run episode {} appending step_data ".format(self.runner.episode_counter))
                        all_data[self.runner.episode_counter]['nav'].append(step_data)


                        if done and not info['success']:
                            print('Crash, terminate episode')
                            break # Also we prevent new data for the new time step to be saved

                        mb_actions.append(action)
                        mb_action_probs.append(action_probs)
                        mb_rewards.append(reward)
                        mb_fc.append(fc)
                        mb_values.append(value)
                        mb_crash.append(self.runner.envs.crash)



                        screen_mssg_variable("Value    : ", np.round(value,3), (168, 350))
                        screen_mssg_variable("Reward: ", np.round(reward,3), (168, 372))
                        pygame.display.update()
                        pygame.event.get()
                        sleep(sleep_time)

                        if action==15:
                            drop_flag = 1
                            # dictionary[runner.episode_counter]['pack-hiker_dist'] = runner.envs.pack_dist
                            # dictionary[runner.episode_counter]['pack condition'] = runner.envs.package_state
                            screen_mssg_variable("Package state:", self.runner.envs.package_state, (20, 350)) # The update of the text will be at the same time with the update of state
                            pygame.display.update()
                            pygame.event.get()  # Update the screen
                            sleep(sleep_time)
                        mb_flag.append(drop_flag)

                        if done:
                            score = sum(mb_rewards)
                            print(">>>>>>>>>>>>>>>>>>>>>>>>>>> episode %d ended in %d steps. Score %f" % (self.runner.episode_counter, t, score))

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
                            all_data[self.runner.episode_counter]['stuck'] = True
                            all_data[self.runner.episode_counter]['nav'].append(step_data)
                            break
                        # else:
                        #     stuck_flag = 0

                    episode_dict = dictionary[self.runner.episode_counter]
                    episode_dict['map_volume'] = mb_map_volume # You might need to save only for epis=0
                    episode_dict['ego'] = mb_ego
                    episode_dict['flag'] = mb_flag
                    episode_dict['actions'] = mb_actions
                    episode_dict['action_probs'] = mb_action_probs
                    episode_dict['rewards'] = mb_rewards
                    episode_dict['fc'] = mb_fc
                    episode_dict['values'] = mb_values
                    episode_dict['drone_pos'] = mb_drone_pos
                    episode_dict['headings'] = mb_heading
                    episode_dict['crash'] = mb_crash
                    episode_dict['pack-hiker_dist'] = self.runner.envs.pack_dist if drop_flag==1 else None
                    episode_dict['pack condition'] = self.runner.envs.package_state if drop_flag==1 else None
                    episode_dict['pack position'] = self.runner.envs.package_position if drop_flag==1 else None
                    episode_dict['stuck_epis'] = stuck_flag# if stuck_flag else 0

                    self.runner.episode_counter += 1
                    clock.tick(15)

                print("...saving dictionary.")


                base_dir_path = os.path.dirname(os.path.realpath(__file__))

                self.data_folder = base_dir_path+'/data/'
                if not os.path.exists( self.data_folder):
                    os.mkdir( self.data_folder)

                map_name = str(self.runner.envs.submap_offset[0]) + '-' + str(self.runner.envs.submap_offset[1])
                drone_init_loc = 'D1118'
                hiker_loc = 'H1010'
                type = '.tj'
                path =  self.data_folder + map_name + '_' + drone_init_loc + '_' + hiker_loc + '_' + str(FLAGS.episodes) + type
                pickle_in = open(path,'wb')
                pickle.dump(dictionary, pickle_in)

                with open('./data/all_data' + map_name + '_' + drone_init_loc + '_' + hiker_loc + str(FLAGS.episodes) + '.lst', 'wb') as handle:
                    pickle.dump(all_data, handle)
                # with open('./data/All_maps_20x20_500.tj', 'wb') as handle:
                #     pickle.dump(dictionary, handle)

            except KeyboardInterrupt:
                pass

        print("Okay. Work is done")


        return all_data

    def close(self):

        self.envs.close()


def extract_episode_trajectory_as_dataframe(episode_step_list):

    key_columns = [
                    (
                       (step['drone'][1][0], step['drone'][2][0]),
                       step['heading'],
                       HEADING.to_short_string(step['heading']),
                       step['altitude'],
                       step['action'],
                       ACTION.to_short_string(step['action']),
                       step['reward'],
                       step['map'],
                       "{}".format(step['info'])
                    )

                    for step in episode_step_list
                  ]


    df  = pd.DataFrame( key_columns, columns=['location','head','hname','alt','act','aname','reward','map','info'])


    return df


def augment_dataframe_with_reward_detail(df):

    info = df['info']

    Rhike=list()
    Rstep=list()
    Rcrash=list()
    Rtime=list()
    Success=list()

    for row in info:

        row = ast.literal_eval(row)

        if 'Rhike' in row:
            Rhike.append(row['Rhike'])
        else:
            Rhike.append(0)

        if 'Rstep' in row:
            Rstep.append(row['Rstep'])
        else:
            Rstep.append(0)

        if 'Rcrash' in row:
            Rcrash.append(row['Rcrash'])
        else:
            Rcrash.append(0)

        if 'Rtime' in row:
            Rtime.append(row['Rtime'])
        else:
            Rtime.append(0)

        if 'ex' in row and row['ex']=='arrived':
            Success.append(1)
        else:
            Success.append(0)

    df['Rhike']=Rhike
    df['Rstep']=Rstep
    df['Rcrash']=Rcrash
    df['Rtime']=Rtime
    df['Success']=Success


    return df


def aggregate_rewards(dataframe):

    stats = dataframe.groupby(['episode'])[['reward','Rstep','Rcrash','Rhike','Rtime','Success']].sum()
    counts = dataframe.groupby(['episode'])[['reward']].count()

    stats['steps'] = counts

    return stats


def extract_all_episodes(result):

    df_all = None

    for episode_num,episode_dictionary in enumerate(result):

        df = extract_episode_trajectory_as_dataframe(episode_dictionary['nav'])
        df['episode']=episode_num

        if df_all is None:
            df_all = df
        else:
            df_all = pd.concat([df_all,df])

    return df_all


def to_mavsim_actions(df_all,episode_num=0):

    """Converts a trajectory from the CMU_DRONE to MAVSim actions.

       The drone takes an action such as UP_RIGHT_45.
       This causes it to change headings from North to North East.
       It causes the drone to descend from altitude 3 to altitude 2.

       We can replace this with a MAVSim action based on
       where the drone gets to instead of the action taken.

       ( FLIGHT HEAD_TO <heading_degrees> <distance> <altitude> )

       We treat this as a MAVSim action (FLIGHT HEAD_TO 45 1 2)

       Note, the very first heading and altitude are the intial state so
       they do not represent an action. We discard them.

       The last action taken by the drone is not executed to produce a new state.
       So, there are only N-1 actions for an N step trajectory."""


    df = df_all[  df_all['episode'] == episode_num  ]


    actions = df['act']
    last_action = actions.iloc[-1]


    altitudes = df['alt']
    last_altitude = altitudes.iloc[-1]
    next_altitude = ACTION.new_altitude(last_altitude,last_action)
    mavsim_altitudes = df['alt'].to_list()
    mavsim_altitudes.pop(0)
    #mavsim_altitudes.append(next_altitude)

    headings = df['head']
    last_heading = headings.iloc[-1]
    next_heading = ACTION.new_heading(last_heading,last_action)
    mavsim_headings = df['head'].to_list()
    mavsim_headings.pop(0)
    #mavsim_headings.append(next_heading)

    mavsim_actions = [ "(FLIGHT HEAD_TO {} 1 {})".format(h,a) for (h,a) in zip( mavsim_headings, mavsim_altitudes)  ]

    return mavsim_actions



def to_mavsim_rewards(df_all, episode_num=0 ):


    df = df_all[  df_all['episode'] == episode_num  ]

    info = df['info']

    Rhike=0
    Rstep=0
    Rcrash=0
    Rtime=0

    for row in info:

        row = ast.literal_eval(row)

        if 'Rhike' in row:
            Rhike=Rhike+row['Rhike']

        if 'Rstep' in row:
            Rstep=Rstep+row['Rstep']

        if 'Rcrash' in row:
            Rcrash=Rcrash+row['Rcrash']

        if 'Rtime' in row:
            Rtime=Rtime+row['Rtime']

    Rsum = Rstep+Rhike+Rcrash+Rtime

    Rtot = df['reward'].sum()

    return { 'Rhike':Rhike, 'Rstep':Rstep, 'Rcrash':Rcrash, 'Rtime':Rtime, 'Rsum':Rsum, 'Rtot':Rtot }


if __name__ == "__main__":

    sim = Simulation()

    result = sim.run(  )

    sim.close()




