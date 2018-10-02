import argparse
import time
import os
import pickle
from baselines import logger
from baselines.common.misc_util import boolean_flag
from baselines.common import set_global_seeds
from baselines import deepq
#import qlearning

import gym
import sys
#sys.path.append("/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/gym_gridworld")
import gym_gridworld
#from gym_gridworld_opeanai_viz import gym_grid
#import cogle_mavsim
#from gym_recording.wrappers import TraceRecordingWrapper
#import tensorflow as tf
import numpy as np
#from qlearning import build_state


def parse_args():
    """ Parse arguments from command line """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env-id', type=str, default='gridworld-v0')
    #parser.add_argument('--num-timesteps', type=int, default=None)
    parser.add_argument('--qfunction', type=str, default=None)
    boolean_flag(parser, 'learning', default=True)
    parser.add_argument('--epsilon', type=float, default=.3)
    boolean_flag(parser, 'drop_payload_agent', default=False)

    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=1)
    parser.add_argument('--prioritized-replay-alpha', type=float, default=0.6)
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--num-timesteps', type=int, default=int(100000))
    parser.add_argument('--checkpoint-freq', type=int, default=10000)
    parser.add_argument('--checkpoint-path', type=str, default=None)

    args = parser.parse_args()

    logger.configure()
    set_global_seeds(args.seed)


    # we don't directly specify timesteps for this script, so make sure that if
    # we do specify them they agree with the other parameters
    # if args.num_timesteps is not None:
    #     assert args.num_timesteps == args.nb_epochs * args.nb_epoch_cycles *\
    #         args.nb_rollout_steps

    #dict_args = vars(args)
    #del dict_args['num_timesteps']
    #return dict_args
    return args

def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved

def main():
    """ DeepQ-learning """
    args = parse_args()
    #env = gym.make(args['env_id'])
    env = gym.make(args.env_id)

    set_global_seeds(args.seed) # You need to put all args here and not in the parse_args function which converts them into dictionaries
    #set_global_seeds(args['seed']) # For args = dictionary

    model = deepq.models.cnn_to_mlp(
        convs=[(64, 4, 2), (64, 3, 1)],#(64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=bool(args.dueling),
    )
    #model = deepq.models.mlp([64])

    logger.set_level(logger.INFO)
    # max_number_of_steps = 10000
    # number_of_episodes = 10000
    # last_time_steps_reward = np.ndarray(0)
    #
    # # File to store que Q function
    # timestr = time.strftime("%Y%m%d-%H%M%S")
    # file_name = "./deepq_functions/" + timestr + '-' + args['env_id'] + ".qf"
    # file_reward = "./rewards/" + timestr + ".csv"
    # # Only dropping payload agent
    # if args['drop_payload_agent']:
    #     logger.info('Exploitation drop payload agent selected')
    #     env.env.only_drop_payload_agent = True
    #     if args['qfunction'] is None:
    #         logger.error('Q-function not espicified')
    #         return 0
    #     # Turning off exploration
    #     args['epsilon'] = 0
    #     args['learning'] = False
    #     number_of_episodes = 1
    # The Q-learn algorithm
    # qlearn = qlearning.QLearn(actions=range(env.action_space.n),
    #                           alpha=0.4, gamma=0.80, epsilon=args['epsilon'])

    # For CNN
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=args.num_timesteps,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=100,
        batch_size=32,
        learning_starts=1000,#10000,
        target_network_update_freq=500,
        #param_noise=0.05,
        gamma=0.99,
        prioritized_replay=bool(args.prioritized),
        prioritized_replay_alpha=args.prioritized_replay_alpha,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_path=args.checkpoint_path,
        #print_freq=1
    )
    # For mlp
    # act = deepq.learn(
    #     env,
    #     q_func=model,
    #     lr=1e-3,
    #     max_timesteps=2000,
    #     buffer_size=50000,
    #     exploration_fraction=0.1,
    #     exploration_final_eps=0.02,
    #     print_freq=10,
    #     #callback=callback
    # )
    print("Saving model to deepq_mlp.pkl")
    act.save("deepq_mlp.pkl.pkl")
    #=========== EVALUATION/LOAD MODEL TO CONTINUE =============
    # Loads Q function
    # if args['qfunction'] is not None:
    #     try:
    #         with open(args['qfunction'], "rb") as file_p:   # Unpickling
    #             logger.info('Loading qfunction: %s' % args['qfunction'])
    #             qlearn.q = pickle.load(file_p)
    #             file_p.close()
    #     except IOError:
    #         logger.error('Q-Function file does not exists: %s'
    #                      % args['qfunction'])

     #       return 1
    #===========================================
    #========== TRAINING Q-LEARNING ============
    # episode_trace = []
    # for i_episode in range(number_of_episodes):
    #     observation = env.reset()
    #     reward = 0
    #     state = qlearning.build_state(observation) # MY EDIT HERE IT WAS JUST build_state
    #     logger.info("Episode: %d/%d" % (i_episode + 1, number_of_episodes))
    #     if args['learning']:
    #         os.makedirs(os.path.dirname(file_name), exist_ok=True)
    #         with open(file_name, "wb") as file_p:   # Pickling
    #             logger.info('Saving Q function to file: %s' % file_name)
    #             pickle.dump(qlearn.q, file_p)
    #             file_p.close()
    #     for step_t in range(max_number_of_steps):
    #         if step_t > 1:  # to have previous step reading
    #             if step_t % 10 == 0:
    #                 logger.info("step: %d/%d" % (step_t, max_number_of_steps))
    #             # Pick an action based on the current state
    #             action = qlearn.chooseAction(state)
    #             # Execute the action and get feedback
    #             observation, reward, done, info = env.step(action)
    #             next_state = qlearning.build_state(observation)
    #             episode_trace.append([info['self_state']['lon'],
    #                                   info['self_state']['lat'],
    #                                   info['self_state']['alt'],
    #                                   reward])
    #             if not(done) and step_t == max_number_of_steps - 1:
    #                 done = True
    #             if not done:
    #                 if args['learning']:
    #                     qlearn.learn(state, action, reward, next_state)
    #                 state = next_state
    #             else:
    #                 # Q-learn stuff
    #                 if args['learning']:
    #                     qlearn.learn(state, action, reward, next_state)
    #                 last_time_steps_reward = np.append(last_time_steps_reward,
    #                                                    [reward])
    #                 step_t = max_number_of_steps - 1
    #             if done:
    #                 break  # TODO: get rid of all breaks
    #     timestr = time.strftime("%Y%m%d-%H%M%S")
    #     file_trace = "./traces/" + timestr + ".csv"
    #     os.makedirs(os.path.dirname(file_trace), exist_ok=True)
    #     trace_file = open(file_trace, 'w')
    #     logger.info('Saving trace of episode in: %s' % file_trace)
    #     for item in episode_trace:
    #         trace_file.write("{}, {}, {}\n".format(item[0], item[1], item[2]))
    #     del episode_trace[:]
    #     trace_file.close()
    #     # Reward trace to file
    #     os.makedirs(os.path.dirname(file_reward), exist_ok=True)
    #     reward_file = open(file_reward, 'a')
    #     logger.info('Saving episode reward to: %s' % file_reward)
    #     reward_file.write("{}\n".format(reward))
    #     reward_file.close()


if __name__ == '__main__':
    main()
