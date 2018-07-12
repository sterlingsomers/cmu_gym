#!/usr/bin/env python3
import os
from baselines.bench import  Monitor
from baselines import logger
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.a2c.a2c import learn
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
from gym_gridworld_opeanai_viz import gym_grid

def train(env_id, num_timesteps, seed, policy, lrschedule, num_env):
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    env = VecFrameStack(make_atari_env(env_id, num_env, seed), 4)
    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule)
    env.close()

def make_env(rank):
    def _thunk():
        env = make_env(env_id, process_idx=rank, outdir=logger.get_dir())
        env.seed(seed + rank)
        if logger.get_dir():
            env = Monitor(env, os.path.join(logger.get_dir(), 'train-{}.monitor.json'.format(rank)))
        return env
    return _thunk

def main():
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    #parser.add_argument('--env', type=str, default='Grid-v0')
    args = parser.parse_args()
    args.env = 'Grid-v0'
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
        policy=args.policy, lrschedule=args.lrschedule, num_env=2)

if __name__ == '__main__':
    main()
