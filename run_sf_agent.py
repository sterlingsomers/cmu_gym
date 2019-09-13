import datetime
import io
import os
import uuid

import numpy as np
import tensorflow as tf
import cv2
import gym
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import spacefortress.gym

class CollectGymDataset(object):
  """Collect transition tuples and store episodes as Numpy files."""

  def __init__(self, env, outdir):
    self._env = env
    self._outdir = outdir and os.path.expanduser(outdir)
    self._episode = None
    self._transition = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action, *args, **kwargs):
    if kwargs.get('blocking', True):
      transition = self._env.step(action, *args, **kwargs)
      return self._process_step(action, *transition) # transition returns obs,r, done, info and with the *transition you indicate exactly this
    else:
      future = self._env.step(action, *args, **kwargs)
      return lambda: self._process_step(action, *future())

  def reset(self, *args, **kwargs):
    if kwargs.get('blocking', True):
      observ = self._env.reset(*args, **kwargs)
      return self._process_reset(observ)
    else:
      future = self._env.reset(*args, **kwargs)
      return lambda: self._process_reset(future())

  def _process_step(self, action, observ, reward, done, info):
    self._transition.update({'action': action, 'reward': reward})
    self._transition.update(info)
    # self._transition.update({'info': np.stack(k for k in info)})
    self._episode.append(self._transition) # append the transition
    self._transition = {} # empty the _transition in order to use it again
    if not done:
      self._transition.update(self._process_observ(observ))
    else: # if done=True
      episode = self._get_episode()
      info['episode'] = episode
      if self._outdir:
        filename = self._get_filename()
        self._write(episode, filename)
    return observ, reward, done, info

  def _process_reset(self, observ):
    self._episode = []
    self._transition = {}
    self._transition.update(self._process_observ(observ))
    return observ

  def _process_observ(self, observ):
    if not isinstance(observ, dict):
      observ = {'observ': observ}
    return observ

  def _get_filename(self):
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    identifier = str(uuid.uuid4()).replace('-', '')
    filename = '{}-{}.npz'.format(timestamp, identifier)
    filename = os.path.join(self._outdir, filename)
    return filename

  def _get_episode(self):
    episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
    episode = {k: np.array(v) for k, v in episode.items()}
    for key, sequence in episode.items():
      if sequence.dtype == 'object':
        message = "Sequence '{}' is not numeric:\n{}"
        raise RuntimeError(message.format(key, sequence))
    return episode

  def _write(self, episode, filename):
    if not tf.gfile.Exists(self._outdir):
      tf.gfile.MakeDirs(self._outdir)
    with io.BytesIO() as file_:
      np.savez_compressed(file_, **episode)
      file_.seek(0)
      with tf.gfile.Open(filename, 'w') as ff:
        ff.write(file_.read())
    name = os.path.splitext(os.path.basename(filename))[0]
    print('Recorded episode {}.'.format(name))

class ObservationDict(object):

  def __init__(self, env, key='observ'):
    self._env = env
    self._key = key

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = {self._key: self._env.observation_space}
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    return self._env.action_space

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs = {self._key: np.array(obs)}
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs = {self._key: np.array(obs)} # simply will create a dictionary with the name of the key and the obs are the value. But if you have multiple ones defined in observation spaces of the env then it changes.
    return obs

def obs_preprocess(obs):
    return cv2.resize(obs, (150, 150), interpolation=cv2.INTER_AREA)

def make_custom_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = gym.make(env_id)
            #env.seed(seed + rank)
            #env = CollectGymDataset(ObservationDict(envs), None) # Monitor doesn't work with these wrappers
            # Monitor should take care of reset!
#             env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True) # SUBPROC NEEDS 4 OUTPUS FROM STEP FUNCTION
            return env
        return _thunk
    #set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


num_episodes = 5
n_envs = 2

env = gym.make('SpaceFortress-autoturn-image-v0')
# env = make_custom_env('SpaceFortress-autoturn-image-v0', n_envs, 1)
env = CollectGymDataset(ObservationDict(env), None) # None for outdir so you do not save. Monitor messes up these wrappers.
episodes = []
for _ in range(num_episodes):  # (MINE) Use your own policy from a trained model (A2C_multiple_packs). You might need some exploration though here if you want MB to perform better than MF
    policy = lambda env,obs: env.action_space.sample()  # lambda defines an anonymous function policy that takes as inputs env and obs and performs the env.action_sample() operation
    done = False
    obs = env.reset()
    while not done:
        action = policy(env, obs)  # If the action is discrete you wont have any probs as the action normalize wrapper will produce a 16-element vector with same elements (the discrete action) and the sokoban wrapper will select the argmax
        # obs, _, done, info = env.step([action] * n_envs)obs, _, done, info = env.step([action] * n_envs)
        obs, _, done, info = env.step(action)# obs, _, done, info = env.step([action]*n_envs)  # if it crashes done comes out from the duration wrappers as false!

    #Below the line is not working. Probably because of the CollectGymDataset
    episodes.append(info['episode'])  # seems that the last obs is not saved, only till one step before the goal. So the images are s_{goal-1}, action, s_goal, and the action is the action that leads to the goal. sgoal is not stored
try:
    env.close()
except AttributeError:
    pass