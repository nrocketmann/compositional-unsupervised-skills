# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example running DQN on BSuite in a single process."""

from matplotlib import pyplot as plt
from absl import app
from absl import flags
import tensorflow as tf
import acme
from acme import specs
from acme import wrappers
from acme.agents.tf import dqn
from IntrinsicRewards.MohammadDQN.gridworld_utils import SinglePrecisionFloatWrapper
import bsuite
import sonnet as snt
from acme.utils import loggers
from babyai.levels.iclr19_levels import Level_GoToRedBallNoDists
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, FullyObsWrapper
import numpy as np
import gym

# Bsuite flags
flags.DEFINE_string('bsuite_id', 'deep_sea/0', 'Bsuite id.')
flags.DEFINE_string('results_dir', '/tmp/bsuite', 'CSV results directory.')
flags.DEFINE_boolean('overwrite', False, 'Whether to overwrite csv results.')
FLAGS = flags.FLAGS


def plotsomestuff(env, spec):
  def show_obs(obs):
    plt.figure()
    plt.imshow(obs)
    # plt.figure()
    # plt.imshow(obs[:,:,0],cmap='gray')
    #
    # plt.figure()
    # plt.imshow(obs[:,:,1],cmap='gray')
    #
    # plt.figure()
    # plt.imshow(obs[:,:,2],cmap='gray')
    #
    plt.show()
    print(obs.shape)
    print("max1 " + str(np.max(obs[:, :, 0])))
    print("max2 " + str(np.max(obs[:, :, 1])))
    print("max3 " + str(np.max(obs[:,:,2])))

  mat = env.reset()
  show_obs(mat.observation)
  for i in range(spec.actions.num_values):
    mat = env.step(0)
    show_obs(mat.observation)
    plt.figure()
    plt.imshow(env.render())
    plt.show()



def main(_):
  #environment = Level_GoToRedBallNoDists()
  environment = gym.make('MiniGrid-Empty-8x8-v0')
  environment = FullyObsWrapper(environment)  # Get full pixel observations
  # environment = RGBImgObsWrapper(environment)
  environment = ImgObsWrapper(environment)

  environment = wrappers.GymWrapper(environment)
  environment = SinglePrecisionFloatWrapper(environment)
  #environment = wrappers.SinglePrecisionWrapper(environment)



  # environment = make_environment('Level1')

  environment_spec = specs.make_environment_spec(environment)
  plotsomestuff(environment, environment_spec)
  environment.reset()

  network = snt.Sequential([
      snt.Conv2D(32,10,2),
      tf.nn.leaky_relu,
      tf.keras.layers.AveragePooling2D(2),
      snt.Conv2D(64, 5, 2),
      tf.keras.layers.AveragePooling2D(2),
      snt.Flatten(),
      snt.nets.MLP([128, 128, environment_spec.actions.num_values])
  ])

  # Construct the agent.
  logger = loggers.CSVLogger('logging/logdir/' +  'VANILLADQN')
  agent = dqn.DQN(
      environment_spec=environment_spec, network=network,learning_rate=.001,epsilon=tf.Variable(.05),logger=logger)

  # Run the environment loop.
  loop = acme.EnvironmentLoop(environment, agent)
  loop.run(num_episodes=1000000)  # pytype: disable=attribute-error


if __name__ == '__main__':
  app.run(main)
