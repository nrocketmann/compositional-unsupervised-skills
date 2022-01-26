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

"""Tests for DQN agent."""

from absl.testing import absltest
from absl import app
import acme
from acme import specs
from IntrinsicRewards import MohammadDQN as dqn
from acme.testing import fakes
from acme import wrappers
from absl import flags
import bsuite

import numpy as np
import sonnet as snt


def _make_Qnetwork(action_spec: specs.DiscreteArray) -> snt.Module:
  return snt.Sequential([
      snt.Flatten(),
      snt.nets.MLP([256, 256, action_spec.num_values]),
  ])

def _make_qnetwork(action_spec: specs.DiscreteArray) -> snt.Module: #takes in s + s' + action, spits out probability
  return dqn.ConditionalProductNetwork(output_dims=action_spec.num_values,categorical=True)

def _make_feat_network(action_spec: specs.DiscreteArray) -> snt.Module: #lol this just makes features, so we'll just flatten for now
  return snt.Sequential([
      snt.Flatten(),snt.Linear(64)
  ])

def _make_rnetwork(action_spec: specs.DiscreteArray) -> snt.Module: #takes in just s and action, spits out probability
  return dqn.RNetwork(output_dims=action_spec.num_values,categorical=True)



def main(_):
  flags.DEFINE_string('bsuite_id', 'deep_sea/0', 'Bsuite id.')
  flags.DEFINE_string('results_dir', '~/tmp/bsuite', 'CSV results directory.')
  flags.DEFINE_boolean('overwrite', True, 'Whether to overwrite csv results.')
  flags.DEFINE_integer('episodes',100,'Number of episodes to write')
  FLAGS = flags.FLAGS

  raw_environment = bsuite.load_and_record_to_csv(
      bsuite_id=FLAGS.bsuite_id,
      results_dir=FLAGS.results_dir,
      overwrite=FLAGS.overwrite,
  )
  environment = wrappers.SinglePrecisionWrapper(raw_environment)
  spec = specs.make_environment_spec(environment)
    # Create a fake environment to test with.

    # Construct the agent.
  agent = dqn.DQNEmpowerment(
        environment_spec=spec,
        Qnetwork=_make_Qnetwork(spec.actions),
        qnetwork = _make_qnetwork(spec.actions),
        feat_network = _make_feat_network(spec.actions),
        feat_dims=64,
        rnetwork = _make_rnetwork(spec.actions),
        batch_size=10,
        samples_per_insert=2,
        min_replay_size=10)

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
  loop = acme.EnvironmentLoop(environment, agent)
  loop.run(num_episodes=FLAGS.episodes)


if __name__ == '__main__':
  app.run(main)
