import tensorflow as tf
import acme
from acme import specs
from IntrinsicRewards import MohammadDQN as dqn
from acme import wrappers
from IntrinsicRewards.MohammadDQN.EnvWrappers import SinglePrecisionFloatWrapper
import dm_env
import gym
from empax import evaluation
import typing

import numpy as np
import sonnet as snt
from gym_minigrid.wrappers import *
from matplotlib import pyplot as plt


def make_environment(envname: str = 'MiniGrid-Empty-8x8-v0') -> dm_env.Environment:

  environment = gym.make(envname)
  environment.reset()
  plt.figure()
  plt.imshow(environment.render())
  plt.show()
  environment = FullyObsWrapper(environment)  # Get full pixel observations
  environment = ImgObsWrapper(environment)  # Get rid of the 'mission' field


  """Creates an OpenAI Gym environment."""

  # Make sure the environment obeys the dm_env.Environment interface.
  environment = wrappers.GymWrapper(environment)
  # Clip the action returned by the agent to the environment spec.
  environment = SinglePrecisionFloatWrapper(environment)

  return environment

#network_function is one of the functions below, and generates all the networks for the model
#environment is the environment, i.e. the output of make_environment
#other_arguments are all the hyperparameters of the agent
def make_environment_loop(Qnet, qnet, featnet, rnet, feat_dims,
                          environment: dm_env.Environment,
                          eval_observer: evaluation.observers.EvaluationObserver,
                          sequence_length: int,
                          **other_arguments
                          )\
        -> (acme.EnvironmentLoop, acme.EnvironmentLoop):

    spec = specs.make_environment_spec(environment)

    agent = dqn.DQNEmpowerment(
        environment_spec=spec,
        Qnetwork=Qnet,
        qnetwork=qnet,
        feat_network=featnet,
        feat_dims=feat_dims,
        rnetwork=rnet,
        sequence_length=sequence_length,
        **other_arguments)
    loop = acme.EnvironmentLoop(environment, agent)

    eval_loop = acme.EnvironmentLoop(
        environment,
        actor=agent.eval_actor,
        logger=acme.utils.loggers.TerminalLogger(print_fn=print),
        observers=[eval_observer],
    )
    return loop, eval_loop



def make_networks_simple(action_spec: specs.DiscreteArray):
    def _make_Qnetwork(action_spec: specs.DiscreteArray) -> snt.Module:
        return snt.Sequential([
            snt.Flatten(),
            snt.nets.MLP([256, 256, action_spec.num_values]),
        ])

    def _make_qnetwork(
            action_spec: specs.DiscreteArray) -> snt.Module:  # takes in s + s' + action, spits out probability
        return dqn.ConditionalProductNetwork(output_dims=action_spec.num_values, categorical=True)

    def _make_feat_network(
            action_spec: specs.DiscreteArray) -> snt.Module:  # lol this just makes features, so we'll just flatten for now
        return snt.Sequential([snt.Conv2D(32, 4, 2), tf.nn.leaky_relu, snt.Conv2D(64, 4, 2), tf.nn.leaky_relu,
                               snt.Flatten(), snt.Linear(64)
                               ])

    def _make_rnetwork(
            action_spec: specs.DiscreteArray) -> snt.Module:  # takes in just s and action, spits out probability
        return dqn.RNetwork(output_dims=action_spec.num_values, categorical=True)

    return _make_Qnetwork(action_spec), _make_qnetwork(action_spec), _make_feat_network(action_spec), _make_rnetwork(action_spec), 64

