import copy
from typing import Optional

from acme import datasets
from acme import specs
from acme.adders import reverb as adders
from acme.agents import agent
from acme.agents.tf import actors
from IntrinsicRewards.MohammadDQN import learning
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import loggers
import reverb
import sonnet as snt
import tensorflow as tf
import trfl


class DQNEmpowerment(agent.Agent):
  """DQN agent.

  This implements a single-process DQN agent. This is a simple Q-learning
  algorithm that inserts N-step transitions into a replay buffer, and
  periodically updates its policy by sampling these transitions using
  prioritization.
  """

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      Qnetwork: snt.Module,
      qnetwork: snt.Module,
      feat_network: snt.Module,
      feat_dims: int,
      rnetwork: snt.Module,
      beta: float = 1.0,
      batch_size: int = 256,
      prefetch_size: int = 4,
      target_update_period: int = 100,
      samples_per_insert: float = 32.0,
      sequence_length = 10,
      replay_period=2,
      min_replay_size: int = 1000,
      max_replay_size: int = 1000000,
      importance_sampling_exponent: float = 0.2,
      priority_exponent: float = 0.6,
      n_step: int = 5,
      epsilon: Optional[tf.Variable] = None,
      learning_rate: float = 1e-3,
      discount: float = 0.99,
      logger: Optional[loggers.Logger] = None,
      checkpoint: bool = True,
      checkpoint_subpath: str = '~/acme',
      policy_network: Optional[snt.Module] = None,
      max_gradient_norm: Optional[float] = None,
      tflogs: str = None
  ):
    """Initialize the agent.

    Args:
      environment_spec: description of the actions, observations, etc.
      network: the online Q network (the one being optimized)
      batch_size: batch size for updates.
      prefetch_size: size to prefetch from replay.
      target_update_period: number of learner steps to perform before updating
        the target networks.
      samples_per_insert: number of samples to take from replay for every insert
        that is made.
      min_replay_size: minimum replay size before updating. This and all
        following arguments are related to dataset construction and will be
        ignored if a dataset argument is passed.
      max_replay_size: maximum replay size.
      importance_sampling_exponent: power to which importance weights are raised
        before normalizing.
      priority_exponent: exponent used in prioritized sampling.
      n_step: number of steps to squash into a single transition.
      epsilon: probability of taking a random action; ignored if a policy
        network is given.
      learning_rate: learning rate for the q-network update.
      discount: discount to use for TD updates.
      logger: logger object to be used by learner.
      checkpoint: boolean indicating whether to checkpoint the learner.
      checkpoint_subpath: string indicating where the agent should save
        checkpoints and snapshots.
      policy_network: if given, this will be used as the policy network.
        Otherwise, an epsilon greedy policy using the online Q network will be
        created. Policy network is used in the actor to sample actions.
      max_gradient_norm: used for gradient clipping.
      tflogs: a directory to write tensorboard logs for wandb
    """

    replay_table = reverb.Table(
        name=adders.DEFAULT_PRIORITY_TABLE,
        sampler=reverb.selectors.Prioritized(priority_exponent),
        remover=reverb.selectors.Fifo(),
        max_size=max_replay_size,
        rate_limiter=reverb.rate_limiters.MinSize(min_size_to_sample=1),
        signature=adders.SequenceAdder.signature(
            environment_spec, sequence_length=sequence_length))
    self._server = reverb.Server([replay_table], port=None)
    address = f'localhost:{self._server.port}'

    # Component to add things into replay.
    adder = adders.SequenceAdder(
        client=reverb.Client(address),
        period=replay_period,
        sequence_length=sequence_length,
    )

    # The dataset object to learn from.
    dataset = datasets.make_reverb_dataset(
        server_address=address,
        batch_size=batch_size,
        prefetch_size=prefetch_size)

    replay_client = reverb.Client(address)


    # Create epsilon greedy policy network by default.
    if policy_network is None:
      # Use constant 0.05 epsilon greedy policy by default.
      if epsilon is None:
        epsilon = tf.Variable(0.05, trainable=False)
      policy_network = snt.Sequential([
          Qnetwork,
          lambda q: trfl.epsilon_greedy(q, epsilon=epsilon).sample(),
      ])

    # Create a target network.
    target_Qnetwork = copy.deepcopy(Qnetwork)

    double_environment_spec = tf.TensorSpec(shape=[feat_dims * 2], dtype=tf.float32)
    single_environment_spec = tf.TensorSpec(shape=[feat_dims], dtype=tf.float32)
    action_spec = tf.TensorSpec(shape=[2,environment_spec.actions.num_values],dtype=tf.int32)
    special_environment_spec = tf.TensorSpec(shape=environment_spec.observations.shape,dtype=tf.float32)

    # Ensure that we create the variables before proceeding (maybe not needed).
    tf2_utils.create_variables(Qnetwork, [special_environment_spec])
    tf2_utils.create_variables(target_Qnetwork, [special_environment_spec])


    tf2_utils.create_variables(qnetwork, [action_spec,double_environment_spec])#,time_series=True)
    tf2_utils.create_variables(feat_network, [special_environment_spec])#,time_series=True)
    tf2_utils.create_variables(rnetwork, [action_spec, single_environment_spec])#,time_series=True)

    # Create the actor which defines how we take actions.
    actor = actors.FeedForwardActor(policy_network, adder)

    self.eval_actor = actors.FeedForwardActor(policy_network, None)

    # The learner updates the parameters (and initializes them).
    learner = learning.DQNEmpowermentLearner(
        Qnetwork=Qnetwork,
        qnetwork= qnetwork,
        feat_network=feat_network,
        rnetwork=rnetwork,
        target_Qnetwork=target_Qnetwork,
        discount=discount,
        beta=beta,
        importance_sampling_exponent=importance_sampling_exponent,
        learning_rate=learning_rate,
        target_update_period=target_update_period,
        dataset=dataset,
        replay_client=replay_client,
        max_gradient_norm=max_gradient_norm,
        logger=logger,
        checkpoint=checkpoint,
        save_directory=checkpoint_subpath,
        tflogs=tflogs)

    if checkpoint:
      self._checkpointer = tf2_savers.Checkpointer(
          directory=checkpoint_subpath,
          objects_to_save=learner.state,
          subdirectory='dqn_learner',
          time_delta_minutes=60.)
    else:
      self._checkpointer = None

    super().__init__(
        actor=actor,
        learner=learner,
        min_observations=max(batch_size, min_replay_size),
        observations_per_step=float(batch_size) / samples_per_insert)

  def update(self):
    super().update()
    if self._checkpointer is not None:
      self._checkpointer.save()
