import time
from typing import Dict, List, Optional, Union

import acme
from acme import types
from acme.adders import reverb as adders
from acme.tf import losses
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
import trfl
import wandb

class DQNEmpowermentLearner(acme.Learner, tf2_savers.TFSaveable):
  """DQN learner.

  This is the learning component of a DQN agent. It takes a dataset as input
  and implements update functionality to learn from this dataset. Optionally
  it takes a replay client as well to allow for updating of priorities.
  """

  def __init__(
      self,
      Qnetwork: snt.Module,
      qnetwork: snt.Module,
      feat_network: snt.Module,
      rnetwork: snt.Module,
      target_Qnetwork: snt.Module,
      discount: float,
      beta: float,
      importance_sampling_exponent: float,
      learning_rate: float,
      target_update_period: int,
      dataset: tf.data.Dataset,
      huber_loss_parameter: float = 1.,
      replay_client: Optional[Union[reverb.Client, reverb.TFClient]] = None,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      checkpoint: bool = True,
      save_directory: str = '~/acme',
      max_gradient_norm: Optional[float] = None,
      tflogs: str = None
  ):
    """Initializes the learner.

    Args:
      network: the online Q network (the one being optimized)
      target_network: the target Q critic (which lags behind the online net).
      discount: discount to use for TD updates.
      importance_sampling_exponent: power to which importance weights are raised
        before normalizing.
      learning_rate: learning rate for the q-network update.
      target_update_period: number of learner steps to perform before updating
        the target networks.
      dataset: dataset to learn from, whether fixed or from a replay buffer (see
        `acme.datasets.reverb.make_dataset` documentation).
      huber_loss_parameter: Quadratic-linear boundary for Huber loss.
      replay_client: client to replay to allow for updating priorities.
      counter: Counter object for (potentially distributed) counting.
      logger: Logger object for writing logs to.
      checkpoint: boolean indicating whether to checkpoint the learner.
      save_directory: string indicating where the learner should save
        checkpoints and snapshots.
      max_gradient_norm: used for gradient clipping.
      tflogs: a directory for tf summary logging for wandb
    """
    self.beta = beta

    # TODO(mwhoffman): stop allowing replay_client to be passed as a TFClient.
    # This is just here for backwards compatability for agents which reuse this
    # Learner and still pass a TFClient instance.
    if isinstance(replay_client, reverb.TFClient):
      # TODO(b/170419518): open source pytype does not understand this
      # isinstance() check because it does not have a way of getting precise
      # type information for pip-installed packages.
      replay_client = reverb.Client(replay_client._server_address)  # pytype: disable=attribute-error

    # Internalise agent components (replay buffer, networks, optimizer).
    # TODO(b/155086959): Fix type stubs and remove.
    self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types
    self._Qnetwork = Qnetwork
    self._target_Qnetwork = target_Qnetwork
    self._optimizerQ = snt.optimizers.Adam(learning_rate)
    self._replay_client = replay_client

    self._rnetwork = rnetwork
    self._qnetwork = qnetwork
    self._feat_network = feat_network

    self._optimizerq = snt.optimizers.Adam(learning_rate)
    self._optimizer_feat = snt.optimizers.Adam(learning_rate)
    self._optimizerr = snt.optimizers.Adam(learning_rate)

    # Make sure to initialize the optimizer so that its variables (e.g. the Adam
    # moments) are included in the state returned by the learner (which can then
    # be checkpointed and restored).
    self._optimizerQ._initialize(Qnetwork.trainable_variables)  # pylint: disable= protected-access
    self._optimizerq._initialize(qnetwork.trainable_variables)
    self._optimizer_feat._initialize(feat_network.trainable_variables)
    self._optimizerr._initialize(rnetwork.trainable_variables)

    # Internalise the hyperparameters.
    self._discount = discount
    self._target_update_period = target_update_period
    self._importance_sampling_exponent = importance_sampling_exponent
    self._huber_loss_parameter = huber_loss_parameter
    if max_gradient_norm is None:
      max_gradient_norm = 1e10  # A very large number. Infinity results in NaNs.
    self._max_gradient_norm = tf.convert_to_tensor(max_gradient_norm)

    # Learner state.
    self._Qvariables: List[List[tf.Tensor]] = [Qnetwork.trainable_variables]
    self._qvariables: List[List[tf.Tensor]] = [qnetwork.trainable_variables]
    self._feat_variables: List[List[tf.Tensor]] = [feat_network.trainable_variables]
    self._rvariables: List[List[tf.Tensor]] = [rnetwork.trainable_variables]
    self._variables = self._Qvariables + self._qvariables + self._feat_variables + self._rvariables
    self._num_steps = tf.Variable(0, dtype=tf.int32)

    # Internalise logging/counting objects.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)

    # Create a snapshotter object.
    if checkpoint:
      self._snapshotter = tf2_savers.Snapshotter(
          objects_to_save={'Qnetwork': Qnetwork,
                           'qnetwork':qnetwork,
                           'feat_network':feat_network,
                           'rnetwork':rnetwork},
          directory=save_directory,
          time_delta_minutes=60.)
    else:
      self._snapshotter = None

    if tflogs:
      self.tf_summary_writer = tf.summary.create_file_writer(tflogs)
    else:
      self.tf_summary_writer = None

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None
  #TODO: implement step
  @tf.function
  def _step(self) -> Dict[str, tf.Tensor]:
    """Do a step of SGD and update the priorities."""

    # Pull out the data needed for updates/priorities.
    inputs = next(self._iterator)
    transitions: types.Transition = inputs.data
    keys, probs = inputs.info[:2]


    if (transitions.action.shape.rank==2):
      action = tf.one_hot(transitions.action,self._qnetwork.output_dims) #TODO: make this not fail in continuous 1D case
    else:
      action = transitions.action



    with tf.GradientTape(persistent=True) as tape:
      q_tm1 = self._Qnetwork(transitions.observation[:,0])
      q_t_value = self._target_Qnetwork(transitions.observation[:,1]) #next observation
      q_t_selector = self._Qnetwork(transitions.observation[:,1]) #next observation

      feature_s = self._feat_network(transitions.observation[:,0])
      feature_sprime = self._feat_network(transitions.observation[:,-1])
      s_sprime = tf.concat([feature_s, feature_sprime],axis=-1)

      q_preds = self._qnetwork(action, s_sprime)
      r_preds, phi_val = self._rnetwork(action, feature_s)

      intrinsic_reward = tf.squeeze(phi_val * 1/self.beta)
      #tf.print(intrinsic_reward)

      q_loss = -tf.reduce_mean(q_preds)
      r_loss = tf.reduce_mean(tf.math.squared_difference(self.beta * q_preds, r_preds))
      feat_loss = r_loss + q_loss

      # The rewards and discounts have to have the same type as network values.
      r_t = tf.cast(intrinsic_reward, q_tm1.dtype)
      r_t = tf.clip_by_value(r_t, -1., 1.)
      d_t = tf.cast(transitions.discount[:,0], q_tm1.dtype) * tf.cast(
          self._discount, q_tm1.dtype)

      # Compute the loss.
      _, extra = trfl.double_qlearning(q_tm1, tf.squeeze(transitions.action[:,0]), r_t, d_t,
                                       q_t_value, q_t_selector)
      loss = losses.huber(extra.td_error, self._huber_loss_parameter)

      # Get the importance weights.
      importance_weights = 1. / probs  # [B]
      importance_weights **= self._importance_sampling_exponent
      importance_weights /= tf.reduce_max(importance_weights)

      # Reweight.
      loss *= tf.cast(importance_weights, loss.dtype)  # [B]
      loss = tf.reduce_mean(loss, axis=[0])  # []

      if self.tf_summary_writer is not None:
        with self.tf_summary_writer.as_default():
          tf.summary.scalar('loss_Q', loss)
          tf.summary.scalar('loss_q', q_loss)
          tf.summary.scalar('loss_r', r_loss)
          tf.summary.scalar('loss_feat', feat_loss)
          tf.summary.scalar('intrinsic_reward', tf.reduce_mean(intrinsic_reward))
          tf.summary.scalar('mean_qval', tf.reduce_mean(q_t_value))
        self.tf_summary_writer.flush()
        wandb.tensorflow.log(tf.summary.merge_all())

    qgradients = tape.gradient(q_loss, self._qnetwork.trainable_variables)
    rgradients = tape.gradient(r_loss, self._rnetwork.trainable_variables)
    featgradients = tape.gradient(feat_loss, self._feat_network.trainable_variables)

    # Do a step of SGD.
    gradients = tape.gradient(loss, self._Qnetwork.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, self._max_gradient_norm)
    self._optimizerQ.apply(gradients, self._Qnetwork.trainable_variables)

    qgradients, _ = tf.clip_by_global_norm(qgradients, self._max_gradient_norm)
    rgradients, _ = tf.clip_by_global_norm(rgradients, self._max_gradient_norm)
    featgradients, _ = tf.clip_by_global_norm(featgradients, self._max_gradient_norm)
    self._optimizerq.apply(qgradients, self._qnetwork.trainable_variables)
    self._optimizerr.apply(rgradients, self._rnetwork.trainable_variables)
    self._optimizer_feat.apply(featgradients, self._feat_network.trainable_variables)

    # Get the priorities that we'll use to update.
    priorities = tf.abs(extra.td_error)

    # Periodically update the target network.
    if tf.math.mod(self._num_steps, self._target_update_period) == 0:
      for src, dest in zip(self._Qnetwork.variables,
                           self._target_Qnetwork.variables):
        dest.assign(src)
    self._num_steps.assign_add(1)

    # Report loss & statistics for logging.
    fetches = {
        'loss_Q': loss,
        'loss_q': q_loss,
        'loss_r': r_loss,
        'loss_feat': feat_loss,
        'keys': keys,
        'priorities': priorities,
        'intrinsic_reward': tf.reduce_mean(intrinsic_reward),
        'mean_qval': tf.reduce_mean(q_t_value)
    }

    return fetches

  def step(self):
    # Do a batch of SGD.
    result = self._step()

    # Get the keys and priorities.
    keys = result.pop('keys')
    priorities = result.pop('priorities')

    # Update the priorities in the replay buffer.
    if self._replay_client:
      self._replay_client.mutate_priorities(
          table=adders.DEFAULT_PRIORITY_TABLE,
          updates=dict(zip(keys.numpy(), priorities.numpy())))

    # Compute elapsed time.
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp

    # Update our counts and record it.
    counts = self._counter.increment(steps=1, walltime=elapsed_time)
    result.update(counts)

    # Snapshot and attempt to write logs.
    if self._snapshotter is not None:
      self._snapshotter.save()
    self._logger.write(result)

  def get_variables(self, names: List[str]) -> List[np.ndarray]:
    return tf2_utils.to_numpy(self._variables)

  @property
  def state(self):
    """Returns the stateful parts of the learner for checkpointing."""
    return {
        'Qnetwork': self._Qnetwork,
        'target_Qnetwork': self._target_Qnetwork,
        'qnetwork':self._qnetwork,
        'feat_network':self._feat_network,
        'rnetwork':self._rnetwork,
        'optimizerQ': self._optimizerQ,
        'optimizerq': self._optimizerq,
        'optimizer_feat': self._optimizer_feat,
        'optimizerr': self._optimizerr,
        'num_steps': self._num_steps
    }