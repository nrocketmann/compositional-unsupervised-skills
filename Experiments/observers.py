"""Visualize Trajectory."""

from typing import Dict

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from empax import types
from empax.evaluation import observers, plot_utils
import sonnet as snt
import tensorflow as tf

class EmpowermentGraph(observers.EpisodeMetric):

  def __init__(self,
               r_network: snt.Module,
               feat_network: snt.Module,
               beta: float,
               num_actions: int
               ):
    self.reset()
    self.feat_network = feat_network
    self.r_network = r_network
    self.beta = beta
    self.num_actions = num_actions

  def reset(self):
    self._trajectories = []

  def compute_metric(self, trajectory: observers.Trajectory
                     ) -> Dict[str, types.Number]:  # yapf: disable
    print(trajectory.observations)
    print(trajectory.observations[0])

    state_feats = self.feat_network(np.asarray(trajectory.observations))
    if (trajectory.actions.shape.rank==2):
      actions = tf.one_hot(np.asarray(trajectory.actions),self.num_actions) #TODO: make this not fail in continuous 1D case
    else:
      actions = np.asarray(trajectory.actions)
    _, phi_val = self.r_network(actions, state_feats)
    intrinsic_rewards = tf.squeeze(phi_val * 1 / self.beta)

    self._trajectories.append((intrinsic_rewards)) #TODO: add place where agent is
    return {}

  def draw_plot(self, ax: Axes):
    ax.set_aspect('equal')

    # assign goal colors from the (continuous) goal vector
    num_trajectories = len(self._trajectories)
    goals = np.vstack([goal for _, _, goal in self._trajectories])
    assert goals.shape[0] == num_trajectories
    colors = plot_utils.get_option_colors(goals)

    L = 20
    for (xs, ys, color) in self._trajectories:
      L = max(L, np.max(xs))
      L = max(L, np.max(ys))
    L = L * 1.2
    ax.axis([-L, L, -L, L])

    for (xs, ys, _), color in zip(self._trajectories, colors):
      ax.plot(xs, ys, color, linewidth=0.7)

  def result(self) -> Dict[str, Figure]:
    """Aggregate the trajectories and draw in a single plot."""
    # fig = Figure()
    # ax: Axes = fig.add_subplot()  # type: ignore
    # self.draw_plot(ax)
    # return {"xy_trajectory": fig}
    print(self._trajectories)
    return {}