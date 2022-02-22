"""Visualize Trajectory and log intrinsic reward"""

from typing import Dict

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from empax import types
from empax.evaluation import observers, plot_utils
import sonnet as snt
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import wandb
import cv2

"""This graphs the intrinsic reward of the agent in an eval loop.
It also graphs where the agent spends its time.
Meant for gym-minigrid environments right now
"""
class EmpowermentGraph(observers.EpisodeMetric):

  def __init__(self,
               r_network: snt.Module,
               feat_network: snt.Module,
               beta: float,
               num_actions: int,
               out_directory: str,
               sequence_length: int = 10
               ):
    self.size = 25
    self.size_set = False
    self.reset()
    self.feat_network = feat_network
    self.r_network = r_network
    self.beta = beta
    self.num_actions = num_actions
    self.sequence_length = sequence_length
    self.out_directory = out_directory
    self.counter = 0
    #just a handy array to figure out where the character is



  def reset(self):
    #trajectories will be a heatmap of where the agent spends its time
    self._trajectories = np.zeros([self.size,self.size],dtype=np.float32)
    #intrinsic rewards will be a heatmap of intrinsic reward at each location
    self._intrinsic_rewards = np.zeros([self.size,self.size],dtype=np.float32)
    self.video_frames = None


  def compute_metric(self, trajectory: observers.Trajectory
                     ) -> Dict[str, types.Number]:  # yapf: disable
    if not self.size_set:
      self.size = trajectory.observations[0].shape[0]
      self.size_set = True
      self.reset()

    self.video_frames = trajectory.renders
    trajectory_length = len(trajectory.observations) - self.sequence_length
    if (trajectory_length<10):
      print("Short trajectory!")
      return {}
    observations = np.asarray(trajectory.observations[:trajectory_length])
    #shape should be 8 x 8 x 3
    #place with character has values [10,0,0]

    #update graph of trajectories based on where agent has been
    # locations_traveled = np.linalg.norm(observations -
    #                                     np.tile(self._character_map,[trajectory_length,1,1,1]),axis=-1)
    #shape [num_trajectories, 8, 8]
    #locations_traveled = np.where(locations_traveled>0,0,1) #shape [trajectory_length, 8, 8]
    locations_traveled = np.where(observations[...,0] ==1.0, 1, 0)
    locations_traveled_sum = np.sum(locations_traveled,axis=0)#shape [8,8]

    print("Should be 1.0: " + str(np.sum(locations_traveled_sum)/trajectory_length)) #should always be 1!
    self._trajectories += locations_traveled_sum

    state_feats = self.feat_network(observations)

  #TODO: This only works for 1d discrete actions, probably should adjust it
    #now we gotta get the actions lined up with states...
    input_actions = list([trajectory.actions[i:i+self.sequence_length] for i in range(trajectory_length)])
    actions = tf.one_hot(np.array(input_actions),self.num_actions)


    _, phi_val = self.r_network(actions, state_feats)
    intrinsic_rewards = tf.squeeze(phi_val * 1 / self.beta) #shape [trajectory_length]

    #now we want to add this to the heatmap of intrinsic reward
    self._intrinsic_rewards += tf.reduce_sum(tf.reshape(intrinsic_rewards, [trajectory_length, 1,1]) * locations_traveled,axis=0)

    return {}

  def draw_plot(self,frequencies,mean_rewards, log_frequencies):
    print("Drawing plots!")
    plt.figure()
    plt.title("frequencies")
    plt.imshow(frequencies,cmap='gray')
    plt.savefig(os.path.join(self.out_directory, 'frequencies' + str(self.counter) + '.jpg'))

    plt.figure()
    plt.title("Log frequencies")
    plt.imshow(log_frequencies,cmap='gray')
    plt.savefig(os.path.join(self.out_directory, 'log_frequencies' + str(self.counter) + '.jpg'))

    plt.figure()
    plt.title("mean rewards")
    plt.imshow(mean_rewards,cmap='gray')
    print("Nans appearing in reward? " + str(np.any(np.isnan(mean_rewards))))
    plt.savefig(os.path.join(self.out_directory, 'mean_rewards' + str(self.counter) + '.jpg'))

    #save video frame
    plt.figure()
    plt.title('environment')
    plt.imshow(np.transpose(self.video_frames[0],[1,0,2]))
    plt.savefig(os.path.join(self.out_directory, 'env_image' + str(self.counter) + '.jpg'))

    plt.show()


    self.counter+=1

    wandb.log({
      "frequency_graph": wandb.Image((np.expand_dims(frequencies,-1)*255).astype(int)),
      "log_frequency_graph":wandb.Image((np.expand_dims(log_frequencies,-1)*255).astype(int)),
      "reward_graph":wandb.Image((np.expand_dims(mean_rewards,-1)*255).astype(int)),
      "environment_graph": wandb.Image(np.transpose(self.video_frames[0],[1,0,2])),
      "video": wandb.Video(np.transpose(np.asarray(self.video_frames),[0,3,2,1]), fps=4, format="gif")
    })


  def result(self) -> Dict[str, Figure]:
    """Aggregate the trajectories and draw in a single plot."""
    frequencies = self._trajectories/np.max(self._trajectories)
    log_frequencies = np.log(self._trajectories + 1)
    log_frequencies = log_frequencies/np.max(log_frequencies)
    mean_rewards = self._intrinsic_rewards/(self._trajectories + 1e-7)
    mean_rewards = mean_rewards - np.min(mean_rewards)
    mean_rewards = mean_rewards/np.max(mean_rewards)
    self.draw_plot(frequencies,mean_rewards, log_frequencies)
    return {}