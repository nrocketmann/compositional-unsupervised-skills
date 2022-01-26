
"""Environment wrapper which converts ints to floats and does single precision"""

from acme import specs
from acme import types
from acme.wrappers import base

import dm_env
import numpy as np
import tree
import gym


class SinglePrecisionFloatWrapper(base.EnvironmentWrapper):
  """Wrapper which converts environments from double- to single-precision."""

  def _convert_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    return timestep._replace(
        reward=_convert_value(timestep.reward),
        discount=_convert_value(timestep.discount),
        observation=_convert_value(timestep.observation,force_float=True))

  def step(self, action) -> dm_env.TimeStep:
    return self._convert_timestep(self._environment.step(action))

  def reset(self) -> dm_env.TimeStep:
    return self._convert_timestep(self._environment.reset())

  def action_spec(self):
    return _convert_spec(self._environment.action_spec())

  def discount_spec(self):
    return _convert_spec(self._environment.discount_spec())

  def observation_spec(self):
    return _convert_spec(self._environment.observation_spec(),force_float=True)

  def reward_spec(self):
    return _convert_spec(self._environment.reward_spec())


def _convert_spec(nested_spec: types.NestedSpec, force_float: bool = False) -> types.NestedSpec:
  """Convert a nested spec."""

  def _convert_single_spec(spec: specs.Array):
    """Convert a single spec."""
    if spec.dtype == 'O':
      # Pass StringArray objects through unmodified.
      return spec
    if force_float:
      dtype = np.float32
    elif np.issubdtype(spec.dtype, np.float64):
      dtype = np.float32
    elif np.issubdtype(spec.dtype, np.int64):
      dtype = np.int32
    else:
      dtype = spec.dtype
    return spec.replace(dtype=dtype)

  return tree.map_structure(_convert_single_spec, nested_spec)


def _convert_value(nested_value: types.Nest, force_float: bool = False) -> types.Nest:
  """Convert a nested value given a desired nested spec."""

  def _convert_single_value(value):
    if value is not None:
      value = np.array(value, copy=False)
      if force_float:
          value = np.array(value, copy=False, dtype=np.float32)
      else:
          if np.issubdtype(value.dtype, np.float64):
            value = np.array(value, copy=False, dtype=np.float32)
          elif np.issubdtype(value.dtype, np.int64):
            value = np.array(value, copy=False, dtype=np.int32)

    return value


  return tree.map_structure(_convert_single_value, nested_value)