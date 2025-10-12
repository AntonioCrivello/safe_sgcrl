"""Utilities for the contrastive RL agent."""
import functools
from typing import Dict
from typing import Optional, Sequence
import re
import jax
from acme import types
from acme.agents.jax import actors
from acme.jax import networks as network_lib
from acme.jax import utils
from acme.utils.observers import base as observers_base
from acme.wrappers import base
from acme.wrappers import canonical_spec
from acme.wrappers import gym_wrapper
from acme.wrappers import step_limit
import dm_env
import env_utils
import jax
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import os, json, numpy as np
from acme.utils import observers as observers_base  # same base as your SuccessObserver

def obs_to_goal_1d(obs, start_index, end_index):
  assert len(obs.shape) == 1
  return obs_to_goal_2d(obs[None], start_index, end_index)[0]


def obs_to_goal_2d(obs, start_index, end_index):
  assert len(obs.shape) == 2
  if end_index == -1:
    return obs[:, start_index:]
  else:
    return obs[:, start_index:end_index]


class SuccessObserver(observers_base.EnvLoopObserver):
  """Measures success by whether any of the rewards in an episode are positive.
  """

  def __init__(self):
    self._rewards = []
    self._success = []

  def observe_first(self, env, timestep
                    ):
    """Observes the initial state."""
    if self._rewards:
      success = np.sum(self._rewards) >= 1
      self._success.append(success)
    self._rewards = []

  def observe(self, env, timestep,
              action):
    """Records one environment step."""
    assert timestep.reward in [0, 1]
    self._rewards.append(timestep.reward)

  def get_metrics(self):
    """Returns metrics collected for the current episode."""
    return {
        'success': float(np.sum(self._rewards) >= 1),
        'success_1000': np.mean(self._success[-1000:]),
    }
  




class RegionVisitObserver(observers_base.EnvLoopObserver):
  """
  Counts how many steps per episode the (x, y) position (from the first 2 dims
  of the observation) lies inside one or more rectangular regions. If
  region_bounds is None, two default regions are tracked; otherwise, a single
  provided region is tracked.
  """

  def __init__(self,
               region_bounds,          # None OR ((x0,y0), (x1,y1)) as lower/upper corners
               env,
               seed,
               save_every=1000,
               save_dir="safety_region_visits_data"):
    # --- define regions ---
    self.regions = []
    if region_bounds is not None:
      x_min, y_min = float(region_bounds[0][0]), float(region_bounds[0][1])
      x_max, y_max = float(region_bounds[1][0]), float(region_bounds[1][1])
      self.regions.append({"x": (x_min, x_max), "y": (y_min, y_max)})
    else:
      self.regions.append({"x": (1.0, 4.0),  "y": (3.0, 10.0)})  # region 0

    self.n_regions = len(self.regions)
    desc = ", ".join(
        [f"R{i}: x=[{r['x'][0]}, {r['x'][1]}], y=[{r['y'][0]}, {r['y'][1]}]"
         for i, r in enumerate(self.regions)]
    )
    print(f"[RegionVisitObserver] Tracking {self.n_regions} region(s): {desc}", flush=True)

    self.save_every = int(save_every)
    print(f"env: {env}, seed: {seed}, save_every: {self.save_every}", flush=True)

    # Ensure the base save directory exists first
    os.makedirs(save_dir, exist_ok=True)
    
    # Root directory: experiments/safety_region_visits/<env>_<seed>/
    self._save_root = os.path.join(save_dir, f"{env}_{seed}")
    os.makedirs(self._save_root, exist_ok=True)

    # episode bookkeeping
    self._eps_seen = 0                                            # total finalized episodes so far
    self._buffer_per_region = [[] for _ in range(self.n_regions)] # only keep the last chunk
    self._in_ep = False
    self._counts_cur = np.zeros(self.n_regions, dtype=int)        # per-episode counters

  # -------- helpers --------
  def _extract_xy(self, obs):
    """Return (x, y) from the first two elements of the observation."""
    if isinstance(obs, dict):
      for k in ("observation", "obs", "state", "position"):
        if k in obs:
          obs = obs[k]
          break
    arr = np.asarray(obs).ravel()
    if arr.size < 2:
      raise ValueError("Observation must have at least 2 elements to extract (x, y).")
    return float(arr[0]), float(arr[1])

  @staticmethod
  def _inside_rect(x, y, rx, ry):
    """Inclusive rectangle test."""
    return (rx[0] <= x <= rx[1]) and (ry[0] <= y <= ry[1])

  def _save_buffer_block(self):
    """Write the current buffered episodes to a new file and clear the buffer."""
    block_len = len(self._buffer_per_region[0])
    if block_len == 0:
      return
    end_ep = self._eps_seen - 1
    start_ep = self._eps_seen - block_len
    fname = f"region_visits_{start_ep:06d}-{end_ep:06d}.json"
    fpath = os.path.join(self._save_root, fname)
    os.makedirs(self._save_root, exist_ok=True)

    data = {
      "episodes": list(range(start_ep, end_ep + 1)),
      "regions": [
        {"x": list(self.regions[i]["x"]), "y": list(self.regions[i]["y"])}
        for i in range(self.n_regions)
      ],
      # per-region lists for just this block:
      "counts_per_region": {
        f"region{i}": self._buffer_per_region[i] for i in range(self.n_regions)
      }
    }
    tmp = fpath + ".tmp"
    with open(tmp, "w") as f:
      json.dump(data, f)
    os.replace(tmp, fpath)
    print(f"[RegionVisitObserver] Saved block {start_ep}-{end_ep} → {fpath}", flush=True)

    # clear buffer to keep memory small
    self._buffer_per_region = [[] for _ in range(self.n_regions)]

  def _finalize_episode(self):
    """Append current counts to the buffer and maybe save a block."""
    for i in range(self.n_regions):
      self._buffer_per_region[i].append(int(self._counts_cur[i]))
    self._eps_seen += 1

    # Save only when we've accumulated `save_every` episodes.
    if (self._eps_seen % self.save_every) == 0:
      self._save_buffer_block()

  # -------- EnvLoopObserver API --------
  def observe_first(self, env, timestep):
    """Called at the beginning of an episode."""
    if self._in_ep:
      self._finalize_episode()
    self._in_ep = True
    self._counts_cur[:] = 0

  def observe(self, env, timestep, action):
    """Called on each step: count membership for each region separately."""
    x, y = self._extract_xy(timestep.observation)
    for i in range(self.n_regions):
      rx, ry = self.regions[i]["x"], self.regions[i]["y"]
      if self._inside_rect(x, y, rx, ry):
        self._counts_cur[i] += 1

  def get_metrics(self):
    """(Keeping your original behavior: return empty dict here.)"""
    return {}


class DistanceObserver(observers_base.EnvLoopObserver):
  """Observer that measures the L2 distance to the goal."""

  def __init__(self, obs_dim, start_index, end_index,
               smooth = True):
    self._distances = []
    self._obs_dim = obs_dim
    self._obs_to_goal = functools.partial(
        obs_to_goal_1d, start_index=start_index, end_index=end_index)
    self._smooth = smooth
    self._history = {}

  def _get_distance(self, env,
                    timestep):
    if hasattr(env, '_dist'):
      assert env._dist  # pylint: disable=protected-access
      return env._dist[-1]  # pylint: disable=protected-access
    else:
      # Note that the timestep comes from the environment, which has already
      # had some goal coordinates removed.
      obs = timestep.observation[:self._obs_dim]
      goal = timestep.observation[self._obs_dim:]
      dist = np.linalg.norm(self._obs_to_goal(obs) - goal)
      return dist

  def observe_first(self, env, timestep
                    ):
    """Observes the initial state."""
    if self._smooth and self._distances:
      for key, value in self._get_current_metrics().items():
        self._history[key] = self._history.get(key, []) + [value]
    self._distances = [self._get_distance(env, timestep)]

  def observe(self, env, timestep,
              action):
    """Records one environment step."""
    self._distances.append(self._get_distance(env, timestep))

  def _get_current_metrics(self):
    metrics = {
        'init_dist': self._distances[0],
        'final_dist': self._distances[-1],
        'delta_dist': self._distances[0] - self._distances[-1],
        'min_dist': min(self._distances),
    }
    return metrics

  def get_metrics(self):
    """Returns metrics collected for the current episode."""
    metrics = self._get_current_metrics()
    if self._smooth:
      for key, vec in self._history.items():
        for size in [10, 100, 1000]:
          metrics['%s_%d' % (key, size)] = np.nanmean(vec[-size:])
    return metrics


class ObservationFilterWrapper(base.EnvironmentWrapper):
  """Wrapper that exposes just the desired goal coordinates."""

  def __init__(self, environment,
               idx):
    """Initializes a new ObservationFilterWrapper.

    Args:
      environment: Environment to wrap.
      idx: Sequence of indices of coordinates to keep.
    """
    super().__init__(environment)
    self._idx = idx
    observation_spec = environment.observation_spec()
    spec_min = self._convert_observation(observation_spec.minimum)
    spec_max = self._convert_observation(observation_spec.maximum)
    self._observation_spec = dm_env.specs.BoundedArray(
        shape=spec_min.shape,
        dtype=spec_min.dtype,
        minimum=spec_min,
        maximum=spec_max,
        name='state')

  def _convert_observation(self, observation):
    return observation[self._idx]

  def step(self, action):
    timestep = self._environment.step(action)
    return timestep._replace(
        observation=self._convert_observation(timestep.observation))

  def reset(self):
    timestep = self._environment.reset()
    return timestep._replace(
        observation=self._convert_observation(timestep.observation))

  def observation_spec(self):
    return self._observation_spec


def make_environment(env_name, start_index, end_index,
                     seed, fixed_start_end = None, extra_dim = 8):
  """Creates the environment.

  Args:
    env_name: name of the environment
    start_index: first index of the observation to use in the goal.
    end_index: final index of the observation to use in the goal. The goal
      is then obs[start_index:goal_index].
    seed: random seed.
  Returns:
    env: the environment
    obs_dim: integer specifying the size of the observations, before
      the start_index/end_index is applied.
  """
  np.random.seed(seed)
  gym_env, obs_dim, max_episode_steps = env_utils.load(env_name, fixed_start_end, extra_dim)
  goal_indices = obs_dim + obs_to_goal_1d(np.arange(obs_dim), start_index,
                                          end_index)
  indices = np.concatenate([
      np.arange(obs_dim),
      goal_indices
  ])
  env = gym_wrapper.GymWrapper(gym_env)
  env = step_limit.StepLimitWrapper(env, step_limit=max_episode_steps)
  env = ObservationFilterWrapper(env, indices)
  return env, obs_dim



class InitiallyRandomActor(actors.GenericActor):
  """Actor that takes actions uniformly at random until the actor is updated.
  """

  def select_action(self,
                    observation):
      # ── helper ---------------------------------------------------------
    def _first_linear0_bias_is_zero(param_tree) -> bool:
      """Return True iff the first bias tensor of a *linear_0 module* is all-zeros.
      Works for both MLP and ResidualMLP trunks because it searches by regex."""
      for name, subdict in param_tree.items():
        if re.search(r'/linear_0$', name) and isinstance(subdict, dict) and 'b' in subdict:
          return (subdict['b'] == 0).all()
      # Fallback: if we didn’t find such a module, assume weights are *not* zeros.
      return False
      # print("param tree structure: {}", jax.tree_util.tree_structure(self._params), flush=True)
      # if (self._params[0]['mlp/~/linear_0']['b'] == 0).all():
    
    
    params_root = self._params[0]              # same as before

    if _first_linear0_bias_is_zero(params_root):
      # print("Using random actions because first linear_0 bias is zero.",
      #       flush=True)
      shape = self._params[0]['Normal/~/linear']['b'].shape
      #print("Action shape: {}".format(shape), flush=True)
      rng, self._state = jax.random.split(self._state)
      action = jax.random.uniform(key=rng, shape=shape,
                                  minval=-1.0, maxval=1.0)
    else:
      # print("param tree structure: {}", jax.tree_util.tree_structure(self._params), flush=True)
      action, self._state = self._policy(self._params, observation,
                                         self._state)
    return utils.to_numpy(action)