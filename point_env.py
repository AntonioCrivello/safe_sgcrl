"""Utility for loading the 2D navigation environments."""
from typing import Optional

import gym
import numpy as np
import scipy

# Only Spiral11x11 is supported, but the other walls can easily be added 
# by adding the start and ending coordinates to the fixed_goal_dict in lp_contrastive.py
WALLS = { 
    'Small':  
        np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]),
    'Cross':  
        np.array([[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]]),
    'Impossible':
        np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 1, 1, 1, 1, 1, 0],
                  [0, 1, 0, 0, 0, 0, 1, 0, 0],
                  [0, 1, 1, 1, 1, 0, 1, 0, 1],
                  [0, 1, 0, 0, 0, 0, 1, 0, 0],
                  [0, 1, 0, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 1],
                  [0, 1, 0, 0, 0, 1, 0, 1, 0]]),
    'FourRooms':  
        np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]]),
    'U': 
        np.array([[0, 0, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [1, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 0, 0]]),
    'Spiral11x11': 
        np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                  [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
                  [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
                  [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
                  [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                  [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]),
    'Wall11x11':  
        np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
    'Maze11x11':  
        np.array([[0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                  [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
                  [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
}


def resize_walls(walls, factor):
  (height, width) = walls.shape
  row_indices = np.array([i for i in range(height) for _ in range(factor)])  # pylint: disable=g-complex-comprehension
  col_indices = np.array([i for i in range(width) for _ in range(factor)])  # pylint: disable=g-complex-comprehension
  walls = walls[row_indices]
  walls = walls[:, col_indices]
  assert walls.shape == (factor * height, factor * width)
  return walls


class PointEnv(gym.Env):
  """Abstract class for 2D navigation environments."""

  def __init__(
    self,
    walls = None, 
    resize_factor = 1, 
    region_bounds = None,
    fixed_start_end = None
    ):
    """Initialize the point environment.

    Args:
      walls: (str or array) binary, H x W array indicating locations of walls.
        Can also be the name of one of the maps defined above.
      resize_factor: (int) Scale the map by this factor.
      region_bounds: absorbing region bounds
    """
    if resize_factor > 1:
      self._walls = resize_walls(WALLS[walls], resize_factor)
    else:
      self._walls = WALLS[walls]
    (height, width) = self._walls.shape
    self._height = height
    self._width = width
    self._action_noise = 0.01
    self._fixed_start_end = fixed_start_end
    self.action_space = gym.spaces.Box(
        low=np.array([-1.0, -1.0]),
        high=np.array([1.0, 1.0]),
        dtype=np.float32)
    self.observation_space = gym.spaces.Box(
        low=np.array([0, 0, 0, 0]),
        high=np.array([height, width, height, width]),
        dtype=np.float32)
    self._timestep = 0
    
    # Initialize values for absorbing region
    self.region_bounds = region_bounds
    self._is_absorbed = False

    if '11x11' in walls:
      self._max_episode_steps = 100
    else:
      self._max_episode_steps = 50
    self.reset()

  def _in_absorbing_region(self):
    if self.region_bounds is None:
      return False
    lower, upper = self.region_bounds
    return np.all(self.state >= lower) and np.all(self.state <= upper)

  def _sample_empty_state(self):
    #TODO 
    # Future should check that initial state is not already in absorbing state
    candidate_states = np.where(self._walls == 0)
    num_candidate_states = len(candidate_states[0])
    state_index = np.random.choice(num_candidate_states)
    state = np.array([candidate_states[0][state_index],
                      candidate_states[1][state_index]],
                     dtype=float)
    state += np.random.uniform(size=2)
    assert not self._is_blocked(state)
    return state

  def _get_obs(self):
    return np.concatenate([self.state, self.goal]).astype(np.float32)

  def reset(self, random = True):
    self._timestep = 0
    # Initialize to not in absorbing state
    self._is_absorbed = False
    
    if self._fixed_start_end is not None:
        # fix the starting and ending position of the agent
        self.state = self._fixed_start_end[0]
        self.goal = self._fixed_start_end[1]
    else:
        self.goal = self._sample_empty_state()
        self.state = self._sample_empty_state()
    if random:
        # print("initial state" , self.state)
        max_shift = 0.1  # Max amount to move in any direction
        for _ in range(100):  # Try multiple random moves in case some are blocked
            shift = np.random.uniform(low=-max_shift, high=max_shift, size=self.state.shape)
            new_state = self.state + shift
            if not self._is_blocked(new_state):
                self.state = new_state
                # print("new state", self.state)
                break
    return self._get_obs()

  def _discretize_state(self, state, resolution=1.0):
    ij = np.floor(resolution * state).astype(int)
    ij = np.clip(ij, np.zeros(2), np.array(self.walls.shape) - 1)
    return ij.astype(int)

  def _is_blocked(self, state):
    assert len(state) == 2
    if (np.any(state < self.observation_space.low[:2])
        or np.any(state > self.observation_space.high[:2])):
      return True
    (i, j) = self._discretize_state(state)
    return (self._walls[i, j] == 1)

  def step(self, action):
    if self._is_absorbed:
      return self._get_obs(), 0.0, False, {'absorbed': True}
    
    action = action.copy()
    if not self.action_space.contains(action):
      print('WARNING: clipping invalid action:', action)
    if self._action_noise > 0:
      action += np.random.normal(0, self._action_noise, (2,))
    action = np.clip(action, self.action_space.low, self.action_space.high)
    assert self.action_space.contains(action)
    num_substeps = 10
    dt = 1.0 / num_substeps
    num_axis = len(action)
    for _ in np.linspace(0, 1, num_substeps):
      for axis in range(num_axis):
        new_state = self.state.copy()
        new_state[axis] += dt * action[axis]
        if not self._is_blocked(new_state):
          self.state = new_state
    if self._in_absorbing_region():
      self._is_absorbed = True

    done = False
    obs = self._get_obs()
    self._last_end_pos = self.state
    self._timestep += 1
    
    if self._is_absorbed:
      rew = 0.0
    else:
      dist = np.linalg.norm(self.goal - self.state)
      rew = float(dist < 1.0)
    return obs, rew, done, {'absorbed': self._is_absorbed}

  @property
  def walls(self):
    return self._walls