import numpy as np
import gym
from point_env import PointEnv
from typing import Optional 
from point_env import WALLS

class PointEnvExtras():
    """
    PointEnv with arbitrary extra per-cell features for BOTH the agent state
    and the goal state.

    Observation length = 4 + 2*extra_dim:
        [state_x, state_y,  extra_1…extra_N,
         goal_x , goal_y ,  goal_extra_1…goal_extra_N]
    """

    def __init__(
        self,
        walls: str = "Spiral11x11",
        resize_factor: int = 1,
        fixed_start_end=None,
        extra_dim: int = 8,
        seed = 0,
        rng: Optional[np.random.Generator] = None,  # ✔ works on 3.9
    ):

        if resize_factor > 1:
            self._walls = resize_walls(WALLS[walls], resize_factor)
        else:
            self._walls = WALLS[walls]
        (height, width) = self._walls.shape
        self._height = height
        self._width = width
        self._extra_dim = int(extra_dim)
        self._rng = rng or np.random.default_rng(seed)
        self._action_noise = 0.01

        # Random feature tensor, values ∈ [0, H]
        # self._features = self._rng.uniform(
        #     0.0,self._height , size=(self._height, self._width, self._extra_dim)
        # ).astype(np.float32)

        # Initialize features to all zeros
        self._features = np.zeros((self._height, self._width, self._extra_dim), dtype=np.float32)

        # Define special cells where features should be 1
        # Format: list of (row, col) positions (i.e., y, x)
        special_cells = [(6 , 1), (3 , 5), (4 , 2), (7 , 9-1), (0 , 6), (6 , 0), (8, 2)]  # <- Replace with your own coordinates

        # Set those cells to 1 in every feature dimension
        for (i, j) in special_cells:
             self._features[i, j, :] = np.ones(self._extra_dim) * 10 # or

        # Redefine the observation space (low/high vectors)
        low = np.concatenate(
            [
                np.array([0.0, 0.0]),
                np.zeros(self._extra_dim),
                np.array([0.0, 0.0]),
                np.zeros(self._extra_dim),
            ]
        )
        high = np.concatenate(
            [
                np.array([self._height, self._width]),
                np.full(self._extra_dim, self._height),
                np.array([self._height, self._width]),
                np.full(self._extra_dim, self._height),
            ]
        )
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Box(
        low=np.array([-1.0, -1.0]),
        high=np.array([1.0, 1.0]),
        dtype=np.float32)

        s, g = fixed_start_end
        si, sj = self._discretize_state(s)
        gi, gj = self._discretize_state(g)

        start = np.concatenate([s, self._features[si, sj]]).astype(np.float32)
        goal = np.concatenate([g, self._features[gi, gj]]).astype(np.float32)

        self._fixed_start_end = [start, goal]

        
        print("fixed_start_end", self._fixed_start_end)
        self._timestep = 0
        if '11x11' in walls:
            self._max_episode_steps = 100
        else:
            self._max_episode_steps = 50

        print("PointEnvExtras: walls", walls, "resize_factor", resize_factor, "extra_dim", extra_dim)
        self.reset()

       

    def _discretize_state(self, state, resolution=1.0):
        ij = np.floor(resolution * state).astype(int)
        ij = np.clip(ij, np.zeros(2), np.array(self.walls.shape) - 1)
        return ij.astype(int)

    def _is_blocked(self, state):
        """Check if (x, y) is inside the maze and not hitting a wall."""
        pos = state[:2]
        if (np.any(pos < self.observation_space.low[:2])
            or np.any(pos > self.observation_space.high[:2])):
            return True
        i, j = self._discretize_state(pos)
        return self._walls[i, j] == 1


       
    def _sample_empty_state(self):
        """Sample a free (x, y) cell and return full state with extra features."""
        candidate_states = np.where(self._walls == 0)
        num_candidate_states = len(candidate_states[0])
        idx = np.random.choice(num_candidate_states)

        # Random base (x, y) inside that free cell
        y, x = candidate_states[0][idx], candidate_states[1][idx]
        xy = np.array([y, x], dtype=float) + np.random.uniform(size=2)

        # Check it’s still unblocked (in case the wall grid is coarse)
        assert not self._is_blocked(xy)

        # Pull corresponding extra dims
        i, j = self._discretize_state(xy)
        extras = self._features[i, j]  # shape = (extra_dim,)

        return np.concatenate([xy, extras])


    def reset(self, random = True):
        self._timestep = 0
        
        if self._fixed_start_end is not None:
            #print("Using fixed start and end positions in the new enviornment")
            # fix the starting and ending position of the agent
            self.state = self._fixed_start_end[0]
            self.goal = self._fixed_start_end[1]
        else:
            self.goal = self._sample_empty_state()
            self.state = self._sample_empty_state()
        # if random:
        #     # print("initial state" , self.state)
        #     max_shift = 1.0  # Max amount to move in any direction
        #     for _ in range(100):  # Try multiple random moves in case some are blocked
        #         shift = np.random.uniform(low=-max_shift, high=max_shift, size=self.state.shape)
        #         new_state = self.state + shift
        #         if not self._is_blocked(new_state):
        #             self.state = new_state
        #             # print("new state", self.state)
        #             break
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.state, self.goal]).astype(np.float32)


    def step(self, action):
        action = action.copy()
        if not self.action_space.contains(action):
            print('WARNING: clipping invalid action:', action)
        if self._action_noise > 0:
            action += np.random.normal(0, self._action_noise, (2,))
        action = np.clip(action, self.action_space.low, self.action_space.high)
        assert self.action_space.contains(action)

        # Get current position
        xy = self.state[:2].copy()
        num_substeps = 10
        dt = 1.0 / num_substeps
        for _ in np.linspace(0, 1, num_substeps):
            for axis in range(2):  # x and y only
                proposed = xy.copy()
                proposed[axis] += dt * action[axis]
                if not self._is_blocked(proposed):
                    xy = proposed

        # Set new state: [x, y, extras...]
        i, j = self._discretize_state(xy)
        new_state = np.concatenate([xy, self._features[i, j]])
        self.state = new_state

        self._last_end_pos = self.state
        self._timestep += 1
        dist = np.linalg.norm(self.goal[:2] - self.state[:2])
        rew = float(dist < 1.0)
        done = False
        return self._get_obs(), rew, done, {}


    

    @property
    def walls(self):
        return self._walls
    



    