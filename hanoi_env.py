import gym
from gym import error, spaces, utils
from gym.utils import seeding

import random
import itertools
import numpy as np


class HanoiEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.num_disks = 4
        self.env_noise = 0
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Tuple(self.num_disks*(spaces.Discrete(3),))

        self.current_state = None
        self.goal_state = self.num_disks*(2,)

        self.done = None
        self.ACTION_LOOKUP = {0 : "(0,1) - top disk of pole 0 to top of pole 1 ",
                              1 : "(0,2) - top disk of pole 0 to top of pole 2 ",
                              2 : "(1,0) - top disk of pole 1 to top of pole 0",
                              3 : "(1,2) - top disk of pole 1 to top of pole 2",
                              4 : "(2,0) - top disk of pole 2 to top of pole 0",
                              5 : "(2,1) - top disk of pole 2 to top of pole 1"}

    def step(self, action):
        """
        * Inputs:
            - action: integer from 0 to 5 (see ACTION_LOOKUP)
        * Outputs:
            - current_state: state after transition
            - reward: reward from transition
            - done: episode state
            - info: dict of booleans (noisy?/invalid action?)
        0. Check if transition is noisy or not
        1. Transform action (0 to 5 integer) to tuple move - see Lookup
        2. Check if move is allowed
        3. If it is change corresponding entry | If not return same state
        4. Check if episode completed and return
        """
        if self.done:
            return self.current_state, 100, self.done, {"transition_failure": False, "invalid_action": False}

        info = {"transition_failure": False,
                "invalid_action": False}

        if self.env_noise > 0:
            r_num = random.random()
            if r_num <= self.env_noise:
                action = random.randint(0, self.action_space.n-1)
                info["transition_failure"] = True

        move = action_to_move[action]

        if self.move_allowed(move):
            disk_to_move = min(self.disks_on_peg(move[0]))
            moved_state = list(self.current_state)
            moved_state[disk_to_move] = move[1]
            self.current_state = tuple(moved_state)
        else:
            info["invalid_action"] = True

        if self.current_state == self.goal_state:
            reward = 100
            self.done = True
        elif info["invalid_action"] == True:
            reward = -1
        else:
            reward = 0

        return self.current_state, reward, self.done, info

    def disks_on_peg(self, peg):
        """
        * Inputs:
            - peg: pole to check how many/which disks are in it
        * Outputs:
            - list of disk numbers that are allocated on pole
        """
        return [disk for disk in range(self.num_disks) if self.current_state[disk] == peg]

    def move_allowed(self, move):
        """
        * Inputs:
            - move: tuple of state transition (see ACTION_LOOKUP)
        * Outputs:
            - boolean indicating whether action is allowed from state!
        move[0] - peg from which we want to move disc
        move[1] - peg we want to move disc to
        Allowed if:
            * discs_to is empty (no disc of peg) set to true
            * Smallest disc on target pole larger than smallest on prev
        """
        disks_from = self.disks_on_peg(move[0])
        disks_to = self.disks_on_peg(move[1])

        if disks_from:
            return (min(disks_to) > min(disks_from)) if disks_to else True
        else:
            return False

    def reset(self):
        self.current_state = self.num_disks * (0,)
        self.done = False
        return self.current_state

    def render(self, mode='human', close=False):
        """
        Renders the current state of the Hanoi environment.
        Returns an ASCII art representation of the towers and disks.
        
        Args:
            mode (str): The mode to render with. Currently only 'human' is supported.
            close (bool): Whether to close the rendering. Not used.
            
        Returns:
            str: ASCII representation of the current state if mode is 'human'
        """
        if close:
            return
            
        if mode != 'human':
            raise NotImplementedError(f"Mode {mode} not supported, only 'human' mode is available")
            
        # Get disks on each peg
        peg_0_disks = sorted(self.disks_on_peg(0))  # Sort normally to put smallest disk first
        peg_1_disks = sorted(self.disks_on_peg(1))
        peg_2_disks = sorted(self.disks_on_peg(2))
        
        # Maximum height needed (number of disks)
        max_height = self.num_disks
        
        # Width of the largest disk (2 * num_disks for visual scaling)
        max_width = 2 * self.num_disks
        
        # Initialize the tower representation
        tower_display = []
        
        # Build the tower representation from top to bottom
        for height in range(max_height - 1, -1, -1):  # Iterate in reverse to build from top down
            row = []
            # For each peg
            for peg_disks in [peg_0_disks, peg_1_disks, peg_2_disks]:
                # If there's a disk at this height from the bottom
                bottom_height = len(peg_disks) - 1 - height
                if bottom_height >= 0:
                    disk_num = peg_disks[bottom_height]
                    disk_width = 2 * (disk_num + 1)
                    disk = ('=' * disk_width).center(max_width)
                else:
                    disk = '||'.center(max_width)
                row.append(disk)
            tower_display.append('  '.join(row))
        
        # Add the base
        base = ('=' * max_width + '  ') * 3
        tower_display.append(base)
        
        # Add peg labels
        labels = 'Peg 0'.center(max_width) + '  ' + 'Peg 1'.center(max_width) + '  ' + 'Peg 2'.center(max_width)
        tower_display.append(labels)
        
        # Join all lines and print
        display = '\n' + '\n'.join(tower_display) + '\n'
        print(display)
        
        return display

    def set_env_parameters(self, num_disks=4, env_noise=0, verbose=True):
        self.num_disks = num_disks
        self.env_noise = env_noise
        self.observation_space = spaces.Tuple(self.num_disks*(spaces.Discrete(3),))
        self.goal_state = self.num_disks*(2,)

        if verbose:
            print("Hanoi Environment Parameters have been set to:")
            print("\t Number of Disks: {}".format(self.num_disks))
            print("\t Transition Failure Probability: {}".format(self.env_noise))

    def get_movability_map(self, fill=False):
        # Initialize movability map
        mov_map = np.zeros(self.num_disks*(3, ) + (6,))

        if fill:
            # Get list of all states as tuples
            id_list = self.num_disks*[0] + self.num_disks*[1] + self.num_disks*[2]
            states = list(itertools.permutations(id_list, self.num_disks))

            for state in states:
                for action in range(6):
                    move = action_to_move[action]
                    disks_from = []
                    disks_to = []
                    for d in range(self.num_disks):
                        if state[d] == move[0]: disks_from.append(d)
                        elif state[d] == move[1]: disks_to.append(d)

                    if disks_from: valid = (min(disks_to) > min(disks_from)) if disks_to else True
                    else: valid = False

                    if not valid: mov_map[state][action] = -np.inf

                    move_from = [m[0] for m in action_to_move]
                    move_to = [m[1] for m in action_to_move]

        return mov_map


action_to_move = [(0, 1), (0, 2), (1, 0),
                  (1, 2), (2, 0), (2, 1)]

def main():
    """Test the Hanoi environment with various scenarios."""
    print("=" * 50)
    print("Testing Hanoi Environment")
    print("=" * 50)
    
    # Create environment
    env = HanoiEnv()
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Goal state: {env.goal_state}")
    print("\nAction lookup:")
    for action, description in env.ACTION_LOOKUP.items():
        print(f"  {action}: {description}")
    
    print("\n" + "-" * 50)
    print("Test 1: Basic environment reset and initial state")
    print("-" * 50)
    
    # Reset environment
    initial_state = env.reset()
    print(f"Initial state after reset: {initial_state}")
    print(f"Done flag: {env.done}")
    
    print("\n" + "-" * 50)
    print("Test 2: Valid moves sequence")
    print("-" * 50)
    
    # Test some valid moves - move smallest disk around
    valid_actions = [1, 4, 1]  # Move disk 0: peg0→peg2, peg2→peg0, peg0→peg2
    
    for i, action in enumerate(valid_actions):
        print(f"\nStep {i+1}: Taking action {action} - {env.ACTION_LOOKUP[action]}")
        print(f"Current state before action: {env.current_state}")
        
        state, reward, done, info = env.step(action)
        
        print(f"New state: {state}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}")
        
        if done:
            print("Goal reached!")
            break
    
    print("\n" + "-" * 50)
    print("Test 3: Invalid move handling")
    print("-" * 50)
    
    # Reset for invalid move test
    env.reset()
    print(f"Reset state: {env.current_state}")
    
    # Try an invalid action (trying to move from empty peg)
    invalid_action = 2  # Move from peg 1 to peg 0, but peg 1 is empty
    print(f"\nTrying invalid action {invalid_action} - {env.ACTION_LOOKUP[invalid_action]}")
    print(f"State before invalid action: {env.current_state}")
    
    state, reward, done, info = env.step(invalid_action)
    print(f"State after invalid action: {state}")
    print(f"Reward: {reward}")
    print(f"Info: {info}")
    
    print("\n" + "-" * 50)
    print("Test 4: Environment with noise")
    print("-" * 50)
    
    # Test with noise
    env.set_env_parameters(num_disks=3, env_noise=0.3)
    env.reset()
    print(f"Environment reset with 3 disks and 30% noise")
    print(f"Initial state: {env.current_state}")
    print(f"Goal state: {env.goal_state}")
    
    # Take a few actions with noise
    for i in range(5):
        action = 1  # Always try the same action
        print(f"\nStep {i+1}: Attempting action {action}")
        print(f"State before: {env.current_state}")
        
        state, reward, done, info = env.step(action)
        
        print(f"State after: {state}")
        print(f"Reward: {reward}")
        print(f"Info: {info}")
        
        if done:
            break
    
    print("\n" + "-" * 50)
    print("Test 5: Optimal solution for 3 disks")
    print("-" * 50)
    
    # Reset to 3 disks with no noise for optimal solution
    env.set_env_parameters(num_disks=3, env_noise=0)
    env.reset()
    print(f"Solving 3-disk Hanoi puzzle optimally")
    print(f"Initial state: {env.current_state}")
    print(f"Goal state: {env.goal_state}")
    
    # Optimal solution for 3 disks: move all disks from peg 0 to peg 2
    optimal_sequence = [1, 0, 5, 1, 2, 3, 1]  # 7 moves for 3 disks
    
    total_reward = 0
    for i, action in enumerate(optimal_sequence):
        print(f"\nMove {i+1}: Action {action} - {env.ACTION_LOOKUP[action]}")
        print(f"State: {env.current_state}")
        
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"New state: {state}")
        print(f"Reward: {reward}")
        
        if done:
            print(f"Puzzle solved in {i+1} moves!")
            print(f"Total reward: {total_reward}")
            break
        
        if info["invalid_action"]:
            print("Invalid action detected!")
            break
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
