import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class MaskedProcessEnv(gym.Env):
    def __init__(self, transitions_df, start_state: str, terminal_states, reward_map=None, max_steps=50):
        super(MaskedProcessEnv, self).__init__()

        self.start_state = start_state
        self.transitions_df = transitions_df
        self.reward_map = reward_map or {}
        self.max_steps = max_steps
        self.current_step = 0

        self.states = sorted(set(transitions_df['from']).union(set(transitions_df['to'])))
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state = {i: s for s, i in self.state_to_idx.items()}

        self.observation_space = spaces.Dict({
            "state": spaces.Discrete(len(self.states)),
            "action_mask": spaces.MultiBinary(len(self.states))
        })
        self.action_space = spaces.Discrete(len(self.states))

        self.transition_dict = transitions_df.groupby("from")["to"].apply(list).to_dict()
        self.transition_times = transitions_df.groupby(["from", "to"])["time"].mean().to_dict()
        self.terminal_states = terminal_states

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_state = self.start_state
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        mask = np.zeros(len(self.states), dtype=np.int8)
        valid_actions = self.transition_dict.get(self.current_state, [])
        for to_state in valid_actions:
            mask[self.state_to_idx[to_state]] = 1
        return {"state": self.state_to_idx[self.current_state], "action_mask": mask}

    def step(self, action_idx):
        self.current_step += 1
        next_state = self.idx_to_state[action_idx]

        valid = next_state in self.transition_dict.get(self.current_state, [])

        if not valid:
            reward = -1000
            terminated = True
            truncated = False
            return self._get_obs(), reward, terminated, truncated, {}

        base_reward = self.reward_map.get(next_state, 0)
        terminated = next_state in self.terminal_states
        truncated = self.current_step >= self.max_steps

        self.current_state = next_state
        return self._get_obs(), base_reward, terminated, truncated, {}

    def render(self):
        print(f"Current state: {self.current_state}")