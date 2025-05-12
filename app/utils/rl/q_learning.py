import random
from collections import defaultdict
from typing import Dict


class QLearningSimulator:
    def __init__(self, transitions_df, reward_map: Dict[str, float], terminal_states, start_state: str):
        self.transitions = transitions_df
        self.reward_map = reward_map
        self.terminal_states = terminal_states
        self.start_state = start_state
        self.state_actions = defaultdict(list)
        self.rewards = defaultdict(dict)
        self._prepare_environment()

    def _prepare_environment(self):
        for _, row in self.transitions.iterrows():
            f, t, time = row["from"], row["to"], row["time"]
            reward = -time + self.reward_map.get(t, 0)
            self.state_actions[f].append(t)
            self.rewards[f][t] = reward

    def reset(self):
        self.current_state = self.start_state
        return self.current_state

    def step(self, action):
        if action not in self.state_actions.get(self.current_state, []):
            return self.current_state, -100, True
        reward = self.rewards[self.current_state][action]
        done = action in self.terminal_states
        self.current_state = action
        return action, reward, done

    def train_q_learning(self, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
        Q = defaultdict(lambda: defaultdict(float))
        for _ in range(episodes):
            state = self.reset()
            for _ in range(50):
                actions = self.state_actions.get(state, [])
                if not actions:
                    break
                action = random.choice(actions) if random.random() < epsilon else max(actions, key=lambda a: Q[state][a])
                next_state, reward, done = self.step(action)
                max_q = max([Q[next_state][a] for a in self.state_actions.get(next_state, [])], default=0)
                Q[state][action] += alpha * (reward + gamma * max_q - Q[state][action])
                state = next_state
                if done:
                    break
        return Q


def simulate_q_policy(Q, transitions_df, terminal_states, start_state: str, n_simulations=100):
    state_actions = defaultdict(list)
    rewards = defaultdict(dict)
    for _, row in transitions_df.iterrows():
        state_actions[row["from"]].append(row["to"])
        rewards[row["from"]][row["to"]] = -row["time"]

    paths = []
    for _ in range(n_simulations):
        state = start_state
        path = [state]
        total_reward = 0
        for _ in range(50):
            actions = state_actions.get(state, [])
            if not actions:
                break
            action = max(Q[state], key=Q[state].get) if Q[state] else random.choice(actions)
            reward = rewards[state][action]
            path.append(action)
            total_reward += reward
            if action in terminal_states:
                break
            state = action
        paths.append((path, -total_reward))
    return paths