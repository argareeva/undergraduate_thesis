import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import trange


class MaskedPPONetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.actor = nn.Linear(64, act_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, obs_tensor, mask_tensor):
        x = self.shared(obs_tensor)
        logits = self.actor(x)
        masked_logits = logits + (1 - mask_tensor) * -1e9
        action_probs = F.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        value = self.critic(x).squeeze(-1)
        return dist, value

class RolloutBuffer:
    def __init__(self):
        self.observations = []
        self.action_masks = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def store(self, obs, mask, action, reward, log_prob, value, done):
        self.observations.append(obs)
        self.action_masks.append(mask)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def get_tensors(self):
        return (
            torch.tensor(self.observations, dtype=torch.float32),
            torch.tensor(self.action_masks, dtype=torch.float32),
            torch.tensor(self.actions, dtype=torch.long),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.tensor(self.log_probs, dtype=torch.float32),
            torch.tensor(self.values, dtype=torch.float32),
            torch.tensor(self.dones, dtype=torch.float32)
        )

def compute_returns_and_advantages(rewards, values, dones, gamma=0.99, lam=0.95):
    returns = []
    advs = []
    gae = 0
    next_value = 0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        adv = gae
        returns.insert(0, adv + values[step])
        advs.insert(0, adv)
        next_value = values[step]
    return torch.tensor(returns), torch.tensor(advs)

def ppo_update(model, optimizer, buffer, clip_eps=0.2, epochs=5, batch_size=64):
    obs, masks, actions, rewards, old_log_probs, values, dones = buffer.get_tensors()
    returns, advantages = compute_returns_and_advantages(rewards, values, dones)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(epochs):
        for i in range(0, len(obs), batch_size):
            idx = slice(i, i + batch_size)
            dist, value = model(obs[idx], masks[idx])
            entropy = dist.entropy().mean()
            new_log_prob = dist.log_prob(actions[idx])
            ratio = (new_log_prob - old_log_probs[idx]).exp()
            surr1 = ratio * advantages[idx]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages[idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value, returns[idx])
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def train_custom_ppo(env, reward_map, total_timesteps=10000, rollout_len=2048):
    obs_dim = 1 + len(env.state_to_idx)
    act_dim = len(env.state_to_idx)
    model = MaskedPPONetwork(obs_dim, act_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    buffer = RolloutBuffer()

    obs, _ = env.reset()

    for _ in trange(total_timesteps // rollout_len):
        buffer.clear()
        for _ in range(rollout_len):
            obs_vec = np.concatenate([[obs['state']], obs['action_mask']])
            obs_tensor = torch.tensor([obs_vec], dtype=torch.float32)
            mask_tensor = torch.tensor([obs['action_mask']], dtype=torch.float32)
            dist, value = model(obs_tensor, mask_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            action = int(action.item())
            next_obs, _, terminated, truncated, _ = env.step(action)
            next_state_name = env.idx_to_state[next_obs['state']]

            reward = reward_map.get(next_state_name, 0)
            buffer.store(obs_vec, obs['action_mask'], action, reward, log_prob.item(), value.item(), float(terminated or truncated))

            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()

        ppo_update(model, optimizer, buffer)

    return model

def simulate_masked_ppo(model, env, n_episodes=5):
    idx_to_state = env.idx_to_state
    results = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        terminated, truncated = False, False
        path = [idx_to_state[obs["state"]]]
        total_reward = 0
        for _ in range(50):
            obs_vec = np.concatenate([[obs["state"]], obs["action_mask"]])
            obs_tensor = torch.tensor([obs_vec], dtype=torch.float32)
            mask_tensor = torch.tensor([obs["action_mask"]], dtype=torch.float32)
            dist, _ = model(obs_tensor.unsqueeze(0), mask_tensor.unsqueeze(0))
            action = int(dist.sample().item())
            obs, reward, terminated, truncated, _ = env.step(action)
            path.append(idx_to_state[obs["state"]])
            total_reward += reward
            if terminated or truncated:
                break
        results.append((path, round(total_reward, 2)))
    return results