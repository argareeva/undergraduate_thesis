import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import trange


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, act_dim), nn.Softmax(dim=-1)
        )

    def forward(self, obs):
        return self.net(obs)

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q(x)

class ReplayBuffer:
    def __init__(self, size=100000):
        self.buffer = []
        self.max_size = size

    def add(self, transition):
        self.buffer.append(transition)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        if buffer_size < batch_size:
            indices = np.random.choice(buffer_size, buffer_size, replace=False)
        else:
            indices = np.random.choice(buffer_size, batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return map(np.array, zip(*batch))

def train_td3(env, transition_times, reward_map, episodes=500, batch_size=64, gamma=0.99):
    obs_dim = 1 + len(env.state_to_idx)
    act_dim = len(env.state_to_idx)

    actor = Actor(obs_dim, act_dim)
    critic1 = Critic(obs_dim, act_dim)
    critic2 = Critic(obs_dim, act_dim)
    target_actor = Actor(obs_dim, act_dim)
    target_critic1 = Critic(obs_dim, act_dim)
    target_critic2 = Critic(obs_dim, act_dim)

    target_actor.load_state_dict(actor.state_dict())
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic1_opt = torch.optim.Adam(critic1.parameters(), lr=1e-3)
    critic2_opt = torch.optim.Adam(critic2.parameters(), lr=1e-3)
    buffer = ReplayBuffer()

    for episode in trange(episodes):
        obs, _ = env.reset()
        obs_vec = np.concatenate([[obs['state']], obs['action_mask']])
        total_reward = 0
        for _ in range(50):
            obs_tensor = torch.tensor([obs_vec], dtype=torch.float32)
            with torch.no_grad():
                probs = actor(obs_tensor).numpy()[0] * obs['action_mask']
                probs /= probs.sum() if probs.sum() > 0 else 1
                action = np.random.choice(len(probs), p=probs)

            next_obs, _, terminated, truncated, _ = env.step(action)
            next_vec = np.concatenate([[next_obs['state']], next_obs['action_mask']])
            next_state_name = env.idx_to_state[next_obs['state']]
            reward = reward_map.get(next_state_name, 0)

            buffer.add((obs_vec, action, reward, next_vec, float(terminated or truncated)))
            obs_vec = next_vec
            obs = next_obs
            total_reward += reward

            if terminated or truncated:
                break

            if len(buffer.buffer) >= batch_size:
                o, a, r, o2, d = buffer.sample(batch_size)
                o = torch.tensor(o, dtype=torch.float32)
                a = torch.tensor(a, dtype=torch.long).unsqueeze(1)
                r = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
                o2 = torch.tensor(o2, dtype=torch.float32)
                d = torch.tensor(d, dtype=torch.float32).unsqueeze(1)

                with torch.no_grad():
                    a2 = target_actor(o2)
                    a2 = F.one_hot(a2.argmax(dim=-1), num_classes=act_dim).float()
                    q1_target = target_critic1(o2, a2)
                    q2_target = target_critic2(o2, a2)
                    q_target = r + gamma * (1 - d) * torch.min(q1_target, q2_target)

                a_onehot = F.one_hot(a.squeeze(1), num_classes=act_dim).float()
                q1 = critic1(o, a_onehot)
                q2 = critic2(o, a_onehot)
                loss1 = F.mse_loss(q1, q_target)
                loss2 = F.mse_loss(q2, q_target)

                critic1_opt.zero_grad(); loss1.backward(); critic1_opt.step()
                critic2_opt.zero_grad(); loss2.backward(); critic2_opt.step()

                pred_act = actor(o)
                pred_onehot = F.one_hot(pred_act.argmax(dim=-1), num_classes=act_dim).float()
                loss_actor = -critic1(o, pred_onehot).mean()

                actor_opt.zero_grad(); loss_actor.backward(); actor_opt.step()

                for t, s in zip(target_actor.parameters(), actor.parameters()):
                    t.data.copy_(0.995 * t.data + 0.005 * s.data)
                for t, s in zip(target_critic1.parameters(), critic1.parameters()):
                    t.data.copy_(0.995 * t.data + 0.005 * s.data)
                for t, s in zip(target_critic2.parameters(), critic2.parameters()):
                    t.data.copy_(0.995 * t.data + 0.005 * s.data)

    return actor

def simulate_td3(actor, env, transition_times, n_episodes=5):
    results = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        obs_vec = np.concatenate([[obs['state']], obs['action_mask']])
        path = [env.idx_to_state[obs['state']]]
        total_reward = 0
        for _ in range(50):
            obs_tensor = torch.tensor([obs_vec], dtype=torch.float32)
            with torch.no_grad():
                probs = actor(obs_tensor).numpy()[0] * obs['action_mask']
                probs /= probs.sum() if probs.sum() > 0 else 1
                action = np.random.choice(len(probs), p=probs)
            obs, reward, terminated, truncated, _ = env.step(action)
            path.append(env.idx_to_state[obs['state']])
            obs_vec = np.concatenate([[obs['state']], obs['action_mask']])
            total_reward += reward
            if terminated or truncated:
                break
        results.append((path, round(total_reward, 2)))
    return results