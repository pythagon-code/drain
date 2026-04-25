import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import sys
import torch
from gymnasium.wrappers import RecordVideo, TimeLimit
from torch import nn, optim
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from tqdm import trange
from rdqn import RDQN
from replay_buffer import ReplayBuffer

env = gym.make("LunarLander-v3")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
device = "cuda" if torch.cuda.is_available() else "cpu"

num_episodes = 1000000
batch_size = 64
replay_capacity = 10000
gamma = 0.99
lr = 3e-4
tau = 0.005
update_every_steps = 5
max_grad_norm = 0.5
epsilon = 0.4
epsilon_decay = 0.9998
epsilon_min = 0.01
hidden_noise_std = 0.1
num_rnn_layers = 4
hidden_dim = 512
trunc_length = 16

policy = RDQN(state_dim=state_dim, action_dim=action_dim, num_rnn_layers=num_rnn_layers, hidden_dim=512).to(device)
target = RDQN(state_dim=state_dim, action_dim=action_dim, num_rnn_layers=num_rnn_layers, hidden_dim=512).to(device)
target.load_state_dict(policy.state_dict())
for param in target.parameters():
    param.requires_grad_(False)
optimizer = optim.Adam(policy.parameters(), lr=lr)

hidden_size = num_rnn_layers * hidden_dim
replay_buffer = ReplayBuffer(
    capacity=replay_capacity,
    trunc_length=trunc_length,
    item_size=state_dim + 1 + 1 + state_dim + 1 + hidden_size*2 + hidden_size*2 + 1,
    batch_size=batch_size,
    dtype=torch.float32,
    device=device,
)

def polyak_update(target: nn.Module, policy: nn.Module, tau: float) -> None:
    for target_param, policy_param in zip(target.parameters(), policy.parameters()):
        target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

curr_integrity = replay_capacity
integrity_floor = 0
iter_count = 0
update_count = 0
episode_rewards = []
hidden = tuple(torch.zeros(1, num_rnn_layers, hidden_dim, dtype=torch.float32, device=device) for _ in range(2))
target_hidden = tuple(torch.zeros(1, num_rnn_layers, hidden_dim, dtype=torch.float32, device=device) for _ in range(2))
for episode in trange(num_episodes):
    done = False
    episode_reward = 0
    state, _ = env.reset()

    while not done:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=device).view(1, 1, -1)
            qs, next_hidden = policy.simple_forward(state, hidden)
            _, next_target_hidden = target.simple_forward(state, target_hidden)
            if torch.rand(()) < epsilon:
                action = torch.randint(action_dim, ())
            else:
                action = qs.argmax()
            action = action.cpu()

        next_state, reward, terminated, truncated, _ = env.step(action.numpy())
        episode_reward += reward
        done = terminated or truncated
        replay_buffer.add(torch.cat([
            state.cpu().flatten(),
            action.flatten().to(torch.float32),
            torch.tensor([reward], dtype=torch.float32),
            torch.tensor(next_state, dtype=torch.float32),
            torch.tensor([done], dtype=torch.float32),
            hidden[0].cpu().flatten(),
            hidden[1].cpu().flatten(),
            next_target_hidden[0].cpu().flatten(),
            next_target_hidden[1].cpu().flatten(),
            torch.tensor([curr_integrity], dtype=torch.float32),
        ]))

        if iter_count % update_every_steps == 0 and replay_buffer.get_size() >= 1000:
            batch, idxs = replay_buffer.sample()
            batch = batch.to(device)
            start_idx = 0
            states = batch[:, :, start_idx:start_idx + state_dim].reshape(-1, trunc_length, state_dim)
            start_idx += state_dim
            actions = batch[:, :, start_idx:start_idx + 1].reshape(-1, trunc_length, 1).to(torch.int32)
            start_idx += 1
            rewards = batch[:, :, start_idx:start_idx + 1].reshape(-1, trunc_length, 1)
            start_idx += 1
            next_states = batch[:, :, start_idx:start_idx + state_dim].reshape(-1, trunc_length, state_dim)
            start_idx += state_dim
            dones = batch[:, :, start_idx:start_idx + 1].reshape(-1, trunc_length, 1)
            start_idx += 1
            hs = batch[:, :, start_idx:start_idx + hidden_size].reshape(
                -1,
                trunc_length,
                num_rnn_layers,
                hidden_dim,
            )
            start_idx += hidden_size
            cs = batch[:, :, start_idx:start_idx + hidden_size].reshape(
                -1,
                trunc_length,
                num_rnn_layers,
                hidden_dim,
            )
            start_idx += hidden_size
            next_hs = batch[:, :, start_idx:start_idx + hidden_size].reshape(
                -1,
                trunc_length,
                num_rnn_layers,
                hidden_dim,
            )
            start_idx += hidden_size
            next_cs = batch[:, :, start_idx:start_idx + hidden_size].reshape(
                -1,
                trunc_length,
                num_rnn_layers,
                hidden_dim,
            )
            start_idx += hidden_size
            integrities = batch[:, :, start_idx:start_idx + 1].reshape(-1, trunc_length, 1)

            with torch.no_grad():
                next_qs, (re_next_hs, re_next_cs), re_integrities = target(
                    next_states,
                    (
                        next_hs[:, 0],
                        next_cs[:, 0],
                    ),
                    integrities[:, 0],
                )
                next_best_qs = next_qs.max(dim=2, keepdim=True)[0]
                target_qs = rewards + gamma * (1 - dones) * next_best_qs
            qs, (re_hs, re_cs), _ = policy(
                states,
                (
                    hs[:, 0],
                    cs[:, 0],
                ),
                None,
            )
            interesting_qs = torch.gather(qs, dim=2, index=actions)
            mean_integrity = integrities[:, 0].mean()
            score = integrities[:, 0] - integrity_floor
            weight = score / score.mean(dim=0, keepdim=True)
            sequence_mse = ((interesting_qs - target_qs)**2).mean(dim=1)
            loss = (sequence_mse * weight).mean()
            noisy_loss_diff = torch.tensor(0.0, device=device)
            if update_count % 100 == 0:
                with torch.no_grad():
                    noisy_h0 = hs[:, 0] + hidden_noise_std * torch.randn_like(hs[:, 0])
                    noisy_c0 = cs[:, 0] + hidden_noise_std * torch.randn_like(cs[:, 0])
                    noisy_next_h0 = next_hs[:, 0] + hidden_noise_std * torch.randn_like(next_hs[:, 0])
                    noisy_next_c0 = next_cs[:, 0] + hidden_noise_std * torch.randn_like(next_cs[:, 0])
                    noisy_next_qs, _, _ = target(
                        next_states,
                        (
                            noisy_next_h0,
                            noisy_next_c0,
                        ),
                        integrities[:, 0],
                    )
                    noisy_next_best_qs = noisy_next_qs.max(dim=2, keepdim=True)[0]
                    noisy_target_qs = rewards + gamma * (1 - dones) * noisy_next_best_qs
                    noisy_qs, _, _ = policy(
                        states,
                        (
                            noisy_h0,
                            noisy_c0,
                        ),
                        None,
                    )
                    noisy_interesting_qs = torch.gather(noisy_qs, dim=2, index=actions)
                    noisy_sequence_mse = ((noisy_interesting_qs - noisy_target_qs)**2).mean(dim=1)
                    noisy_loss = (noisy_sequence_mse * weight).mean()
                    noisy_loss_diff = noisy_loss - loss.detach()
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            polyak_update(target, policy, tau)

            with torch.no_grad():
                replay_buffer.realign_hidden_states(
                    idxs = idxs,
                    hs = torch.cat([hs[:, 0:1], re_hs[:, :-1]], dim=1),
                    cs = torch.cat([cs[:, 0:1], re_cs[:, :-1]], dim=1),
                    next_hs = re_hs,
                    next_cs = re_cs,
                    integrities = re_integrities,
                )

            if update_count % 100 == 0:
                print(f"\n{update_count}, l: {loss.item()}, nld: {noisy_loss_diff.item()}, eps: {epsilon:.4f}, i: {mean_integrity - integrity_floor:.4f}, er: {sum(episode_rewards[-20:])/20 if len(episode_rewards) > 20 else episode_rewards[-1] if episode_rewards else 0:.0f}")
            update_count += 1
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
        state = next_state
        hidden = next_hidden
        target_hidden = next_target_hidden
        curr_integrity += 1
        integrity_floor += 1
        iter_count += 1
    
    episode_rewards.append(episode_reward)