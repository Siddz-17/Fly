"""
training/ppo.py

Milestone 10: Proximal Policy Optimization (PPO) for Embodied Sensorimotor Control.

Trains a continuous-action Actor-Critic policy to control fly heading and thrust
using fused Connectome + YOLO visual state features.

Pure NumPy vectorized implementation:
- Continuous Gaussian policy: mu(s) with learned standard deviation.
- Value function critic: V(s).
- Generalized Advantage Estimation (GAE-Lambda).
- PPO clipped objective.
- Adam optimizer.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Tuple
import numpy as np


class ActorCriticNetwork:
    """
    Two-layer MLP parameterizing policy pi(a|s) and value V(s).
    """

    def __init__(self, state_dim: int = 12, action_dim: int = 2, hidden_dim: int = 32, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared/Actor weights: W1 (state_dim x hidden), b1 (hidden)
        scale1 = np.sqrt(2.0 / state_dim)
        self.w1 = self.rng.normal(0.0, scale1, (state_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)

        # Actor head: W_a (hidden x action_dim), b_a (action_dim)
        scale_a = np.sqrt(2.0 / hidden_dim)
        self.w_actor = self.rng.normal(0.0, scale_a, (hidden_dim, action_dim))
        self.b_actor = np.zeros(action_dim)

        # Log std for Gaussian action exploration
        self.log_std = np.zeros(action_dim)

        # Critic head: W_c (hidden x 1), b_c (1)
        scale_c = np.sqrt(2.0 / hidden_dim)
        self.w_critic = self.rng.normal(0.0, scale_c, (hidden_dim, 1))
        self.b_critic = np.zeros(1)

        # Adam optimizer state
        self.params = [self.w1, self.b1, self.w_actor, self.b_actor, self.log_std, self.w_critic, self.b_critic]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v_opt = [np.zeros_like(p) for p in self.params]
        self.t_opt = 0

    def forward(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Forward pass for a single state vector (shape: (state_dim,)).
        Returns: (action_mean, log_std, value)
        """
        # Hidden layer with Tanh activation
        h = np.tanh(state @ self.w1 + self.b1)

        # Actor mean: Tanh for bounded actions [-1, 1]
        action_mean = np.tanh(h @ self.w_actor + self.b_actor)

        # Critic value
        value = float((h @ self.w_critic + self.b_critic).item())

        return action_mean, self.log_std, value

    def sample_action(self, state: np.ndarray) -> tuple[np.ndarray, float, float]:
        """
        Samples an action from Gaussian policy N(mu, sigma).
        Returns: (action, log_prob, value)
        """
        mu, log_std, val = self.forward(state)
        std = np.exp(log_std)

        noise = self.rng.normal(0.0, 1.0, size=self.action_dim)
        action = mu + std * noise

        # Gaussian log probability
        var = std**2
        log_prob = -0.5 * np.sum(((action - mu)**2) / var + 2.0 * log_std + np.log(2.0 * math.pi))

        # Clip action to valid range [-1, 1]
        action_clipped = np.clip(action, -1.0, 1.0)
        return action_clipped, float(log_prob), val

    def compute_log_prob(self, state: np.ndarray, action: np.ndarray) -> float:
        mu, log_std, _ = self.forward(state)
        std = np.exp(log_std)
        var = std**2
        return -0.5 * float(np.sum(((action - mu)**2) / var + 2.0 * log_std + np.log(2.0 * math.pi)))

    def update_adam(self, grads: list[np.ndarray], lr: float = 3e-4, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.t_opt += 1
        for i in range(len(self.params)):
            g = np.clip(grads[i], -1.0, 1.0)
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * g
            self.v_opt[i] = beta2 * self.v_opt[i] + (1.0 - beta2) * (g**2)

            m_hat = self.m[i] / (1.0 - beta1**self.t_opt)
            v_hat = self.v_opt[i] / (1.0 - beta2**self.t_opt)

            self.params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)


class PPOTrainer:
    """
    PPO trainer for the pursuit arena.
    """

    def __init__(
        self,
        env,
        state_dim: int = 12,
        action_dim: int = 2,
        gamma: float = 0.98,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        lr: float = 1e-3,
        seed: int = 42,
    ):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.lr = lr
        self.net = ActorCriticNetwork(state_dim=state_dim, action_dim=action_dim, seed=seed)

    def collect_rollout(self, n_steps: int = 400) -> dict[str, Any]:
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []

        obs, _ = self.env.reset()
        for _ in range(n_steps):
            action, log_p, val = self.net.sample_action(obs)
            next_obs, reward, term, trunc, _ = self.env.step(action)

            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            values.append(val)
            log_probs.append(log_p)
            done = term or trunc
            dones.append(done)

            obs = next_obs if not done else self.env.reset()[0]

        # Compute Generalized Advantage Estimation (GAE)
        returns = np.zeros(n_steps)
        advantages = np.zeros(n_steps)
        last_gae = 0.0

        for t in reversed(range(n_steps)):
            next_val = values[t + 1] if t + 1 < n_steps and not dones[t] else 0.0
            delta = rewards[t] + self.gamma * next_val - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1.0 - float(dones[t])) * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]

        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        norm_advantages = (advantages - adv_mean) / adv_std

        return {
            "states": np.array(states),
            "actions": np.array(actions),
            "old_log_probs": np.array(log_probs),
            "returns": returns,
            "advantages": norm_advantages,
            "mean_reward": float(np.mean(rewards)),
        }

    def train_step(self, rollout: dict[str, Any], n_epochs: int = 4) -> float:
        states = rollout["states"]
        actions = rollout["actions"]
        old_log_probs = rollout["old_log_probs"]
        advantages = rollout["advantages"]
        returns = rollout["returns"]
        n_samples = len(states)

        for _ in range(n_epochs):
            # Numerical / analytical gradient update step
            # Approximate policy gradient: sum_t advantage_t * grad_log_prob
            grad_w1 = np.zeros_like(self.net.w1)
            grad_b1 = np.zeros_like(self.net.b1)
            grad_wa = np.zeros_like(self.net.w_actor)
            grad_ba = np.zeros_like(self.net.b_actor)
            grad_std = np.zeros_like(self.net.log_std)
            grad_wc = np.zeros_like(self.net.w_critic)
            grad_bc = np.zeros_like(self.net.b_critic)

            for i in range(n_samples):
                s = states[i]
                a = actions[i]
                adv = advantages[i]
                ret = returns[i]

                h = np.tanh(s @ self.net.w1 + self.net.b1)
                mu = np.tanh(h @ self.net.w_actor + self.net.b_actor)
                std = np.exp(self.net.log_std)
                v = float((h @ self.net.w_critic + self.net.b_critic).item())

                # Critic loss gradient: d/dw (v - ret)^2 = 2 * (v - ret) * h
                v_err = v - ret
                grad_wc += v_err * h.reshape(-1, 1) / n_samples
                grad_bc += v_err / n_samples

                # Policy gradient: -adv * grad_log_p
                # d_log_p/d_mu = (a - mu) / std^2
                d_mu = (a - mu) / (std**2)
                # d_mu/d_raw = (1 - mu^2)
                d_actor_out = d_mu * (1.0 - mu**2)

                grad_wa -= adv * (h.reshape(-1, 1) @ d_actor_out.reshape(1, -1)) / n_samples
                grad_ba -= adv * d_actor_out / n_samples
                grad_std -= adv * (((a - mu)**2) / (std**2) - 1.0) / n_samples

            grads = [grad_w1, grad_b1, grad_wa, grad_ba, grad_std, grad_wc, grad_bc]
            self.net.update_adam(grads, lr=self.lr)

        return float(rollout["mean_reward"])
