import math
import time
from tqdm import tqdm
from typing import Literal, Optional
import time
import matplotlib.pyplot as plt

import numpy as np
import torch.nn as nn
import torch
from torch import Tensor, optim

from .base_policy import Policy
from .base_env import Env
from .rollout_buffer import RolloutBuffer
from src.utils import *
    
class PPO:
    def __init__(self, policy: Policy, env: Env, 
                 clip_epsilon=0.2, 
                 gamma=1, 
                 gae_lambda=0.95, 
                 normalize_advantages=True, 
                 entropy_schedule=lambda t: 0.2,
                 n_epochs=5, 
                 batch_size=10, 
                 buffer_size=20,
                 device='cuda', 
                 log_interval=10, 
                 shuffle=False,
                 log_callback=None,
                 plots_path=None,
                 
                 **kwargs
                 ):
        """
        Initialize the PPO algorithm with hyperparameters and required objects.

        Parameters:
        - clip_epsilon: Controls how much the policy can change between updates (clipped ratio).
        - gamma: Discount factor for future rewards (1 means full future reward impact).
        - gae_lambda: Trade-off between bias and variance for GAE (lambda-return).
        - normalize_advantages: Normalize advantages for better numerical stability.
        - n_epochs: Number of passes over the data to update the policy.
        - batch_size: Number of samples per batch for policy updates.
        - buffer_size: Number of steps to store in the rollout buffer.
        - log_interval: Interval (in iterations) for printing summary logs.
        """
        self.policy = policy.to(device)
        self.env = env
        assert env.device == device or device == 'cuda', f"Policy device ({device}) and environment device ({env.device}) must match."

        # PPO Hyperparameters
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma                              # Discount factor for future rewards.
        self.gae_lambda = gae_lambda                    # Governs the advantage estimation trade-off.
        self.normalize_advantages = normalize_advantages
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        self.shuffle = shuffle
        self.entropy_schedule = entropy_schedule

        # Initialize the rollout buffer to store experiences
        self.rollout_buffer = RolloutBuffer(
            buffer_size=buffer_size,
            observation_shape=env.observation_shape,
            action_shape=env.action_shape,
            gae_lambda=gae_lambda,
            gamma=gamma,
            device=device
        )

        # Logger to track losses and training progress
        self.logger = {
            "t_so_far": 0,  
            "i_so_far": 0, 
            "actor_losses": [],
            "critic_losses": [],
            "avg_rewards": [],
            "avg_critic_values": [],
            "avg_advantages": [],
            "entropy": [],
            "entropy_coeff": [],
            "surr_loss": [],
            "timesteps": [],
            "num_match": []
        }
        self.log_interval = log_interval
        self.log_callback = log_callback
        self.plots_path = plots_path

    def collect_rollout(self):
        """
        Collect a batch of experiences by interacting with the environment.
        This data is used to update the policy and value functions.
        """
        self.rollout_buffer.reset()  # Clear buffer at the start of collection
        obs = self.env.reset()  # Get the initial observation from the environment

        while not self.rollout_buffer.is_full():
            with torch.no_grad():  # No gradients required during environment interaction
                action, log_prob, value = self.policy.select_action(obs)

            next_obs, reward, done = self.env.step(action)

            # print(f"action: {action}, log_prob: {log_prob}, value: {value}, reward: {reward}, done: {done}")
            self.rollout_buffer.add(
                obs, action, reward, log_prob, value, done
            )

            obs = next_obs if not done else self.env.reset()

        # Compute value estimate for the last state for bootstrapping
        with torch.no_grad():
            values = self.policy.predict_values(obs)

            # Compute returns and advantages for the collected rollout
            self.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=done)

    def compute_loss(self, batch_data):
        """
        Compute the PPO loss for both the actor and critic.

        Parameters:
        - batch_data: A batch of experiences containing states, actions, log_probs, returns, etc.

        Returns:
        - actor_loss: The loss for the actor (policy network).
        - critic_loss: The loss for the critic (value network).
        - entropy: The entropy of the policy for exploration monitoring.
        """
        obs = batch_data['obs']
        actions = batch_data['actions']
        old_log_probs = batch_data['log_probs']
        returns = batch_data['returns']
        advantages = batch_data['advantages']
        rewards = batch_data['rewards']
        
        # print("=" * 100)
        # Normalize advantages if required (for better numerical stability)
        if self.normalize_advantages and len(advantages) != 1:
            # print(f"adv mean: {advantages.mean()}, adv std: {advantages.std()}")
            if advantages.std() > 1e-7:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
            else:
                # If std is too small, just center?
                advantages = advantages - advantages.mean()

        # Evaluate the policy with the current obs and actions
        values_new, log_probs_new, entropy = self.policy.evaluate_actions(obs, actions)
        
        # Calculate the ratio of new and old action probabilities
        ratios = torch.exp(log_probs_new - old_log_probs)
        # print(f"ratios: {ratios}")
        # print(f"advantages: {advantages}")

        # Compute the surrogate objectives (clipped vs unclipped)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        # print(f"ratios: {ratios}")
        # print(f"log_probs_new: {log_probs_new}")
        # print(f"old_log_probs: {old_log_probs}")
        # actor_loss = -(log_probs_new * rewards).mean()
        # Actor loss: Minimize the worst-case surrogate
        actor_loss = -torch.min(surr1, surr2).mean()
        self.logger["surr_loss"].append(actor_loss.item())
        
        # print(f"actor loss before: {actor_loss}")
        # print(f"surr1: {surr1}, surr2: {surr2}")
        # print(f"ratios: {ratios}")
        entropy_loss = self.entropy_schedule(self.logger["i_so_far"]) * entropy.mean()
        actor_loss -= entropy_loss
        
        # print(f"actions: {actions}")
        # print(f"returns: {returns}")
        # print(f"rewards: {rewards}")
        # print("=" * 100)

        # print(f"actor_loss: {actor_loss}")
        # print(f"values_new: {values_new}")
        # Critic loss: Minimize MSE between predicted and actual returns
        # print(f"values_new: {values_new}, returns: {returns}")
        critic_loss = nn.MSELoss()(values_new, returns)

        return actor_loss, critic_loss, entropy.mean()

    def update(self):
        """
        Update the policy and value networks using collected rollouts.
        """
        
        for _ in range(self.n_epochs):  # Update multiple times per rollout
            for batch_data in self.rollout_buffer.get(self.batch_size, shuffle=self.shuffle):
                actor_loss, critic_loss, entropy = self.compute_loss(batch_data)

                # Update policy and value networks
                self.policy.optimizer_step(actor_loss, critic_loss)

                self.logger["actor_losses"].append(actor_loss.item())
                self.logger["critic_losses"].append(critic_loss.item())
                self.logger["entropy"].append(entropy.item())
        
        self.logger["entropy_coeff"].append(self.entropy_schedule(self.logger["i_so_far"]))

        self.logger["avg_rewards"].append(self.rollout_buffer.rewards.mean().item())
        self.logger["avg_advantages"].append(self.rollout_buffer.advantages.mean().item())
        self.logger["avg_critic_values"].append(self.rollout_buffer.values.mean().item())
        
    def train(self, total_timesteps):
        """
        Main training loop to run the PPO algorithm for the specified timesteps.
        """
        t_so_far = 0  # Total timesteps seen so far
        i_so_far = 0  # Total iterations completed

        while t_so_far < total_timesteps:
            # Collect rollouts and populate the buffer
            self.collect_rollout()
            t_so_far += self.rollout_buffer.buffer_size
            i_so_far += 1

            self.logger["timesteps"].append(t_so_far)
            self.logger["i_so_far"] = i_so_far

            # Perform policy (actor+critic) updates
            self.update()

            if i_so_far % self.log_interval == 0:
                self._log_summary()

    def _log_summary(self):
        """
        Print a summary of the current training progress.
        """
        print(f"Update {len(self.logger['timesteps'])}:")
        print(f"  Timesteps so far: {self.logger['t_so_far']}")
        print(f"  Actor Loss: {self.logger['actor_losses'][-1]:.6f}")
        print(f"  Entropy: {self.logger['entropy'][-1]:.4f}")
        print(f"  Entropy Coefficient: {self.logger['entropy_coeff'][-1]:.4f}")
        print(f"  Surrogate Loss: {self.logger['surr_loss'][-1]:.4f}")
        print(f"  Critic Loss: {self.logger['critic_losses'][-1]:.4f}")
        print(f"  Avg Rewards: {self.logger['avg_rewards'][-1]:.4f}")
        print(f"  Avg Advantages: {self.logger['avg_advantages'][-1]:.4f}")
        print(f"  Avg Critic Values: {self.logger['avg_critic_values'][-1]:.4f}")

        if self.log_callback:
            num_match = self.log_callback(self.policy)
            self.logger["num_match"].append(num_match)
            print(f"  Num Match: {num_match}")
        print("=" * 50)    

    def save_policy(self, path="policy_checkpoint.pth"):
        checkpoint = {
            'actor_state_dict': self.policy.actor.state_dict(),
            'critic_state_dict': self.policy.critic.state_dict(),
            'actor_optimizer_state_dict': self.policy.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.policy.critic_optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"Policy saved to {path}")

    def load_policy(self, path="policy_checkpoint.pth", device='cuda'):
        checkpoint = torch.load(path, map_location=device)
        
        # Load actor and critic state dictionaries
        self.policy.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.policy.critic.load_state_dict(checkpoint['critic_state_dict'])

        # Load optimizers' state dictionaries
        self.policy.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.policy.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        print(f"Policy loaded from {path} onto {device}")
