import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, optim
import torch.distributions as dist
from abc import ABC, abstractmethod
from typing import Any

class Critic(nn.Module, ABC):
    """
    The state value function V
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, obs):
        """
        Output the critic's predicted values.

        Parameters:
        - obs (Tensor): The current obs 

        Returns:
        - value_estimate (Tensor): The value estimate of the obs.
        """
        pass

class Actor(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, obs: Tensor):
        """
        Forward through the Actor network
        
        Parameters:
        - obs (Tensor): The current obs
        
        Returns:
        - intermediate (Any): The result of the forward pass,
                parameterizing the actor's action distribution
        """
        pass

    @abstractmethod
    def select_action(self, obs: Tensor):
        """
        Runs the actor network to return the action and log_prob of the action.
        
        Parameters:
        - obs (Tensor): The current obs
        
        Returns:
        - action (Any): The action to be taken.
        - log_prob (Tensor): The log probability of the action.
        """
        pass

    @abstractmethod
    def evaluate_actions(self, obs: Tensor, action: Tensor):
        """
        Returns log probabilities of choosing the action given the obs
        and the entropy of the action distribution.

        Parameters:
        - None

        Returns:
        - log_probs: Tensor: the log probabilities of all actions.
        - entropy: Tensor: the entropy of the action distribution.
        """
        pass

class Policy(nn.Module):
    def __init__(self,
                actor: Actor,
                critic: Critic,
                actor_lr=3e-4,
                critic_lr=1e-3,
                device='cuda'
                ):
        super().__init__()
        print(f'initializing policy with device: {device}')
        self.actor = actor.to(device)
        self.critic = critic.to(device)

        # print(f"actor: {self.actor}")
        # print(f"critic: {self.critic}")
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    def select_action(self, obs):
        """
        Given an observation, use the actor to select an action and return
        the action, log probability, and the value from the critic.

        Parameters:
        - obs (Tensor): The observation returned by the environment.

        Returns:
        - action (Tensor): The action to be taken.
        """
        # Sample action from actor
        action, log_prob = self.actor.select_action(obs)  

        # Get value estimate from critic
        value = self.critic(obs)
        return action, log_prob, value

    def evaluate_actions(self, obs, actions):
        """
        Given observations and actions, return log probabilities of actions, entropy of the action distribution,
        and value estimates. This will be used by PPO to compute the loss.
        """
        # Evaluate actions from actor
        log_probs, entropy = self.actor.evaluate_actions(obs, actions)

        # Get value estimates from critic
        values = self.critic(obs)  
        return values, log_probs, entropy

    def optimizer_step(self, actor_loss, critic_loss):
        """
        Perform optimization steps for both the actor and critic.
        """
        # Actor update
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # print(f"Did optimizer step with actor loss: {actor_loss}")

        # # Critic update
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
    
    def set_env(self, train_mode: bool):
        """
        Set the env based on train or test mode.
        """
        self.actor.set_env(train_mode)
        self.critic.set_env(train_mode)

    def predict_values(self, obs):
        """
        Predict the value of the observation using the critic.
        """
        return self.critic(obs)