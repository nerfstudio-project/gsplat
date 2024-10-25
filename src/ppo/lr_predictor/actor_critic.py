import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim
import torch.distributions as dist
from src.ppo.base_policy import Actor, Critic


class LRCritic(Critic):
    def __init__(self, env, input_dim: int = 1024, h_dim: int = 64):
        super().__init__()
        # should map obs (image) to value
        self.layers = [
            nn.Linear(input_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
        ]
        self.network = nn.Sequential(*self.layers)
        self.env = env

    def forward(self, obs: Tensor):
        # Ensure obs is properly reshaped for the network
        batch_size = obs.shape[0] if len(obs.shape) > 3 else 1
        # obs = obs.view(batch_size, -1)  # Flatten input to (batch_size, features)
        
        # return self.network(obs)

        # For debugging, simply return mean of psnr's over diff lr for this img
        return self.env.get_mean_reward()

class LRActor(Actor):
    def __init__(self, lrs: list[float] = None, input_dim: int = 1024, h_dim: int = 64, env=None):
        super().__init__()
        if lrs:
            self.lrs = lrs
        else:
            self.lrs = Tensor([10**-i for i in range(10)])
        
        print(f"Actor using learning rates: {self.lrs}")
        self.layers = [
            nn.Linear(input_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, len(self.lrs)),
            nn.Softmax(dim=-1)
        ]
        self.network = nn.Sequential(*self.layers)
        
    def forward(self, obs: Tensor):
        return self.network(obs)
    # def __init__(self, input_dim: int = 1024, h_dim: int = 64):
    #     super().__init__()
    #     self.logits = nn.Parameter(torch.zeros(10))
    #     self.lrs = Tensor([10**-i for i in range(10)])

    # def forward(self, obs: Tensor):
    #     return F.softmax(self.logits, dim=-1)

    def actor_distribution(self, obs: Tensor):
        # print(f"logits will be {self.forward(obs)[0]}")
        return dist.Categorical(self.forward(obs))
    
    def select_action(self, obs: Tensor):
        actor_dist = self.actor_distribution(obs)
        lr_idx = actor_dist.sample()
        return lr_idx, actor_dist.log_prob(lr_idx)

    def log_prob(self, obs, acts):
        actor_dist = self.actor_distribution(obs)
        return actor_dist.log_prob(acts)
    
    def evaluate_actions(self, obs: Tensor, action: Tensor):
        actor_dist = self.actor_distribution(obs)
        # print(f"Evaluating actions: obs shape: {obs.shape}, action shape: {action.shape}, log_prob shape: {actor_dist.log_prob(action.squeeze(-1)).shape}")        
        # print(f"action: {action}, dist logits: {actor_dist.logits[0]}, entropy: {entropy[0]}")
        return actor_dist.log_prob(action.squeeze(-1)).unsqueeze(-1), actor_dist.entropy()
        
    def get_best_lr(self, obs: Tensor):
        logits = self.forward(obs)
        return self.lrs[logits.argmax().item()]
    
    # def get_best_lr_prob(self):
    #     return self.forward().max().item()

