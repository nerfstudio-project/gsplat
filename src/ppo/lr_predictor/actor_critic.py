import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim
import torch.distributions as dist
from src.ppo.base_policy import Actor, Critic

class LRCritic(Critic):
    def __init__(self):
        super().__init__()
        # should map obs (image) to value
        self.constant_value = nn.Parameter(torch.tensor(0.0001))  # Initial constant
        # self.network = torch.nn.Sequential(
        #     nn.Linear(1, 1),
        #     nn.Sigmoid(),
        # )

    def forward(self, obs: Tensor):
        # Ensure obs is properly reshaped for the network
        batch_size = obs.shape[0] if len(obs.shape) > 3 else 1
        # obs = obs.view(batch_size, -1)  # Flatten input to (batch_size, features)
        
        return self.constant_value.expand(batch_size).unsqueeze(-1)
    
class LRActor(Actor):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(10))
        self.lrs = Tensor([10**-i for i in range(10)])

    def forward(self, obs: Tensor):
        return F.softmax(self.logits, dim=-1)

    def actor_distribution(self, obs: Tensor):
        return dist.Categorical(logits=self.logits)
    
    def select_action(self, obs: Tensor):
        actor_dist = self.actor_distribution(obs)
        lr_idx = actor_dist.sample()
        return lr_idx, actor_dist.log_prob(lr_idx)

    def log_prob(self, obs, acts):
        actor_dist = self.actor_distribution(obs)
        return actor_dist.log_prob(acts)
    
    def evaluate_actions(self, obs: Tensor, action: Tensor):
        actor_dist = self.actor_distribution(obs)
        return actor_dist.log_prob(action), actor_dist.entropy()
        
    def get_best_lr(self):
        return self.lrs[self.logits.argmax().item()]
    
    def get_best_lr_prob(self):
        return self.forward().max().item()

