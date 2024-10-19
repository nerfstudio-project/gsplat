import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, optim
import torch.distributions as dist
from src.ppo.base_policy import Actor, Critic

class LRCritic(Critic):
    def __init__(self):
        super().__init__()
        self.network = torch.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid(),
        )

    def forward(self, action: int, state=None):
        action = torch.tensor([[action]], dtype=torch.float32)
        return self.network(action).squeeze(-1)

class LRActor(Actor):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(10))
        self.lrs = [10**-i for i in range(10)]

    def forward(self):
        return nn.Softmax(self.logits, dim=-1)

    def select_action(self):
        log_probs = self.forward()
        actor_dist = dist.Categorical(logits=log_probs)
        lr_idx = actor_dist.sample()
        return self.lrs[lr_idx], actor_dist.log_probs[lr_idx]

    def log_prob(self, obs, acts):
        log_probs = self.forward()
        actor_dist = dist.Categorical(logits=log_probs)
        return log_probs[acts]

