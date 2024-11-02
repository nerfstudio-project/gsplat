import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim
import torch.distributions as dist
from src.ppo.base_policy import Actor, Critic
from src.ppo.base_env import Env


class LRCritic(Critic):
    def __init__(self, train_env: Env, test_env: Env, input_dim: int = 1024, h_dim: int = 64):
        super().__init__()
        # should map obs (image) to value
        self.layers = [
            nn.Linear(input_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
        ]
        self.network = nn.Sequential(*self.layers)
        self.train_env = train_env
        self.test_env = test_env
        
        self.set_env(train_mode=True)
    
    def set_env(self, train_mode: bool):
        assert train_mode or self.test_env is not None, "Test env must be set if train mode is False"
        self.env = self.train_env if train_mode else self.test_env

    def forward(self, obs: Tensor):        
        obs = obs.to(torch.int)
        enc_images = self.env.get_encoded_images(obs)
        
        # Ensure obs is properly reshaped for the network
        batch_size = obs.shape[0] if len(obs.shape) > 3 else 1
        # obs = obs.view(batch_size, -1)  # Flatten input to (batch_size, features)

        return self.network(enc_images)

        # For debugging, simply return mean of psnr's over diff lr for this img
        # values = self.env.get_mean_reward(obs)
        # return values

class LRActor(Actor):
    def __init__(self, train_env: Env, test_env: Env = None, lrs: list[float] = None, input_dim: int = 1024, h_dim: int = 64):
        super().__init__()
        if lrs:
            self.lrs = Tensor(lrs)
        else:
            self.lrs = Tensor([10**-i for i in range(10)])
        self.lrs = self.lrs.to(device=train_env.device)
        
        print(f"Actor using learning rates: {self.lrs}")
        self.layers = [
            nn.Linear(input_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, len(self.lrs)),
            nn.Softmax(dim=-1)
        ]
        self.network = nn.Sequential(*self.layers)
        self.train_env = train_env
        self.test_env = test_env
        
        self.set_env(train_mode=True)

    def set_env(self, train_mode: bool):
        assert train_mode or self.test_env is not None, "Test env must be set if train mode is False"
        self.env = self.train_env if train_mode else self.test_env
        
    def forward(self, obs: Tensor):
        # convert from torch float to int
        obs = obs.to(torch.int)
        
        enc_images = self.env.get_encoded_images(obs) # get imgs from idx
        return self.network(enc_images)

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
        best_lr_idx = logits.argmax(dim=-1)

        return self.lrs[best_lr_idx]
    
    # def get_best_lr_prob(self):
    #     return self.forward().max().item()

