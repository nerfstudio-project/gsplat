import math
import os
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch.nn as nn
import torch
import tyro
from PIL import Image
from torch import Tensor, optim

from gsplat import rasterization, rasterization_2dgs

from image_fitting import SimpleTrainer
import random

def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

seed_everything(42)

class PredictLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(10))
        self.lrs = [10**-i for i in range(10)]
        # self.lrs = [0] * 10
        # self.target_idx = 0
        # self.lrs[self.target_idx] = 0.01

    def forward(self):
        return torch.log_softmax(self.logits, dim=-1)

    def select_action(self):
        log_probs = self.forward()
        action = torch.multinomial(log_probs.exp(), 1)
        return action, log_probs[action]

    def get_lr(self, action):
        return self.lrs[action.item()]

device = "cuda" if torch.cuda.is_available() else "cpu"
lr_predictor = PredictLR().to(device)

optimizer = optim.Adam(lr_predictor.parameters(), lr=0.01)

height: int = 256
width: int = 256
num_points: int = 100000
save_imgs: bool = True
training_iterations: int = 1000
lr: float = 0.01
model_type: Literal["3dgs", "2dgs"] = "3dgs"

gt_image = torch.ones((height, width, 3)) * 1.0
# make top left and bottom right red, blue
gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])

# [(lr, final_loss)]
buffer_size = 1
num_epochs = 2
num_updates = int(1e4)
lr_losses = {}
import json

if not os.path.exists(f"lr_losses_{training_iterations}.json"):
        
    for lr in lr_predictor.lrs:
        print("*" * 50)
        print(f'currently training with lr={lr}')
        trainer = SimpleTrainer(gt_image=gt_image, num_points=num_points)
        losses = trainer.train(
            iterations=training_iterations,
            lr=lr,
            save_imgs=save_imgs,
            model_type=model_type,
        )
        lr_losses[lr] = losses

    # Write the lr_losses to a JSON file
    output_filename = f"lr_losses_{training_iterations}.json"
    with open(output_filename, 'w') as f:
        json.dump(lr_losses, f)
else:
    with open(f"lr_losses_{training_iterations}.json", 'r') as f:
        lr_losses = json.load(f)


for update in range(num_updates):
    # Collect rollout
    log_probs = []
    rewards = []

    for _ in range(buffer_size):
        lr_idx, log_prob = lr_predictor.select_action()
        lr = lr_predictor.get_lr(lr_idx)
        losses = lr_losses[str(lr)]

        final_loss, initial_loss = losses[-1], losses[0]
        print(final_loss, initial_loss)
        loss_reduction = final_loss - initial_loss
        loss_reduction = loss_reduction
        # loss_reduction = loss_reduction.detach().cpu().numpy()
        reward = -loss_reduction
        
        log_probs.append(log_prob)
        rewards.append(reward)

    # print(f"rewards: {rewards}")
    log_probs = torch.stack(log_probs)
    rewards = torch.tensor(rewards, device=device)

    # Calculate policy gradient loss
    policy_loss = -torch.mean(log_probs * rewards)
    
    # Update the network
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    # Optionally, print some statistics
    print("=" * 100)
    print(f"started at loss = {initial_loss}, finished at loss = {final_loss} (difference of {-1 * loss_reduction})for lr idx {lr_idx} (lr {lr})")
    print(f"Update {update}, Avg Reward: {rewards.mean().item()}")
    print(f'best LR: {lr_predictor.lrs[lr_predictor.logits.argmax().item()]}')
    print(f'softmax: {lr_predictor.forward().exp()}')
    print(f'best index: {lr_predictor.logits.argmax().item()}')
