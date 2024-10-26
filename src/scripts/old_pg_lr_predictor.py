import math
import os
import json
import matplotlib.pyplot as plt
import time
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from typing import Literal, Optional
from scipy.signal import butter, filtfilt
import time

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
# choosing lr
# - based on this image, choose a lr
# primitive initialization
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Linear(1, 1)

    def forward(self, action: int, state=None):
        return self.network(action)
    

class Actor(nn.Module):
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
        lr_idx = torch.multinomial(log_probs.exp(), 1)
        return self.lrs[lr_idx], log_probs[lr_idx]

    
    def get_best_lr(self):
        return self.lrs[self.logits.argmax().item()]
    
    def get_best_lr_prob(self):
        return self.forward().exp().max().item()

def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor 

device = "cuda" if torch.cuda.is_available() else "cpu"
lr_predictor = Actor().to(device)

optimizer = optim.Adam(lr_predictor.parameters(), lr=0.01)

log_iter = 50
height: int = 256
width: int = 256
num_points: int = 100000
save_imgs: bool = True
training_iterations: int = 1000
lr: float = 0.01
model_type: Literal["3dgs", "2dgs"] = "3dgs"

im_path = 'images/adam.jpg'
if not im_path:    
    gt_image = torch.ones((height, width, 3)) * 1.0
    # make top left and bottom right red, blue
    gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
    gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])
else:
    gt_image = image_path_to_tensor(im_path)

buffer_size = 10
num_epochs = 2
num_updates = int(1e4)
lr_losses = {}

if True:
# if not os.path.exists(f"lr_losses_{training_iterations}.json"):
    print("generating trajectories")
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

lr_probs = defaultdict(list)
reinforce_losses = []
rewards_history = []

print("training lr predictor")
pbar = tqdm(range(num_updates), desc="best LR prob")

rollout_times = []
update_times = []
log_probs = torch.empty(buffer_size, device=device)
rewards = torch.empty(buffer_size, device=device)
for update in pbar:

    time_start = time.time()
    # Collect rollout
    log_probs = []
    rewards = []

    # Pre-allocate tensors
    log_probs = torch.empty(buffer_size, device=device)
    rewards = torch.empty(buffer_size, device=device)

    for i in range(buffer_size):
        lr, log_prob = lr_predictor.select_action()
        losses = lr_losses[str(lr)]

        final_loss, initial_loss = losses[-1], losses[0]
        loss_reduction = final_loss - initial_loss
        reward = -loss_reduction
        
        log_probs[i] = log_prob
        rewards[i] = reward

    time_rollouts = time.time()
    rollout_times.append(time_rollouts - time_start)

    # Calculate policy gradient loss
    policy_loss = -(log_probs * rewards).mean()
    reinforce_losses.append(policy_loss.item())
    rewards_history.append(rewards.mean().item())

    pbar.set_description(f"best LR {lr_predictor.get_best_lr()} with prob: {lr_predictor.get_best_lr_prob():.4f}")

    # Update the network
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    verbose = False

    time_end = time.time()
    update_times.append(time_end - time_start)

    if update % 50:
        print(f"avg rollout time: {np.mean(rollout_times)}, avg update time: {np.mean(update_times)}")
        
    if update % log_iter == 0 and verbose:
        # Optionally, print some statistics
        print("=" * 100)
        print(f"started at loss = {initial_loss}, finished at loss = {final_loss} (difference of {-1 * loss_reduction})for lr idx {lr_idx} (lr {lr})")
        print(f"Update {update}, Avg Reward: {rewards.mean().item()}")
        print(f'best LR: {lr_predictor.get_best_lr()}')
        print(f'softmax: {lr_predictor.forward().exp()}')
        # print(f'best index: {lr_predictor.logits.argmax().item()}')
        print(f'best lr: {lr_predictor.get_best_lr()}')
    for prob, lr in zip(lr_predictor.forward().exp(), lr_predictor.lrs):
        lr_probs[str(lr)].append(prob.detach().cpu().numpy())



updates = list(range(num_updates))
optimal_lr = lr_predictor.get_best_lr()

optimal_lr_probs = lr_probs[str(optimal_lr)]
plt.plot(updates, optimal_lr_probs, label=f'Optimal LR: {optimal_lr}')
plt.xlabel('Updates')
plt.ylabel('Probability')
plt.title('Updates vs Probability of Optimal Learning Rate')
plt.legend()
# plt.show()
plt.savefig(f'lr_probs_{training_iterations}.png')

# Plot REINFORCE losses


def lowpass_filter(data, cutoff=0.1, fs=1.0, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

window_size = 100
box_filter = np.ones(window_size) / window_size

filtered_reinforce_losses = np.convolve(reinforce_losses, box_filter, mode='same')

plt.figure()
plt.plot(range(num_updates), reinforce_losses, label='REINFORCE Loss')
plt.xlabel('Updates')
plt.ylabel('Loss')
plt.title('REINFORCE Loss over Updates')
plt.legend()
plt.savefig(f'reinforce_losses_{training_iterations}.png')

plt.plot(range(num_updates), filtered_reinforce_losses, label='filtered REINFORCE Loss')
plt.xlabel('Updates')
plt.ylabel('Loss')
plt.title('REINFORCE Loss over Updates')
plt.legend()
plt.savefig(f'filtered_reinforce_losses_{training_iterations}.png')

# Plot rewards
plt.figure()
plt.plot(range(num_updates), rewards_history, label='Average Reward')
plt.xlabel('Updates')
plt.ylabel('Reward')
plt.title('Average Reward over Updates')
plt.legend()
plt.savefig(f'rewards_{training_iterations}.png')

print("executing training run with optimal lr")
# Launch a training job with the optimal LR for 2000 points
# num_points = 100000
save_img = True
training_iterations = 1000
trainer = SimpleTrainer(gt_image=gt_image, num_points=num_points)
# Assuming we have a function `train` that takes learning rate and number of points as arguments
save_path = f'results/{optimal_lr}_lr_{training_iterations}_iterations_{num_points}_points.png'
trainer.train(iterations=training_iterations, lr=optimal_lr, save_path=save_path)

# Save the training results if save_img is True
if save_img:
    plt.figure()
    plt.plot(range(num_updates), rewards_history[:num_updates], label='Average Reward')
    plt.xlabel('Points')
    plt.ylabel('Reward')
    plt.title('Average Reward over Points')
    plt.legend()
    plt.savefig(f'rewards_{training_iterations}_2000_points.png')

