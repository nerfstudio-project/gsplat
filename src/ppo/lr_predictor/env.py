import numpy as np
import torch
import json
import os
from abc import ABC, abstractmethod
from examples.image_fitting import SimpleTrainer
from src.ppo.base_env import Env
from PIL import Image
from torchvision import transforms

def preprocess(img_path: str) -> torch.tensor:
    input_image = Image.open(img_path)
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    return input_tensor


class LREnv(Env):
    """
    A test environment where the agent's actions are used to simulate 
    the process of training a neural network.
    """
    def __init__(
        self, 
        img_path: str,
        num_points: int, 
        iterations: int,
        # TODO: remove observation_shape?
        observation_shape: tuple, 
        action_shape: tuple,
        device='cuda',
        img_encoder: str = 'dino'
    ):
        # Environment state would be the 2d image
        # action: tile weights
        # 
        self.max_steps = 1
        self.num_points = num_points
        self.iterations = iterations
        self.lrs = [10**-i for i in range(10)]
        self.device = device
        # self.observation_shape = img.shape
        self.action_shape = action_shape

        if img_encoder == 'dino':
            self.img_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            self.observation_shape = (self.img_encoder.embed_dim,)
            print("Using DINO large distilled as encoder")
        
        with torch.no_grad():
            preprocessed = preprocess(img_path)
            self.img = self.img_encoder(preprocessed.unsqueeze(0)).to(device) # TODO: maybe squeeze?

        # compute losses for each LR
        current_dir = os.path.dirname(os.path.abspath(__file__))
        losses_json_path = os.path.join(
            current_dir, f"lr_losses_{self.iterations}_iterations{self.num_points}_points.json"
        )
        if os.path.exists(losses_json_path):
            with open(losses_json_path, 'r') as f:
                self.lr_losses = json.load(f)
        else:
            lr_losses = {}
            for lr in self.lrs:
                print("*" * 50)
                print(f'currently training with lr={lr}')
                trainer = SimpleTrainer(gt_image=self.img, num_points=num_points)
                losses, _ = trainer.train(
                    iterations=self.iterations,
                    lr=lr,
                    save_imgs=False,
                    model_type='3dgs',
                )
                lr_losses[str(lr)] = losses

            # Write the lr_losses to a JSON file
            output_filename = losses_json_path
            with open(output_filename, 'w') as f:
                json.dump(lr_losses, f)
            self.lr_losses = lr_losses
        
        # map from idx to lr_losses
        self.lr_losses_idx_map = {i: self.lr_losses[str(lr)] for i, lr in enumerate(self.lrs)}
        
        actions = [i for i in range(len(self.lrs))]
        batch_losses = torch.stack(
            [torch.tensor(self.lr_losses_idx_map[int(idx)], device=self.device) for idx in actions]
        )
        reward = batch_losses[:, 0] - batch_losses[:, -1]
        
        print(f"initial reward: {reward}")
                
    def reset(self):
        """
        Reset the environment to an initial state.
        """
        
        return self.get_observation()

    def step(self, action: int):
        """
        Simulate applying the action (inputs to a neural network) and return 
        the next state, reward, and whether the episode is done.
        
        Args:
            action: The input action that simulates neural network input.
        
        Returns:
            next_state: The new state after applying the action.
            reward: The reward, which could be based on the network performance.
            done: A boolean indicating if the episode has ended.
        """
        # unsqueeze action if 0-d tensor
        if len(action.shape) == 0:
            action = action.unsqueeze(-1)
        # Convert actions to indices and lookup losses
        batch_losses = torch.stack(
            [torch.tensor(self.lr_losses_idx_map[int(idx)], device=self.device) for idx in action]
        )

        # Compute reward as losses[0] - losses[-1] for each action in batch (vectorized)
        reward = batch_losses[:, 0] - batch_losses[:, -1]

        # torch bool
        done = torch.as_tensor(True, device=self.device, dtype=torch.bool)
        
        # print(f"for action: {action}, reward: {reward}, using lr: {self.lrs[int(action.item())]}")
        return self.get_observation(), reward, done

    def get_observation(self):
        """
        Return the current state
        """
        return self.img