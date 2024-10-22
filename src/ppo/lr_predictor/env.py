import numpy as np
import torch
import json
import os
from abc import ABC, abstractmethod
from examples.image_fitting import SimpleTrainer
from src.ppo.base_env import Env
from PIL import Image
from torchvision import transforms
from src.utils import image_path_to_tensor

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
        n_trials: int = 5,
        device='cuda',
        img_encoder: str = 'dino'
    ):
        # Environment state would be the 2d image
        # action: tile weights
        # 
        self.max_steps = 1
        self.num_points = num_points
        self.iterations = iterations
        # self.lrs = [0.007 + i*0.001 for i in range(10)] #(1, 0.1, 0.01,)
        self.lrs = [0.005 + i*0.001 for i in range(20)] #(1, 0.1, 0.01,)
        self.lrs = [round(lr, 5) for lr in self.lrs]

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
        img_name = img_path.split('/')[-1].split('.')[0]
        
        losses_json_path = os.path.join(
            current_dir, f"lr_losses_{self.iterations}_iterations{self.num_points}_points_{n_trials}_trials_{img_name}.json"
        )
        
        self.original_image = image_path_to_tensor(img_path)
        
        # if os.path.exists(losses_json_path):
        if True:
            with open(losses_json_path, 'r') as f:
                self.lr_losses = json.load(f)
                self.lr_losses = {str(float(lr)) : losses for lr, losses in self.lr_losses.items()}
        else:
            
            lr_losses = {str(lr): [] for lr in self.lrs}
            for lr in self.lrs:
                for trial_num in range(n_trials):
                    print("*" * 50)
                    print(f'currently training with lr={lr}, trial {trial_num}')
                    trainer = SimpleTrainer(gt_image=self.original_image, num_points=num_points)
                    losses, _ = trainer.train(
                        iterations=self.iterations,
                        lr=lr,
                        save_imgs=False,
                        model_type='3dgs',
                    )
                    losses[0] = 1.0
                    losses = np.array(losses)
                    print(f"achieved final loss: {losses[-1]}, psnr: {10 * np.log10(1 / losses[-1])}")
                    lr_losses[str(lr)].append(losses) 

            lr_losses_means = {lr: np.mean(np.array(losses), axis=0).tolist() for lr, losses in lr_losses.items()}
            
            # Write the lr_losses to a JSON file
            output_filename = losses_json_path
            with open(output_filename, 'w') as f:
                json.dump(lr_losses_means, f)
            self.lr_losses = lr_losses_means
        
        # tensor of losses where at index i, the losses for lr=lrs[i] are stored
        self.lr_losses_tensor = torch.tensor(
            [self.lr_losses[str(lr)] for lr in self.lrs], device=self.device
        ).to(self.device)
                
        batch_losses = self.lr_losses_tensor
        mse_err = batch_losses[:, 0] - batch_losses[:, -1]
        psnr = 10 * torch.log10(1 /  batch_losses[:, -1])
        
        print(f"initial mse_err: {mse_err},\n initial psnr: {psnr}")

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
            
        # Convert actions to int to use as indices
        actions_int = action.long()
        batch_losses = self.lr_losses_tensor[actions_int]

        # mse_diff = batch_losses[:, 0] - batch_losses[:, -1]
        # Using PSNR as reward (TODO: normalize for critic?)
        reward_psnr = 10 * torch.log10(1 / batch_losses[:, -1])
        # print(f"batch_losses: {reward_psnr} for action: {action}")

        # torch bool
        done = torch.as_tensor(True, device=self.device, dtype=torch.bool)
        
        # print(f"for action: {action}, reward: {reward}, using lr: {self.lrs[int(action.item())]}")
        return self.get_observation(), reward_psnr, done

    def get_observation(self):
        """
        Return the current state
        """
        return self.img