import numpy as np
import torch
import json
import warnings
import os
from collections import defaultdict
from examples.image_fitting import SimpleTrainer
from src.ppo.base_env import Env

from src.utils import image_path_to_tensor, dino_preprocess


class LREnv(Env):
    """
    A test environment where the agent's actions are used to simulate 
    the process of training a neural network.
    """
    def __init__(
        self, 
        lrs: list[float],
        dataset_path: str,
        num_points: int, 
        num_iterations: int,
        # TODO: remove observation_shape?
        observation_shape: tuple, 
        action_shape: tuple,
        num_trials: int = 1,
        device='cuda',
        img_encoder: str = 'dino'
    ):
        # Environment state would be the 2d image
        # action: tile weights
        # 
        self.max_steps = 1
        self.num_points = num_points
        self.num_trials = num_trials
        self.num_iterations = num_iterations
        
        self.lrs = lrs
        self.lrs = [round(lr, 5) for lr in self.lrs]

        self.device = device
        # self.observation_shape = img.shape
        self.action_shape = action_shape

        if img_encoder == 'dino':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.img_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            self.encoded_img_shape = (self.img_encoder.embed_dim,)
            print("Using DINO large distilled as encoder")
        self.observation_shape = (1,) # Just img index

        self.num_images = 0
        self.encoded_images = []
        self.orig_images = []
        self.img_names = []
        
        for img_name in os.listdir(dataset_path):
            self.num_images += 1
            full_path = os.path.join(dataset_path, img_name)
            orig_img = image_path_to_tensor(full_path)
            self.orig_images.append(orig_img)
            self.img_names.append(img_name)

            with torch.no_grad():
                preprocessed = dino_preprocess(full_path)
                encoded_img = self.img_encoder(preprocessed.unsqueeze(0)).to(device)

            self.encoded_images.append(encoded_img)
        print("=" * 100)    
        print(f'num images: {self.num_images}\n num_trials: {self.num_trials}\n num_points: {self.num_points}\n num_iterations: {self.num_iterations}, num learning rates: {len(self.lrs)}')
        print("=" * 100)    
        self.encoded_images = torch.stack(self.encoded_images)
        self.original_images = torch.stack(self.orig_images)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_name = dataset_path.split('/')[-1]
        losses_json_path = os.path.join(
            current_dir, f"{dataset_name}_lr_losses_final.json"
        )
        
        # self.num_images = 8
        
        # compute losses for each LR        
        if os.path.exists(losses_json_path):
            with open(losses_json_path, 'r') as f:
                self.img_to_lr_dict = json.load(f)
        else:
            img_to_lr_dict = {}
            for (i, _) in enumerate(self.original_images):
                original_img = self.orig_images[i]
                print(f"precomputing image {i+1}/{self.num_images}")
                lr_losses = {str(lr): [] for lr in self.lrs}
                for lr in self.lrs:
                    for trial_num in range(self.num_trials):
                        print("*" * 50)
                        print(f'currently training with lr={lr}, trial {trial_num}')
                        trainer = SimpleTrainer(gt_image=original_img, num_points=num_points)
                        losses, _ = trainer.train(
                            iterations=self.num_iterations,
                            lr=lr,
                            save_imgs=False,
                            model_type='3dgs',
                        )
                        losses = np.array(losses)
                        print(f"achieved final loss: {losses[-1]}, psnr: {10 * np.log10(1 / losses[-1])}")

                        lr_losses[str(lr)].append(losses) 
                
                lr_losses_means = {lr: np.mean(np.array(losses), axis=0).tolist() for lr, losses in lr_losses.items()}
                
                img_to_lr_dict[i] = {"lr_losses": lr_losses_means, "img_name": self.img_names[i]}

                # every other image, we write to json file as backup
                if i % 2 == 0:
                    output_filename = losses_json_path
                    with open(output_filename, 'w') as f:
                        json.dump(img_to_lr_dict, f, indent=4)
                    self.img_to_lr_dict = img_to_lr_dict    
                
            # Write the lr_losses to a JSON file
            output_filename = losses_json_path
            with open(output_filename, 'w') as f:
                json.dump(img_to_lr_dict, f)
            self.img_to_lr_dict = img_to_lr_dict

        lr_losses_list = []
        for img_idx, values in self.img_to_lr_dict.items():
            img_idx = int(img_idx)
            lr_losses, img_name = values['lr_losses'], values['img_name']
            img_lr_losses = torch.tensor(
                [lr_losses[str(lr)] for lr in self.lrs], device=self.device
            )
            lr_losses_list.append(img_lr_losses)
            
        self.lr_losses_tensor = torch.stack(lr_losses_list)
        
        self.psnr = 10 * torch.log10(1 / self.lr_losses_tensor[:, :, -1])
        self.psnr = (self.psnr - 14.0) / 9.0
        
        self.psnr_stats = {
            "mean": self.psnr.mean(dim=1),
            "max": self.psnr.max(dim=1).values,
            "argmax": self.psnr.argmax(dim=1),
            "lr_max": torch.tensor(self.lrs).to(self.device)[self.psnr.argmax(dim=1)],
            "min": self.psnr.min(dim=1).values,
            "std": self.psnr.std(dim=1),
        }
        
        self.best_lr = self.psnr_stats['lr_max']
        
        grouped_stats = defaultdict(list)
        for img_idx, img_name in enumerate(self.img_names):
            scene_name = img_name.split('_')[0]
            grouped_stats[scene_name].append(img_idx)
    
        for scene_name, indices in grouped_stats.items():
            print("=" * 100)
            print(f"Scene: {scene_name}")
            print(f"{'Image Index':<12} {'Mean':<12} {'Max':<12} {'Best LR Idx':<12} {'Best LR'} {'Min':<12} {'Std':<12}")
            for img_idx in indices:
                print(f"{img_idx:<12} {self.psnr_stats['mean'][img_idx]:<12.4f} {self.psnr_stats['max'][img_idx]:<12.4f} {self.psnr_stats['argmax'][img_idx]:<12} {self.psnr_stats['lr_max'][img_idx]:<12.4f} {self.psnr_stats['min'][img_idx]:<12.4f} {self.psnr_stats['std'][img_idx]:<12.4f}")
            print("=" * 100)
                
        self.sample_new_img()

    def sample_new_img(self):
        self.current_img_idx = torch.tensor(np.random.randint(0, self.num_images), device=self.device)

    def reset(self):
        """
        Reset the environment to an initial state.
        """
        # Randomly select new img idx
        self.sample_new_img()
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
        batch_psnr = self.psnr[self.current_img_idx][actions_int]

        # mse_diff = batch_losses[:, 0] - batch_losses[:, -1]
        # Using PSNR as reward (TODO: normalize for critic?)
        # print(f"batch_losses: {reward_psnr} for action: {action}")

        # torch bool
        done = torch.as_tensor(True, device=self.device, dtype=torch.bool)
        
        # print(f"for action: {action}, reward: {reward}, using lr: {self.lrs[int(action.item())]}")
        return self.get_observation(), batch_psnr, done

    def get_observation(self):
        """
        Return the current state (img_idx)
        """
        return self.current_img_idx
    
    def get_encoded_images(self, img_idx: torch.tensor):
        imgs = self.encoded_images[img_idx]
        if len(imgs.shape) == 4:
            imgs = imgs.squeeze(1)
        if len(imgs.shape) == 3:
            imgs = imgs.squeeze(1)
        return imgs
    
    def get_mean_reward(self, img_idx: torch.tensor):
        return self.psnr_stats['mean'][img_idx]