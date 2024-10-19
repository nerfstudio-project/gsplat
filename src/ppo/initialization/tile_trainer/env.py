import numpy as np
import torch
import json
import os
from abc import ABC, abstractmethod
from src.ppo.tile_trainer import TileTrainer
from src.ppo.base_env import Env

class Tiled2DEnv(Env):
    """
    A test environment where the agent's actions are used to simulate 
    the process of training a neural network.
    """

    def __init__(self, img, num_points: int, iterations: int, lr: float):
        # Environment state would be the 2d image
        # action: tile weights
        # 
        self.max_steps = 1

        self.num_points = num_points
        self.iterations = iterations
        self.lr = lr
    
    def reset(self):
        """
        Reset the environment to an initial state.
        """
        
        return self.img

    def step(self, action: list[float]):
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
        
        trainer = TileTrainer(
            gt_image=self.img,
            num_points=self.num_points,
            tile_weights=action
        )
        losses, _ = trainer.train(
            iterations=self.iterations,
            lr=self.lr,
            save_imgs=True,
            model_type='3dgs',
        )
        reward = losses[0] - losses[-1]
        done = True
        
        return self.get_observation(), reward, done

    def get_observation(self):
        """
        Return the current state
        """
        return self.img