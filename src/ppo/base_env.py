import numpy as np
from abc import ABC, abstractmethod

class Env(ABC):
    def __init__(self, observation_shape=None, act_shape=None) -> None:
        self.step_count = 0
        self.num_steps = 100
        self.observation_shape = observation_shape
        self.action_shape = act_shape
        self.eval_mode = 'train'
        super().__init__()
        
    @abstractmethod
    def reset(self):
        """Reset the environment and return the initial observation."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        """
        Perform an action in the environment.
        Returns: next_state, reward, done (boolean)
        """
        raise NotImplementedError

    @abstractmethod
    def get_observation(self):
        """Return the current state/observation of the environment."""
        raise NotImplementedError