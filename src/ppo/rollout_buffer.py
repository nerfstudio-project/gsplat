import torch
from torch import Tensor

class RolloutBuffer:
    # TODO: make vectorized if we can vectorize the env
    def __init__(
        self,
        buffer_size,
        observation_shape,
        action_shape,
        gamma: float = 0.99,
        gae_lambda: float = 1,
        device="cuda"
    ):
        self.buffer_size = buffer_size
        self.device = device
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Pre-allocate storage for the buffer on the correct device
        self.obs = torch.zeros((buffer_size, *observation_shape), device=device)
        self.actions = torch.zeros((buffer_size, *action_shape), device=device)
        self.log_probs = torch.zeros(buffer_size, device=device)
        self.rewards = torch.zeros(buffer_size, device=device)
        self.returns = torch.zeros(buffer_size, device=device)    # rtg - to train the critic
        self.advantages = torch.zeros(buffer_size, device=device) # to train the actor
        self.values = torch.zeros(buffer_size, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.bool, device=device)
        
        self.ptr = 0  # Pointer to keep track of the current position

    def add(self, obs, action, reward, log_prob, value, done):
        """
        Add a new experience to the buffer.
        Make sure that inputs are already on the correct device (cuda or cpu).
        """
        assert not self.is_full(), "Buffer is already full. Cannot add new experiences."

        obs = torch.as_tensor(obs).to(self.device)
        action = torch.as_tensor(action).to(self.device)
        reward = torch.as_tensor(reward).to(self.device)
        log_prob = torch.as_tensor(log_prob).to(self.device)
        value = torch.as_tensor(value).to(self.device)
        done = torch.as_tensor(done, dtype=torch.bool).to(self.device)

        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        self.ptr += 1

    def compute_returns_and_advantage(self, last_values: Tensor, dones: Tensor) -> None:
        """
        *** From sb3's RolloutBuffer class ***
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        # last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        last_values = last_values.to(self.device)

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = (~dones).float().to(self.device)
                next_values = last_values
            else:
                next_non_terminal = (~self.dones[step + 1]).float().to(self.device)
                next_values = self.values[step + 1]

            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        self.returns = self.advantages + self.values

    def get(self, batch_size=None, shuffle=False):
        """
        Retrieve the data from the buffer in batches for optimization.
        If batch_size is None, return the entire buffer.
        Otherwise, return the data in batches of the specified size.

        If shuffle is True, shuffle the indices before extracting batches.
        """
        assert self.is_full(), "Buffer was not fully filled before calling get()."
        
        indices = torch.randperm(self.ptr, device=self.device) if shuffle else torch.arange(self.ptr, device=self.device)
        yield from self._generate_batches(indices, batch_size)

    def _generate_batches(self, indices, batch_size):
        """
        Generate data batches based on the provided indices.
        """
        if batch_size is None:
            batch_size = self.buffer_size

        num_batches = (len(indices) + batch_size - 1) // batch_size  # Handle remainder automatically
        for i in range(num_batches):
            batch_indices = indices[i * batch_size: min(self.buffer_size, (i + 1) * batch_size)]
            yield self._extract_batch(batch_indices)

    def _extract_batch(self, batch_indices):
        """
        Extract a batch given the provided indices.
        """
        return {
            'obs': self.obs[batch_indices],
            'actions': self.actions[batch_indices],
            'log_probs': self.log_probs[batch_indices].unsqueeze(-1),
            'rewards': self.rewards[batch_indices].unsqueeze(-1),
            'returns': self.returns[batch_indices].unsqueeze(-1),
            'advantages': self.advantages[batch_indices].unsqueeze(-1),
            'values': self.values[batch_indices].unsqueeze(-1),
            'dones': self.dones[batch_indices].unsqueeze(-1)
        }

    def reset(self):
        """
        Reset the buffer for the next iteration.
        """
        self.ptr = 0
        self.obs.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.returns.zero_()
        self.advantages.zero_()
        self.log_probs.zero_()
        self.values.zero_()
        self.dones.zero_()

    def is_full(self):
        """
        Check if the buffer has reached its size limit.
        """
        return self.ptr >= self.buffer_size
