import torch

from rollout_buffer import RolloutBuffer
from envs.env_2d import TiledInit2DEnv

env = TiledInit2DEnv()
# Define observation and action space shapes
obs_shape = (env.obs_dim,)  # for example, (4,) for a 4D observation
action_shape = (env.action_dim,)  # for example, (2,) for a 2D action space
buffer_size = 10  # Can be set based on the number of steps or episodes per update
device = "cuda" if torch.cuda.is_available() else "cpu"

class Policy:
    def select_action(self, state):
        """
        Select an action based on the current state.
        Returns: action, log_prob, value
        """
        return torch.zeros(state.shape), 0.0, 1
    
policy = Policy()
# Initialize the buffer
rollout_buffer = RolloutBuffer(buffer_size, obs_shape, action_shape, device=device)

# During rollout collection
while not rollout_buffer.is_full():
    state = env.get_observation()
    action, log_prob, value = policy.select_action(state)  # Actor-Critic networks
    next_state, reward, done = env.step(action)

    # Add to the buffer
    rollout_buffer.add(state, action, reward, log_prob, value, done)

    # If done, reset the environment
    if done:
        state = env.reset()

# After rollout is complete, get data for optimization
for data in rollout_buffer.get():
    print(data)
    states = data['states']
    actions = data['actions']
    log_probs = data['log_probs']
    rewards = data['rewards']
    values = data['values']
    dones = data['dones']

# Perform PPO optimization based on collected data
# For example, calculate advantages, returns, and update policy and value networks

# Clear buffer after optimization
rollout_buffer.clear()