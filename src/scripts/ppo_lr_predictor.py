
import matplotlib.pyplot as plt
import torch

from src.ppo.ppo import PPO
from src.ppo.base_policy import Policy
from src.ppo.lr_predictor.actor_critic import LRCritic, LRActor
from src.ppo.lr_predictor.env import LREnv
from src.utils import image_path_to_tensor, seed_everything

device = 'cuda'
seed_everything(42)

height: int = 256
width: int = 256
# im_path = 'images/adam.jpg'
im_path = None
if not im_path:    
    gt_image = torch.ones((height, width, 3)) * 1.0
    # make top left and bottom right red, blue
    gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
    gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])
    # save the image
    plt.imsave('src/data/simple.jpg', gt_image.numpy())
else:
    gt_image = image_path_to_tensor(im_path)


# Initialize environment and policy
env = LREnv(
    dataset_path='src/data/small_mipnerf',
    num_points=100000,
    iterations=1000,
    observation_shape=(height, width, 3),
    action_shape=(1,),
    device=device,
    img_encoder='dino'
)
print(f"observatino space: {env.observation_shape}")
actor = LRActor(lrs=env.lrs, input_dim=env.observation_shape[0])
critic = LRCritic(input_dim=env.observation_shape[0])
policy = Policy(
    actor=actor,
    critic=critic,
    device=device,
    actor_lr=3e-4,
    critic_lr=3e-4
)

print(f"initial: actor probs: {policy.actor(env.img)}")
log_callback = lambda policy: print(f"actor probs: {policy.actor(env.img)} \n"
                                    f"best lr: {policy.actor.get_best_lr(env.img)}")

num_updates = 100
buffer_size = 100
ppo = PPO(
    policy=policy,
    env=env,
    n_epochs=5,
    batch_size=buffer_size,
    buffer_size=buffer_size,
    log_interval=1,
    device=device,
    entropy_coeff=0.0,
    shuffle=True,
    normalize_advantages=True,
    log_callback=log_callback,
    plots_path='src/results/plot.png'
)

ppo.train(total_timesteps=num_updates*buffer_size)

print("done training")
