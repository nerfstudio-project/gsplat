
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

def eval_policy(policy, env):
    policy.actor.eval()
    policy.critic.eval()
    num_match = 0
    with torch.no_grad():
        for i in range(env.num_images):
            embed = env.encoded_images[i]
            obs = embed.to(device)
            lr = policy.actor.get_best_lr(obs)
            idx = int((lr - env.lrs[0]) / (env.lrs[1] - env.lrs[0]))
            value = policy.critic(obs)
            num_match += int(lr == env.best_lr[i])
            print(f"img idx {i}: best: {env.best_lr[i]:8.4f}; pred: lr {lr:8.4f}, idx {idx}, value: {value.item():8.3f}")
    policy.actor.train()
    policy.critic.train()
    return num_match
        
log_callback = lambda policy: eval_policy(policy, env)
# log_callback = lambda policy: print(f"Current env image idx: {env.current_img_idx} \n"
#                                     f"actor probs: {policy.actor(env.get_observation())} \n"
#                                     f"best lr: {policy.actor.get_best_lr(env.get_observation())}")





# Initialize environment and policy
env = LREnv(
    dataset_path='src/data/small_mipnerf',
    num_points=100000,
    iterations=2000,
    observation_shape=(height, width, 3),
    action_shape=(1,),
    device=device,
    img_encoder='dino'
)

print(f"observation space: {env.observation_shape}")
actor = LRActor(lrs=env.lrs, input_dim=env.observation_shape[0])
critic = LRCritic(input_dim=env.observation_shape[0])
policy = Policy(
    actor=actor,
    critic=critic,
    device=device,
    actor_lr=3e-4,
    critic_lr=3e-4
)

#TODO: make PPO config dataclass? Only need to pass in config
wandb_sweep = True
if wandb_sweep:
    import wandb

    epochs_list = []
    buffer_size_list = []
    batch_size_list = []
    num_updates_list = []
    entropy_coeff_list = []
    parameters_dict = {
        'n_epochs': {
            'values': epochs_list
            },
        'buffer_size': {
            'values': buffer_size_list
            },
        'batch_size': {
            'values': batch_size_list
            },
        'num_updates': {
            'values': num_updates_list
            },
        'entropy_coeff': {
            'values': entropy_coeff_list
            }
        }
    sweep_config = {
        'method': 'grid',
        'parameters': parameters_dict
        }

    sweep_config['parameters'] = parameters_dict



    metric1 = {
        'name': 'num_matches',
        'goal': 'maximize'   
    }

    metric2 = {
        'name': 'max_avg_reward',
        'goal': 'maximize'
    }

    sweep_config['metric'] = metric1

    sweep_id = wandb.sweep(sweep_config, project="ppo_lr_predictor")



n_epochs = 5
buffer_size = 128
batch_size = 64
log_interval = 1
entropy_coeff = 0.0
num_updates = 250
plot_str = f"{n_epochs}_num{batch_size}_ent{entropy_coeff}_up{num_updates}"

@dataclass
class PPOConfig:
    num_epochs: int = 5
    batch_size: int = 64
    buffer_size: int = 128
    num_updates: int = 250
    entropy_coeff: float = 0.0
    log_interval: int = 1

config = PPOConfig()

def train_ppo(config: PPOConfig, wandb_sweep=False):
    if wandb_sweep:
        with wandb.init(config=config):
            config = wandb.config
    
    ppo = PPO(
    policy=policy,
    env=env,
    n_epochs=config.n_epochs,
    batch_size=config.batch_size,
    buffer_size=config.buffer_size,
    log_interval=log_interval,
    device=device,
    entropy_coeff=config.entropy_coeff,
    shuffle=True,
    normalize_advantages=True,
    log_callback=log_callback,
    plots_path=f'src/results/plot_{plot_str}.png'
)

ppo.train(total_timesteps=num_updates*buffer_size)

print("done training")
