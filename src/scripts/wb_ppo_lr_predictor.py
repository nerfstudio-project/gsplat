#!/secondary/home/aayushg/miniconda3/envs/gsplat/bin/ python3
import matplotlib.pyplot as plt
import torch
import wandb
from dataclasses import dataclass
import argparse
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
        img_idx_tensor = torch.arange(env.num_images, device=device)
        lr = policy.actor.get_best_lr(img_idx_tensor)
        value = policy.critic(img_idx_tensor)

        num_match += (lr == env.best_lr[:env.num_images]).sum().item()
        # print(f"best: {env.best_lr[:env.num_images]}; pred: lr {lr}")         
            # print(f"img idx {i}: best: {env.best_lr[i]:8.4f}; pred: lr {lr:8.4f}, idx {idx}, value: {value.item():8.3f}")
    policy.actor.train()
    policy.critic.train()
    return num_match


@dataclass
class PPOConfig:
    n_epochs: int = 5
    batch_size: int = 128
    buffer_size: int = 512
    num_updates: int = 300
    entropy_coeff: float = 0.2
    log_interval: int = 1
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    clip_epsilon: float = 0.18
    device: str = 'cuda'
    dataset_path: str = 'src/data/small_mipnerf'

class WandbCallback:
    def __init__(self, policy, env, ppo, is_sweep: bool = False):
        self.policy = policy
        self.env = env
        self.ppo = ppo
        self.is_sweep = is_sweep
        self.iteration = 0

    def __call__(self, policy):
        num_matches = eval_policy(policy, self.env)
        
        # Base metrics logged in both modes
        metrics = {
            "num_matches": num_matches,
            "actor_loss": self.ppo.logger["actor_losses"][-1] if self.ppo.logger["actor_losses"] else None,
            "critic_loss": self.ppo.logger["critic_losses"][-1] if self.ppo.logger["critic_losses"] else None,
            "avg_reward": self.ppo.logger["avg_rewards"][-1] if self.ppo.logger["avg_rewards"] else None,
        }
        
        metrics.update({
            "entropy": self.ppo.logger["entropy"][-1],
            "surrogate_loss": self.ppo.logger["surr_loss"][-1],
            "avg_advantage": self.ppo.logger["avg_advantages"][-1],
            "avg_critic_value": self.ppo.logger["avg_critic_values"][-1],
        })
        
        wandb.log(metrics, step=self.iteration)
        self.iteration += 1
        return num_matches

def setup_wandb_sweep():
    sweep_config = {
        'program': 'src/scripts/wb_ppo_lr_predictor.py',  # Add this line
        'command': ['/secondary/home/aayushg/miniconda3/envs/gsplat/bin/python3', '${program}', '--run_sweep_agent'],  # Updated this line
        'method': 'grid',
        'parameters': {
            'n_epochs': {'values': [3, 5, 7]},
            'batch_size': {'values': [32, 64, 128]},
            'buffer_multiplier': {'values': [1, 2, 4, 8]},
            'num_updates': {'values': [200, 300, 400, 500]},
            'entropy_coeff': {'values': [0.0, 0.01, 0.05, 0.1]},
            'actor_lr': {'values': [1e-4, 3e-4, 1e-3]},
            'critic_lr': {'values': [1e-4]}, # no effect as critic is not trained (debug)
            'clip_epsilon': {'values': [0.15, 0.18, 0.2]}
            # 'critic_lr': {'values': [1e-4, 3e-4, 1e-3]}
        },
        'metric': {
            'name': 'num_matches',
            'goal': 'maximize'
        }
    }
    
    total_combinations = 1
    for _, values in sweep_config['parameters'].items():
        total_combinations *= len(values['values'])
    print(f"Total combinations: {total_combinations}")

    return sweep_config, total_combinations

def train(config: PPOConfig, is_sweep: bool = False) -> None:
    """Main training function used for both sweep and single runs"""
    if is_sweep:
        # For sweep: compute buffer_size from multiplier
        buffer_size = wandb.config.batch_size * wandb.config.buffer_multiplier
        config = PPOConfig(
            n_epochs=wandb.config.n_epochs,
            batch_size=wandb.config.batch_size,
            buffer_size=buffer_size,
            num_updates=wandb.config.num_updates,
            entropy_coeff=wandb.config.entropy_coeff,
            actor_lr=wandb.config.actor_lr,
            critic_lr=wandb.config.critic_lr,
            clip_epsilon=wandb.config.clip_epsilon,
            device=config.device
        )
        wandb.config.update({"buffer_size": buffer_size}, allow_val_change=True)
        
    # Initialize environment and models
    env = LREnv(
        dataset_path=config.dataset_path,
        num_points=100000,
        iterations=2000,
        observation_shape=(256, 256, 3),
        action_shape=(1,),
        device=config.device,
        img_encoder='dino'
    )

    actor = LRActor(
        env=env,
        lrs=env.lrs,
        input_dim=env.encoded_img_shape[0]
    )
    critic = LRCritic(
        env=env,
        input_dim=env.encoded_img_shape[0]
    )
    policy = Policy(
        actor=actor,
        critic=critic,
        device=config.device,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr
    )

    # First create PPO without callback
    ppo = PPO(
        policy=policy,
        env=env,
        n_epochs=config.n_epochs,
        batch_size=config.batch_size,
        buffer_size=config.buffer_size,
        log_interval=config.log_interval,
        device=config.device,
        entropy_coeff=config.entropy_coeff,
        clip_epsilon=config.clip_epsilon,
        shuffle=True,
        normalize_advantages=True,
        plots_path=f'src/results/plot_{wandb.run.id}.png'
    )

    # Then create and set the callback
    wandb_callback = WandbCallback(policy, env, ppo, is_sweep=is_sweep)
    ppo.log_callback = wandb_callback

    # Train
    ppo.train(total_timesteps=config.num_updates * config.buffer_size)

    # Log final plots for single runs
    if not is_sweep and ppo.plots_path:
        wandb.log({"training_plots": wandb.Image(ppo.plots_path)})

def train_with_wandb(config: PPOConfig, is_sweep: bool = False, project_name: str = "ppo_lr_predictor"):
    """Wrapper function to handle wandb initialization and training"""
    with wandb.init(
        project=project_name,
        config=vars(config) if not is_sweep else None,
    ) as run:
        train(config, is_sweep)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', action='store_true', help='Run hyperparameter sweep')
    parser.add_argument('--create_sweep_only', action='store_true', help='Only create sweep, do not run agent')
    parser.add_argument('--run_sweep_agent', action='store_true', help='Run as sweep agent')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
    args = parser.parse_args()

    config = PPOConfig(device=args.device)
    project = "ppo_lr_predictor"
    entity = "rl_gsplat"
    if args.create_sweep_only:
        # Just create and print sweep ID
        wandb_config, total_combinations = setup_wandb_sweep()
        sweep_id = wandb.sweep(
            wandb_config,
            project=project,
            entity=entity
        )
        print(f"*******Sweep ID: {entity}/{project}/{sweep_id} **********")
        return

    if args.run_sweep_agent:
        # Run as an agent for an existing sweep
        train_with_wandb(config, is_sweep=True)
        return

    if args.sweep:
        # Create sweep and run agent in same process
        wandb_config, total_combinations = setup_wandb_sweep()
        sweep_id = wandb.sweep(
            wandb_config,
            project=project,
            entity=entity
        )
        print(f"*******Sweep ID: {entity}/{project}/{sweep_id} **********")
        wandb.agent(
            sweep_id,
            lambda: train_with_wandb(config, is_sweep=True),
            count=total_combinations
        )
    else:
        # Regular single run
        print("Running single training run with fixed parameters...")
        train_with_wandb(config, is_sweep=False)

if __name__ == "__main__":
    main()