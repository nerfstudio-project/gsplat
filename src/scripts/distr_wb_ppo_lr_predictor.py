#!/secondary/home/aayushg/miniconda3/envs/gsplat/bin/ python3
import matplotlib.pyplot as plt
import torch
import wandb
import os
from dataclasses import dataclass
import tyro
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

def eval_policy(policy, env, verbose=False):
    policy.actor.eval()
    policy.critic.eval()
    num_match = 0
    with torch.no_grad():
        img_idx_tensor = torch.arange(env.num_images, device=device)
        lr = policy.actor.get_best_lr(img_idx_tensor)
        value = policy.critic(img_idx_tensor)

        num_match += (lr == env.best_lr[:env.num_images]).sum().item()
        if verbose:
            for i in range(env.num_images):
                idx = img_idx_tensor[i]
                print(f"img idx {i}: best: {env.best_lr[i]:8.4f}; pred: lr {lr[i]:8.4f}, idx {idx}, value: {value[i].item():8} ")
        # print(f"best: {env.best_lr[:env.num_images]}; pred: lr {lr}")         
            # print(f"img idx {i}: best: {env.best_lr[i]:8.4f}; pred: lr {lr:8.4f}, idx {idx}, value: {value.item():8.3f}")
    policy.actor.train()
    policy.critic.train()
    return num_match


@dataclass
class PPOConfig:
    n_epochs: int = 5
    batch_size: int = 32
    buffer_size: int = 128
    num_updates: int = 300
    entropy_coeff: float = 0.2
    log_interval: int = 1
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    clip_epsilon: float = 0.18
    device: str = 'cuda'

from dataclasses import dataclass, field

def get_env_lrs():
    return [0.005 + 0.001 * i for i in range(40)]

@dataclass
class EnvConfig:
    dataset_path: str = 'src/data/small_mipnerf'
    num_points: int = 100000
    iterations: int = 1000
    observation_shape: tuple = (256, 256, 3)
    lrs: list[float] = field(default_factory=get_env_lrs)
    num_trials: int = 10
    action_shape: tuple = (1,)
    device: str = 'cuda'
    img_encoder: str = 'dino'

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

            "entropy": self.ppo.logger["entropy"][-1],
            "surrogate_loss": self.ppo.logger["surr_loss"][-1],
            "avg_advantage": self.ppo.logger["avg_advantages"][-1],
            "avg_critic_value": self.ppo.logger["avg_critic_values"][-1],
        }
        
        wandb.log(metrics, step=self.iteration)
        self.iteration += 1
        return num_matches

def setup_wandb_sweep(method: str='grid'):
    sweep_config = {
        'program': 'src/scripts/wb_ppo_lr_predictor.py',
        'command': ['/secondary/home/aayushg/miniconda3/envs/gsplat/bin/python3', '${program}', '--run_sweep_agent'],  # Updated this line
        'method': method,
        'parameters': {
            'n_epochs': {'values': [3, 5, 7]},
            'batch_size': {'values': [256, 512, 1024]},
            'buffer_multiplier': {'values': [1]},
            'num_updates': {'values': [300, 400, 500]},
            'entropy_coeff': {'values': [0.05, 0.1, 0.15, 0.20]},
            'actor_lr': {'values': [3e-4, 1e-3]},
            'critic_lr': {'values': [1e-4]},
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

def train(config: PPOConfig, is_sweep: bool = False, save_ckpt: bool = False, load_ckpt_id: str = None) -> None:
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
    train_env_config = EnvConfig(device=config.device)
    
    train_env = LREnv(
        dataset_path=train_env_config.dataset_path,
        num_points=train_env_config.num_points,
        num_iterations=train_env_config.iterations,
        observation_shape=train_env_config.observation_shape,
        action_shape=train_env_config.action_shape,
        device=train_env_config.device,
        img_encoder=train_env_config.img_encoder,
        lrs=train_env_config.lrs,
        num_trials=train_env_config.num_trials,
    )

    actor = LRActor(
        env=train_env,
        lrs=train_env.lrs,
        input_dim=train_env.encoded_img_shape[0]
    )
    critic = LRCritic(
        env=train_env,
        input_dim=train_env.encoded_img_shape[0]
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
        env=train_env,
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
    wandb_callback = WandbCallback(policy, train_env, ppo, is_sweep=is_sweep)
    ppo.log_callback = wandb_callback

    # Train
    if load_ckpt_id:
        ckpt_path = f"src/results/ppo_lr_predictor_{load_ckpt_id}.pt"
        ppo.load_policy(ckpt_path, device=config.device)
    else:
        ppo.train(total_timesteps=config.num_updates * config.buffer_size)

    if save_ckpt:
        ckpt_path = f"src/results/ppo_lr_predictor_{wandb.run.id}.pt"
        ppo.save_policy(ckpt_path)
        wandb.save(ckpt_path)

    # Eval
    test_env_config = EnvConfig(device=config.device)
    test_env = LREnv(
        dataset_path=test_env_config.dataset_path+"_test",
        num_points=test_env_config.num_points,
        num_iterations=test_env_config.iterations,
        observation_shape=test_env_config.observation_shape,
        action_shape=test_env_config.action_shape,
        device=test_env_config.device,
        img_encoder=test_env_config.img_encoder,
        lrs = test_env_config.lrs,
        num_trials = test_env_config.num_trials,
    )

    actor.env = test_env
    critic.env = test_env
    num_match = eval_policy(policy, test_env, verbose=True)
    wandb.log({"num_matches_test": num_match})
    print(f"Number of matches on test set: {num_match}")
    

def train_with_wandb(
    config: PPOConfig, 
    project_name: str, 
    is_sweep: bool = False,
    eval: bool = False, 
    save_ckpt: bool = False, 
    load_ckpt_id: str = None
) -> None:
    """Wrapper function to handle wandb initialization and training"""
    with wandb.init(
        project=project_name,
        config=vars(config) if not is_sweep else None,
    ) as run:
        train(config, is_sweep, save_ckpt, load_ckpt_id)

def main(
    sweep: bool = False,
    sweep_id: str = None,
    device: str = 'cuda',
    gpu_id: int = 0,
    project_name: str = "ppo_lr_predictor_distr_temp",
    entity: str = "rl_gsplat",
    load_ckpt_id: str = None
):
    # must create new sweep
    if sweep and not sweep_id:

        # Create sweep and run agent in same process
        wandb_config, total_combinations = setup_wandb_sweep()
        sweep_id = wandb.sweep(
                wandb_config,
                project=project_name,
                entity=entity
        )
        full_id = f"{entity}/{project_name}/{sweep_id}"
        print(f"*******Sweep: {entity}/{project_name}/{sweep_id} **********")
        print(f"*******Sweep ID: {sweep_id} **********")
        with open("./temp/sweep_id.txt", "w") as f:
            f.write(full_id)
        
        return
    
    elif sweep and sweep_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = f"cuda:{0}"
        print(f'cuda visible devices: {os.environ["CUDA_VISIBLE_DEVICES"]}')
        print(f'gpu id: {gpu_id}')
        print(f'device: {device}')
        config = PPOConfig(device=device)
        wandb_config, total_combinations = setup_wandb_sweep()
        
        wandb.agent(
            sweep_id,
            lambda: train_with_wandb(config, project_name, is_sweep=True),
            count=total_combinations
        )

        return
    
    else: # train without sweep
        # our 
        config = PPOConfig(
            batch_size=64,
            buffer_size=512,
            clip_epsilon=.18,
            n_epochs=7,
            num_updates=300,
            entropy_coeff=0.15,
            device=device
        )

        train_with_wandb(
            config,
            project_name,
            is_sweep=False,
            eval=True,
            save_ckpt=True if not load_ckpt_id else False,
            load_ckpt_id=load_ckpt_id
        )
    

if __name__ == "__main__":
    tyro.cli(main)