
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
    img_path='src/data/simple.jpg',
    num_points=1000,
    iterations=100,
    observation_shape=(height, width, 3),
    action_shape=(1,),
    device=device,
    img_encoder='dino'
)
print(env.observation_shape)
actor = LRActor(input_dim=env.observation_shape[0])
critic = LRCritic(input_dim=env.observation_shape[0])
policy = Policy(
    actor=actor,
    critic=critic,
    device=device,
    actor_lr=1e-2,
    critic_lr=1e-2
)

log_callback = lambda policy: print(f"actor probs: {policy.actor(env.img)} \n"
                                    f"critic values: {policy.critic(env.img)} \n"
                                    f"best lr: {policy.actor.get_best_lr(env.img)}")

num_updates = 100
buffer_size = 10
ppo = PPO(
    policy=policy,
    env=env,
    n_epochs=5,
    batch_size=buffer_size//2,
    buffer_size=buffer_size,
    log_interval=1,
    device=device,
    entropy_coeff=0.0,
    shuffle=True,
    normalize_advantages=True,
    log_callback=log_callback
)

ppo.train(total_timesteps=num_updates*buffer_size)

print("done training")

# updates = list(range(num_updates))
# optimal_lr = lr_predictor.get_best_lr()

# optimal_lr_probs = lr_probs[str(optimal_lr)]
# plt.plot(updates, optimal_lr_probs, label=f'Optimal LR: {optimal_lr}')
# plt.xlabel('Updates')
# plt.ylabel('Probability')
# plt.title('Updates vs Probability of Optimal Learning Rate')
# plt.legend()
# # plt.show()
# plt.savefig(f'lr_probs_{training_iterations}.png')

# # Plot REINFORCE losses


# def lowpass_filter(data, cutoff=0.1, fs=1.0, order=5):
#     nyquist = 0.5 * fs
#     normal_cutoff = cutoff / nyquist
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     y = filtfilt(b, a, data)
#     return y

# window_size = 100
# box_filter = np.ones(window_size) / window_size

# filtered_reinforce_losses = np.convolve(reinforce_losses, box_filter, mode='same')

# plt.figure()
# plt.plot(range(num_updates), reinforce_losses, label='REINFORCE Loss')
# plt.xlabel('Updates')
# plt.ylabel('Loss')
# plt.title('REINFORCE Loss over Updates')
# plt.legend()
# plt.savefig(f'reinforce_losses_{training_iterations}.png')

# plt.plot(range(num_updates), filtered_reinforce_losses, label='filtered REINFORCE Loss')
# plt.xlabel('Updates')
# plt.ylabel('Loss')
# plt.title('REINFORCE Loss over Updates')
# plt.legend()
# plt.savefig(f'filtered_reinforce_losses_{training_iterations}.png')

# # Plot rewards
# plt.figure()
# plt.plot(range(num_updates), rewards_history, label='Average Reward')
# plt.xlabel('Updates')
# plt.ylabel('Reward')
# plt.title('Average Reward over Updates')
# plt.legend()
# plt.savefig(f'rewards_{training_iterations}.png')

# print("executing training run with optimal lr")
# # Launch a training job with the optimal LR for 2000 points
# # num_points = 100000
# save_img = True
# training_iterations = 1000
# trainer = SimpleTrainer(gt_image=gt_image, num_points=num_points)
# # Assuming we have a function `train` that takes learning rate and number of points as arguments
# save_path = f'results/{optimal_lr}_lr_{training_iterations}_iterations_{num_points}_points.png'
# trainer.train(iterations=training_iterations, lr=optimal_lr, save_path=save_path)

# # Save the training results if save_img is True
# if save_img:
#     plt.figure()
#     plt.plot(range(num_updates), rewards_history[:num_updates], label='Average Reward')
#     plt.xlabel('Points')
#     plt.ylabel('Reward')
#     plt.title('Average Reward over Points')
#     plt.legend()
#     plt.savefig(f'rewards_{training_iterations}_2000_points.png')

