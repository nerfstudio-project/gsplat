import ray
import torch
import torch.nn as nn

import os

cuda_devices = os.getenv("CUDA_VISIBLE_DEVICES")
if cuda_devices is not None:
    print(f"CUDA_VISIBLE_DEVICES={cuda_devices}")
else:
    print("CUDA_VISIBLE_DEVICES is not set.")


# Initialize Ray
ray.init()

# Define your model training function
@ray.remote(num_gpus=1)
def train_model_on_gpu(gpu_id, config):
    device = torch.device(f"cuda:{gpu_id}")
    print(device)
    
    # Define a simple model and move it to the specific GPU
    model = nn.Linear(10, 10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])

    # Simulate a training loop
    for epoch in range(config["epochs"]):
        inputs = torch.randn(32, 10).to(device)
        outputs = model(inputs)
        loss = outputs.sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return f"Finished training on GPU {gpu_id} with config {config}"

# Launch asynchronous training tasks on multiple GPUs
configs = [{"lr": 0.01, "epochs": 5}, {"lr": 0.001, "epochs": 5}]
futures = [train_model_on_gpu.remote(i, config) for i, config in enumerate(configs)]

# Collect results (non-blocking)
results = ray.get(futures)
print(results)
