import random
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def image_path_to_tensor(image_path: Path):

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]

    return img_tensor 

def dino_preprocess(img_path: str) -> torch.tensor:
    input_image = Image.open(img_path)
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    
    return input_tensor

class LinearSchedule:
    def __init__(self, num_updates: int, start: float, stride: float, frequency: float):
        self.num_updates = num_updates
        self.start = start
        self.stride = stride
        self.frequency = frequency

    def __call__(self, t: int) -> float:
        return max(0, self.start - self.stride * (t // (self.num_updates * self.frequency)))