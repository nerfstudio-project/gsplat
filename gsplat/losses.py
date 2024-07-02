import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchmetrics.functional import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

class L1(nn.Module):
    def __init__(self, device: torch.device):
        self.l1 = F.l1_loss
        self.device = device
    
    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        preds = preds.to(self.device)
        target = target.to(self.device)
        return self.l1(preds, target)
        
class SSIM(nn.Module):
    def __init__(self, device: torch.device, data_range: float = 1.0):
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)
        
    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        return self.ssim(preds, target)

class PSNR(nn.Module):
    def __init__(self, device: torch.device, data_range: float = 1.0):
        self.psnr = PeakSignalNoiseRatio(data_range=data_range).to(device)
    
    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        return self.psnr(preds, target)
    
class LPIPS(nn.Module):
    def __init__(self, device: torch.device, normalize: bool = True):
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=normalize).to(device)
    
    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        return self.lpips(preds, target)
