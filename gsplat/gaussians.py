import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import nerfview
from datasets.colmap import Dataset, Parser
from datasets.traj import generate_interpolated_path
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import (
    AppearanceOptModule,
    CameraOptModule,
    knn,
    normalized_quat_to_rotmat,
    rgb_to_sh,
    set_random_seed,
)

from gsplat.rendering import rasterization

class Gaussians(torch.nn.Modules):
	
	def __init__(self):
	  # We don't predefine any parameters to make it general
	  # for example, "scales" in 2dgs is 2-D and in 3dgs is 3-D.
		self.params: torch.nn.ParameterDict = {}
		
	def __len__(self):
		if len(self.params) == 0:
			return 0
		else: 
			first_param = self.params.values()[0]
			return len(first_param)

	def __getitem__(self, key: str) -> torch.nn.Parameter:
		"""Allow dictionary-like access to parameters."""
		if key in self.params:
			return self.params[key]
		else:
            raise KeyError(f"Parameter '{key}' not found in Gaussians object.")
		
	def register_param(self, name: str, param: torch.nn.Parameter):
		""" Allows usage of customized parameters. 

		For example, 
		```
		features = torch.randn(len(gs), 64)
		gs.register_param("features", features)
		```
		"""

		if name in self.params:
			raise Warning(f"Parameter {name} already exists.")
		self.params[name] = param
	
	@classmethod
	def init_from_pcl(cls, points: Tensor, rgbs: Optional[Tensor] = None) -> Gaussians:
		"""Initialize a set of gaussians from a point cloud.

		Args:
			points: the 3D points from the point cloud [N, 3]
			rgbs: the RGB values of the points, normalized to be in the range [0, 1] [N, 3]

		Returns:
			A Gaussians object with the points and rgbs registered as parameters.
		"""

		instance = cls()

		instance.register_param("points", points)

		if rgbs:
			instance.register_param("rgbs", rgbs)
		else:
			instance.register_param("rgbs", torch.rand((points.shape[0], 3)))

		return instance

	@classmethod
	def init_randn(cls, center: Tensor, std: Tensor, init_num_pts: int) -> Gaussians:
		""" 
		Initialize a set of gaussians from a random normal distribution.
		"""
		
		instance = cls()
		points = center + std * torch.randn((init_num_pts, 3))
		rgbs = torch.rand((init_num_pts, 3))

		instance.register_param("points", points)
		instance.register_param("rgbs", rgbs)

		return instance
	
	@classmethod
	def init_rand(cls, center: Tensor, half_edge: Tensor, init_num_pts: int) -> Gaussians:
		""" Initialize a set of gaussians from a random uniform distribution.
		4"""
		instance = cls()
		points = torch.rand((init_num_pts, 3)) * 2 - 1
		points = points * half_edge + center
		rgbs = torch.rand((init_num_pts, 3))

		instance.register_param("points", points)
		instance.register_param("rgbs", rgbs)

		return instance
		