import math
import json
import torch
torch.use_deterministic_algorithms(True,warn_only=True)
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from einops import rearrange
import numpy as np
# from sklearn.metrics.pairwise import cosine_distances
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
# import sys
# sys.path.insert(0,'/home/girmaji08/ACARSaliency/pt_code/SaliencyModel')
# import config as global_config
from matplotlib import pyplot as plt
import argparse
import pdb

import gc
import os

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


def check_gpu_memory():
	return torch.cuda.memory_allocated(), torch.cuda.memory_cached()

def garbage_collect_if_needed(threshold_percent=60):
	allocated, cached = check_gpu_memory()
	utilization_percent = (allocated / cached) * 100
	if utilization_percent > threshold_percent:
		gc.collect()
		torch.cuda.empty_cache()
		# print("Garbage collection performed. GPU memory freed.")
	
class ShuffleBlock(nn.Module):
	def __init__(self, groups):
		super(ShuffleBlock, self).__init__()
		self.groups = groups
	def forward(self, x):
		'''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
		N,C,H,W = x.size()
		g = self.groups
		return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).reshape(N,C,H,W)

# 

class neck(nn.Module):
	def __init__(self):
		super(neck, self).__init__()

		self.adaptive_maxpool = nn.AdaptiveMaxPool3d((8, 16, 29))

		self.conv_slow= nn.Sequential(
				nn.Conv3d(2048, 1024, kernel_size=(1, 1, 1),
						stride=(1,1,1), padding=(0, 0, 0), bias=False),
				nn.ReLU()
				)
		
		self.conv_skip1= nn.Sequential(
				nn.Conv3d(80, 40, kernel_size=(1, 1, 1),
						stride=(1,1,1), padding=(0, 0, 0), bias=False),
				nn.ReLU()
				)	
		self.conv_skip2= nn.Sequential(
				nn.Conv3d(320, 160, kernel_size=(1, 1, 1),
						stride=(1,1,1), padding=(0, 0, 0), bias=False),
				nn.ReLU()
				)	
		self.conv_skip3= nn.Sequential(
				nn.Conv3d(640, 320, kernel_size=(1, 1, 1),
						stride=(1,1,1), padding=(0, 0, 0), bias=False),
				nn.ReLU()
				)	
		self.conv_skip4= nn.Sequential(
				nn.Conv3d(1280, 640, kernel_size=(1, 1, 1),
						stride=(1,1,1), padding=(0, 0, 0), bias=False),
				nn.ReLU()
				)	


	def forward(self, slow_features,fast_features,skip_connections,mask=None):
		
			fast_features = rearrange(fast_features, 'b c t h w -> b (c t) h w')
			fast_features = rearrange(fast_features, 'b (c t) h w -> b c t h w',
							c=256*2, t=int(32/2))

			slow_features = self.conv_slow(slow_features)

			fast_features = self.adaptive_maxpool(fast_features)

			slow_fast_features = torch.cat((slow_features, fast_features), 1)

			skip1,skip2,skip3,skip4 = skip_connections

			skip1 = self.conv_skip1(skip1)
			skip2 = self.conv_skip2(skip2)
			skip3 = self.conv_skip3(skip3)
			skip4 = self.conv_skip4(skip4)


			return slow_fast_features,(skip1,skip2,skip3,skip4)
