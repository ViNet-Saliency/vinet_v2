import torch
from torch import nn
import math
from model_utils import *
# from block import fusions
from collections import OrderedDict
from einops import rearrange
import copy
import pdb

from encoder import *

import decoder

import sys
sys.path.insert(0,'/home/girmaji08/EEAA/SaliencyModel')

import neck
import yaml
from easydict import EasyDict
import torch
import pickle
#added
BN = nn.BatchNorm3d
# from action_model import *

class ViNet_A(nn.Module):
	def __init__(self,
	      		args,use_upsample=True 
				
			):
		super(ViNet_A, self).__init__()


		self.use_skip = bool(args.use_skip)

		self.backbone = slowfast50(self.use_skip)
		
		self.decoder_groups = args.decoder_groups

		self.neck_name = args.neck_name



		if self.neck_name == 'neck':
			
			self.neck = neck.neck()

# 112996279
		if use_upsample:

			if self.neck_name == 'neck':
				self.decoder = decoder.decoder(args)



		encoder_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
		decoder_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
		neck_params = sum(p.numel() for p in self.neck.parameters() if p.requires_grad)
		total_params = encoder_params + decoder_params + neck_params
		print("Total number of parameters in the encoder:", encoder_params)
		print("Total number of parameters in the decoder:", decoder_params)
		print("Total number of parameters in the neck:", neck_params)
		print("Total number of parameters in the model (encoder + decoder + neck):", total_params)

		size_in_bytes = total_params * 4  # Each parameter is 4 bytes
		size_in_mb = size_in_bytes / (1024 * 1024)  # Convert bytes to MB
		print("Size of the model: {:.2f} MB".format(size_in_mb))

	def forward(self, x, temp_sal_model = None,temp_sal=None,mask=None):


		y = self.backbone(x)

		slow_features,fast_features = y[:2]

		if self.use_skip:

			skip_connections = y[2:]

			if self.neck_name == 'neck':

				slow_fast_features,skip_connections = self.neck(slow_features,fast_features,skip_connections,mask=mask)

				return self.decoder(slow_fast_features, skip_connections,decoder_mask = mask)


		return self.decoder(slow_features,fast_features)
	