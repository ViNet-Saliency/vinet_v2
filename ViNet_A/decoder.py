import math
import json
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from einops import rearrange
import numpy as np

import argparse
import pdb

import gc 

class ShuffleBlock(nn.Module):
	def __init__(self, groups):
		super(ShuffleBlock, self).__init__()
		self.groups = groups
	def forward(self, x):
		'''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
		N,C,T,H,W = x.size()
		g = self.groups
		return x.view(N,g,C//g,T,H,W).permute(0,2,1,3,4,5).reshape(N,C,T,H,W)
	

	


class decoder(nn.Module):
	def __init__(self,args,verbose=False):
		super(decoder, self).__init__()
		self.verbose = verbose
		self.use_skip = args.use_skip
		self.decoder_groups = args.decoder_groups

		self.use_channel_shuffle = args.use_channel_shuffle

		print("DECODER GROUPS USED IS : ",self.decoder_groups)

		if self.use_channel_shuffle:
			if max(1,self.decoder_groups) >= 8:
				self.shuffle1 = ShuffleBlock(max(1,self.decoder_groups))
				print("SHUFFLE1 USED")
			if max(1,self.decoder_groups // 2) >= 8:
				self.shuffle2 = ShuffleBlock(max(1,self.decoder_groups // 2))
				print("SHUFFLE2 USED")
			if max(1,self.decoder_groups // 4) >= 8:
				self.shuffle3 = ShuffleBlock(max(1,self.decoder_groups // 4))
				print("SHUFFLE3 USED")
			if max(1,self.decoder_groups // 8) >= 8:
				self.shuffle4 = ShuffleBlock(max(1,self.decoder_groups // 8))
				print("SHUFFLE4 USED")


		self.convtsp1 = nn.Sequential(
			nn.Conv3d(1536, 640, kernel_size=(3, 3, 3),
					  stride=(1,1,1), padding=(1, 1, 1), bias=False, groups=self.decoder_groups),
			nn.ReLU(),
		)
		self.convtsp2 = nn.Sequential(
			nn.Conv3d(1280, 320, kernel_size=(3, 3, 3), stride=(1,1,1), padding=(1, 1, 1), bias=False, groups=max(1,self.decoder_groups // 2)),
			nn.ReLU(),
			nn.Upsample(
				scale_factor=(1, 32/16, 57/29), mode='trilinear')
		)
		self.convtsp3 = nn.Sequential(
			nn.Conv3d(640, 160, kernel_size=(3, 3, 3), stride=(1,1,1), padding=(1, 1, 1), bias=False, groups=max(1,self.decoder_groups // 4)),
			nn.ReLU(),
			nn.Upsample(
				scale_factor=(1, 2, 2), mode='trilinear')
		)
		self.convtsp4 = nn.Sequential(
			nn.Conv3d(320, 40, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, groups=max(1,self.decoder_groups // 4)),
			nn.ReLU()
		)
		self.convtsp5 = nn.Sequential(
			nn.Conv3d(80, 64, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False, groups=max(1,self.decoder_groups // 8)),
			nn.ReLU(),
			nn.Upsample(
				scale_factor=(1, 2, 2), mode='trilinear')
		)
		self.convtsp6 = nn.Sequential(
			nn.Conv3d(64, 32, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False, groups=max(1,self.decoder_groups // 16)),
			nn.ReLU(),
			nn.Upsample(
				scale_factor=(1, 2, 2), mode='trilinear'),

			nn.Conv3d(32, 16, kernel_size=(2, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False, groups=max(1,self.decoder_groups // 32)),
			nn.ReLU(),

			# 4 time dimension
			nn.Conv3d(16, 16, kernel_size=(1, 1, 1),
					  stride=(1, 1, 1), bias=False),
			nn.ReLU(),
			nn.Conv3d(16, 1, kernel_size=(1, 1, 1),
					  stride=1, bias=True),
			nn.Sigmoid(),
		)
		

	def forward(self, high_order_feats,skip_features=None,decoder_mask=None):

		if skip_features and len(skip_features) == 4:
			skip1, skip2, skip3, skip4 = skip_features
		elif skip_features and len(skip_features) == 5:
			skip1, skip2, skip3, skip4, skip5 = skip_features

		
		z = self.convtsp1(high_order_feats)
		if self.verbose:
			print('convtsp1', z.shape)


		# added
		if skip_features:
			z = torch.cat((z, skip4), 1)
		if self.verbose:
			print('cat_convtsp2', z.shape)

		if self.use_channel_shuffle and max(1,self.decoder_groups) >= 8:
			z = self.shuffle1(z)

		z = self.convtsp2(z)
		if self.verbose:
			print('convtsp2', z.shape)

		# added
		if skip_features:
			z = torch.cat((z, skip3), 1)
		if self.verbose:
			print("cat_convtsp3", z.shape)

		if self.use_channel_shuffle and max(1,self.decoder_groups // 2) >= 8:
			z = self.shuffle2(z)

		z = self.convtsp3(z)
		if self.verbose:
			print('convtsp3', z.shape)

		# added
		if skip_features:
			z = torch.cat((z, skip2), 1)
		if self.verbose:
			print("cat_convtsp4", z.shape)

		if self.use_channel_shuffle and max(1,self.decoder_groups // 4) >= 8:
			z = self.shuffle3(z)

		z = self.convtsp4(z)
		if self.verbose:
			print('convtsp4', z.shape)

		# added
		z = torch.cat((z, skip1), 1)
		if self.verbose:
			print("cat_convtsp5", z.shape)

		if self.use_channel_shuffle and max(1,self.decoder_groups // 8) >= 8:
			z = self.shuffle4(z)

		z = self.convtsp5(z)
		if self.verbose:
			print('convtsp5', z.shape)

		z = self.convtsp6(z)
		if self.verbose:
			print('convtsp6', z.shape)

		z = z.view(z.size(0), z.size(3), z.size(4))
		if self.verbose:
			print('output', z.shape)


		return z
	

	