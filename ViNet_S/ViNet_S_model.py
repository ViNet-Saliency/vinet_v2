import torch
from torch import nn
import math
from model_utils import *

# added
# from encoder import *
# from neck import *
# from decoder import *

class Interpolate(nn.Module):
	def __init__(self, mode='bilinear', scale_factor=None, align_corners = True):
		super(Interpolate , self).__init__()
		self.interpolate = nn.functional.interpolate
		self.scale_factor = scale_factor
		self.mode = mode
		self.align_corners = align_corners
	
	def forward(self , x):
		B, C, D, H, W = x.size()
		x = x.view(B * C, D, H, W)
		if self.scale_factor:
			out = self.interpolate(x, scale_factor = self.scale_factor, mode = self.mode, align_corners=self.align_corners)
		else:
			out = self.interpolate(x, size = (2*H, 2*W), mode = self.mode, align_corners=self.align_corners)
		out = out.view(B, C, D, out.size(2), out.size(3))
		return out

class BackBoneS3D(nn.Module):
	def __init__(self, maxpool3d = True):
		super(BackBoneS3D, self).__init__()
		

		self.base1 = nn.Sequential(
			SepConv3d(3, 64, kernel_size=7, stride=2, padding=3),
			BackBone_Maxpool_Base1(),
			BasicConv3d(64, 64, kernel_size=1, stride=1),
			SepConv3d(64, 192, kernel_size=3, stride=1, padding=1),
		)


		self.maxp2 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))
		self.base2 = nn.Sequential(
			Mixed_3b(),
			Mixed_3c(),
		)
		# self.maxp3 = nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
		self.maxp3_1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))
		self.maxp3_2 = nn.MaxPool2d(kernel_size=(3,1), stride=(2,1), padding=(1,0))
		self.base3 = nn.Sequential(
			Mixed_4b(),
			Mixed_4c(),
			Mixed_4d(),
			Mixed_4e(),
			Mixed_4f(),
		)
		# self.maxt4 = nn.MaxPool3d(kernel_size=(2,1,1), stride=(2,1,1), padding=(0,0,0))
		# self.maxp4 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0))
		self.maxt4 = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1), padding=(0,0))
		self.maxp4 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=(0,0))
		self.base4 = nn.Sequential(
			Mixed_5b(),
			Mixed_5c(),
		)

	def forward(self, x):
		# print('input', x.shape)
		y3 = self.base1(x)

		
		n,c,t,h,w = y3.shape
		y = reshape(y3, (y3.size(0), y3.size(1) * y3.size(2), y3.size(3), y3.size(4)))
		y = self.maxp2(y)
		y = reshape(y, (n, c, t, y.size(2), y.size(3)))
		# print('maxp2', y.shape)

		y2 = self.base2(y)
		# print('base2', y2.shape)

		n,c,t,h,w = y2.shape
		y = reshape(y2, (y2.size(0), y2.size(1) * y2.size(2), y2.size(3), y2.size(4)))
		y = self.maxp3_1(y)
		y = reshape(y, (n, c, t, y.size(2), y.size(3)))

		n,c,t,h,w = y.shape
		y = reshape(y, (y.size(0), y.size(1), y.size(2), y.size(3) * y.size(4)))
		y = self.maxp3_2(y)
		y = reshape(y, (n, c, y.size(2), h, w))
		# print('maxp3', y.shape)

		y1 = self.base3(y)
		# print('base3', y1.shape)

		n,c,t,h,w = y1.shape
		y = reshape(y1, (y1.size(0), y1.size(1), y1.size(2), y1.size(3) * y1.size(4)))
		y = self.maxt4(y)
		y = reshape(y, (n, c, y.size(2), h, w))

		n,c,t,h,w = y.shape
		y = reshape(y, (y.size(0), y.size(1) * y.size(2), y.size(3), y.size(4)))
		y = self.maxp4(y)
		y = reshape(y, (n, c, t, y.size(2), y.size(3)))
		# print('maxt4p4', y.shape)

		y0 = self.base4(y)

		# # Save the returning array in a numpy file
		# import numpy as np
		# np.save('backbone.npy', [y0.detach().cpu().numpy(), y1.detach().cpu().numpy(), y2.detach().cpu().numpy(), y3.detach().cpu().numpy()])

		return [y0, y1, y2, y3]

class DecoderConvUpGrouped(nn.Module):
	def __init__(self, root_grouping=True, BiCubic = False):
		super(DecoderConvUpGrouped, self).__init__()
		
		if BiCubic:
			# self.upsampling = Interpolate(scale_factor=(2,2), mode='bicubic' , align_corners=True)
			self.upsampling = Interpolate(mode='bilinear', align_corners=True)
		else:
			self.upsampling = nn.Upsample(scale_factor=(1,2,2), mode='trilinear')

		if root_grouping:
			self.convtsp1 = nn.Sequential(
				nn.Conv3d(1024, 832, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False, groups=32),
				nn.ReLU(),
				self.upsampling
			)
		else:
			self.convtsp1 = nn.Sequential(
				nn.Conv3d(1024, 832, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
				nn.ReLU(),
				self.upsampling
			)
		self.convtsp2 = nn.Sequential(
			nn.Conv3d(832, 480, kernel_size=(3,3,3), stride=(3,1,1), padding=(0,1,1), bias=False, groups=16),
			nn.ReLU(),
			self.upsampling
		)
		self.convtsp3 = nn.Sequential(
			nn.Conv3d(480, 192, kernel_size=(5,3,3), stride=(5,1,1), padding=(0,1,1), bias=False, groups=8),
			nn.ReLU(),
			self.upsampling
		)
		self.convtsp4 = nn.Sequential(
			nn.Conv3d(192, 64, kernel_size=(5,3,3), stride=(5,1,1), padding=(0,1,1), bias=False, groups=4),
			nn.ReLU(),
			self.upsampling, # 112 x 192

			nn.Conv3d(64, 32, kernel_size=(2,3,3), stride=(2,1,1), padding=(0,1,1), bias=False, groups=2),
			nn.ReLU(),
			self.upsampling, # 224 x 384

			# 4 time dimension
			nn.Conv3d(32, 32, kernel_size=(2,1,1), stride=(2,1,1), bias=False),
			nn.ReLU(),            
			nn.Conv3d(32, 1, kernel_size=1, stride=1, bias=True),
			nn.Sigmoid(),
		)

	def forward(self, y0, y1, y2, y3):
		z = self.convtsp1(y0)
		# print('convtsp1', z.shape)

		z = torch.cat((z,y1), 2)
		# print('cat_convtsp1', z.shape)
		
		z = self.convtsp2(z)
		# print('convtsp2', z.shape)

		z = torch.cat((z,y2), 2)
		# print('cat_convtsp2', z.shape)
		
		z = self.convtsp3(z)
		# print('convtsp3', z.shape)

		z = torch.cat((z,y3), 2)
		# print("cat_convtsp3", z.shape)
		
		z = self.convtsp4(z)
		# print('convtsp4', z.shape)
		
		z = z.view(z.size(0), z.size(3), z.size(4))
		# print('output', z.shape)

		return z

class DecoderConvUpDepth(nn.Module):
	def __init__(self, BiCubic = True):
		super(DecoderConvUpDepth, self).__init__()

		if BiCubic:
			self.upsampling = Interpolate(scale_factor=(2,2), mode='bicubic' , align_corners=True)
		else:
			self.upsampling = nn.Upsample(scale_factor=(1,2,2), mode='trilinear')

		self.convtsp1 = nn.Sequential(
			nn.Conv3d(1024, 1024, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False,groups = 1024),
			nn.ReLU(),

			nn.Conv3d(1024, 832, kernel_size=(1,1,1), stride=1, padding=(0,0,0), bias=False),
			nn.ReLU(),

			self.upsampling
		)
		self.convtsp2 = nn.Sequential(
			nn.Conv3d(832, 832, kernel_size=(3,3,3), stride=(3,1,1), padding=(0,1,1), bias=False,groups = 832),
			nn.ReLU(),

			nn.Conv3d(832, 480, kernel_size=(1,1,1), stride=1, padding=(0,0,0), bias=False),
			nn.ReLU(),
			self.upsampling
		)
		self.convtsp3 = nn.Sequential(
			nn.Conv3d(480, 480, kernel_size=(5,3,3), stride=(5,1,1), padding=(0,1,1), bias=False,groups = 480),
			nn.ReLU(),

			nn.Conv3d(480, 192, kernel_size=(1,1,1), stride=1, padding=(0,0,0), bias=False),
			nn.ReLU(),
			self.upsampling
		)
		self.convtsp4 = nn.Sequential(
			nn.Conv3d(192, 192, kernel_size=(5,3,3), stride=(5,1,1), padding=(0,1,1), bias=False,groups = 192),
			nn.ReLU(),
			nn.Conv3d(192, 64, kernel_size=(1,1,1), stride=1, padding=(0,0,0), bias=False),
			nn.ReLU(),
			self.upsampling, # 112 x 192

			nn.Conv3d(64, 64, kernel_size=(2,3,3), stride=(2,1,1), padding=(0,1,1), bias=False,groups = 64),
			nn.ReLU(),
			nn.Conv3d(64, 32, kernel_size=(1,1,1), stride=1, padding=(0,0,0), bias=False),
			nn.ReLU(),
			self.upsampling, # 224 x 384

			# 4 time dimension
			nn.Conv3d(32, 32, kernel_size=(2,1,1), stride=(2,1,1), bias=False),
			nn.ReLU(),            
			nn.Conv3d(32, 1, kernel_size=1, stride=1, bias=True),
			nn.Sigmoid(),
		)

	def forward(self, y0, y1, y2, y3):
		z = self.convtsp1(y0)
		# print('convtsp1', z.shape)

		z = torch.cat((z,y1), 2)
		# print('cat_convtsp1', z.shape)

		z = self.convtsp2(z)
		# print('convtsp2', z.shape)

		z = torch.cat((z,y2), 2)
		# print('cat_convtsp2', z.shape)

		z = self.convtsp3(z)
		# print('convtsp3', z.shape)

		z = torch.cat((z,y3), 2)
		# print("cat_convtsp3", z.shape)

		z = self.convtsp4(z)
		# print('convtsp4', z.shape)

		z = z.view(z.size(0), z.size(3), z.size(4))
		# print('output', z.shape)

		return z




class VideoSaliencyModel(nn.Module):
	def __init__(self, 
				transformer_in_channel=32,
				# nhead=4,
				use_upsample=True,
				num_hier=3,
				num_clips=32,
				grouped_conv=True,
				root_grouping=True,
				depth=False,
				efficientnet=False,
				BiCubic = False,
				maxpool3d = True
			):
		super(VideoSaliencyModel, self).__init__()

		self.backbone = BackBoneS3D(maxpool3d=maxpool3d)
		self.num_hier = num_hier
		if use_upsample:
			if num_hier==0:
				self.decoder = DecoderConvUpNoHier()
			elif num_hier==1:
				self.decoder = DecoderConvUp1Hier()
			elif num_hier==2:
				self.decoder = DecoderConvUp2Hier()
			elif num_hier==3:
				if num_clips==8:
					self.decoder = DecoderConvUp8()
				elif num_clips==16:
					self.decoder = DecoderConvUp16()
				elif num_clips==32:
					if grouped_conv:
						if depth:
							self.decoder = DecoderConvUpDepth(BiCubic = BiCubic)
						else:
							self.decoder = DecoderConvUpGrouped(root_grouping = root_grouping, BiCubic = BiCubic)
					elif efficientnet:
						self.decoder = DecoderConvUpEfficientNet()
					else:
						self.decoder = DecoderConvUp()
				elif num_clips==48:
					self.decoder = DecoderConvUp48()
		else:
			self.decoder = DecoderConvT()

		encoder_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
		decoder_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
		total_params = encoder_params + decoder_params
		print("Total number of parameters in the encoder:", encoder_params)
		print("Total number of parameters in the decoder:", decoder_params)
		print("Total number of parameters in the model (encoder + decoder):", total_params)

		size_in_bytes = total_params * 4  # Each parameter is 4 bytes
		size_in_mb = size_in_bytes / (1024 * 1024)  # Convert bytes to MB
		print("Size of the model: {:.2f} MB".format(size_in_mb))

	def forward(self, x):
		[y0, y1, y2, y3] = self.backbone(x)
		if self.num_hier==0:
			return self.decoder(y0)
		if self.num_hier==1:
			return self.decoder(y0, y1)
		if self.num_hier==2:
			return self.decoder(y0, y1, y2)
		if self.num_hier==3:
			return self.decoder(y0, y1, y2, y3)


