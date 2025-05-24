import torch
from torch import nn
import math
from model_utils import *

# added
from encoder import *
from neck import *
from decoder import *

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
		# n,c,t,h,w = y3.shape
		# y3 = reshape(x, (y3.size(0), y3.size(1) * y3.size(2), y3.size(3), y3.size(4)))
		# y3 = self.base1_2(y3)
		# y3 = reshape(y3, (n, c, t, y3.size(2), y3.size(3)))
		# y3 = self.base1_3(y3)
		# print('base1', y3.shape)

		# Printing shape
		# y3 = x
		# for layer in self.base1:
		# 	x = layer(x)
		# 	print(x.shape)
		
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
	def __init__(self, root_grouping=True, BiCubic = True):
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
				nhead=4,
				use_upsample=True,
				num_hier=3,
				num_clips=32,
				grouped_conv=False,
				root_grouping=True,
				depth=False,
				efficientnet=False,
				BiCubic = True,
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




# class PositionalEncoding(nn.Module):

# 	def __init__(self, feat_size, dropout=0.1, max_len=4):
# 		super(PositionalEncoding, self).__init__()
# 		self.dropout = nn.Dropout(p=dropout)

# 		pe = torch.zeros(max_len, feat_size)
# 		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
# 		div_term = torch.exp(torch.arange(0, feat_size, 2).float() * (-math.log(10000.0) / feat_size))
# 		pe[:, 0::2] = torch.sin(position * div_term)
# 		pe[:, 1::2] = torch.cos(position * div_term)
# 		pe = pe.unsqueeze(0).transpose(0, 1)
# 		self.register_buffer('pe', pe)

# 	def forward(self, x):
# 		# print(x.shape, self.pe.shape)
# 		x = x + self.pe
# 		# return self.dropout(x)
# 		return x

# class Transformer(nn.Module):
# 	def __init__(self, feat_size, hidden_size=256, nhead=4, num_encoder_layers=3, max_len=4, num_decoder_layers=-1, num_queries=4, spatial_dim=-1):
# 		super(Transformer, self).__init__()
# 		self.pos_encoder = PositionalEncoding(feat_size, max_len=max_len)
# 		encoder_layers = nn.TransformerEncoderLayer(feat_size, nhead, hidden_size)
		
# 		self.spatial_dim = spatial_dim
# 		if self.spatial_dim!=-1:
# 			transformer_encoder_spatial_layers = nn.TransformerEncoderLayer(spatial_dim, nhead, hidden_size)
# 			self.transformer_encoder_spatial = nn.TransformerEncoder(transformer_encoder_spatial_layers, num_encoder_layers)
		
# 		self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
# 		self.use_decoder = (num_decoder_layers != -1)
		
# 		if self.use_decoder:
# 			decoder_layers = nn.TransformerDecoderLayer(hidden_size, nhead, hidden_size)
# 			self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_decoder_layers, norm=nn.LayerNorm(hidden_size))
# 			self.tgt_pos = nn.Embedding(num_queries, hidden_size).weight
# 			assert self.tgt_pos.requires_grad == True

# 	def forward(self, embeddings, idx):
# 		''' embeddings: CxBxCh*H*W '''
# 		# print(embeddings.shape)
# 		batch_size = embeddings.size(1)

# 		if self.spatial_dim!=-1:
# 			embeddings = embeddings.permute((2,1,0))
# 			embeddings = self.transformer_encoder_spatial(embeddings)
# 			embeddings = embeddings.permute((2,1,0))

# 		x = self.pos_encoder(embeddings)
# 		x = self.transformer_encoder(x)
# 		if self.use_decoder:
# 			if idx!=-1:
# 				tgt_pos = self.tgt_pos[idx].unsqueeze(0)
# 				# print(tgt_pos.size())
# 				tgt_pos = tgt_pos.unsqueeze(1).repeat(1,batch_size,1)
# 			else:
# 				tgt_pos = self.tgt_pos.unsqueeze(1).repeat(1,batch_size,1)
# 			tgt = torch.zeros_like(tgt_pos)
# 			x = self.transformer_decoder(tgt + tgt_pos, x)
# 		return x

# class EEAA(nn.Module):
# 	def __init__(self,
# 	      		args,use_upsample=True 
				
# 			):
# 		super(EEAA, self).__init__()

# 		self.use_skip = bool(args.use_skip)

# 		self.backbone = slowfast50(self.use_skip)
		
# 		self.neck = neck()

# 		self.decoder = decoder(args)

# 		self.decoder_groups = args.decoder_groups

# 		encoder_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
# 		decoder_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
# 		neck_params = sum(p.numel() for p in self.neck.parameters() if p.requires_grad)
# 		total_params = encoder_params + decoder_params + neck_params
# 		print("Total number of parameters in the encoder:", encoder_params)
# 		print("Total number of parameters in the decoder:", decoder_params)
# 		print("Total number of parameters in the neck:", neck_params)
# 		print("Total number of parameters in the model (encoder + decoder + neck):", total_params)

# 		size_in_bytes = total_params * 4  # Each parameter is 4 bytes
# 		size_in_mb = size_in_bytes / (1024 * 1024)  # Convert bytes to MB
# 		print("Size of the model: {:.2f} MB".format(size_in_mb))

# 	def forward(self, x):

# 		y = self.backbone(x)

# 		slow_features,fast_features = y[:2]

# 		if self.use_skip:

# 			skip_connections = y[2:]

# 			slow_fast_features,skip_connections = self.neck(slow_features,fast_features,skip_connections)
# 			return self.decoder(slow_fast_features, skip_connections)

# 		return self.decoder(slow_features,fast_features)




# class VideoAudioSaliencyFusionModel(nn.Module):
# 	def __init__(self, 
# 				use_transformer=True,
# 				transformer_in_channel=512,
# 				num_encoder_layers=3,
# 				nhead=4,
# 				use_upsample=True,
# 				num_hier=3,
# 				num_clips=32
# 			):
# 		super(VideoAudioSaliencyFusionModel, self).__init__()
# 		self.use_transformer = use_transformer
# 		self.visual_model = VideoSaliencyModel(
# 				transformer_in_channel=transformer_in_channel,
# 				nhead=nhead,
# 				use_upsample=use_upsample,
# 				num_hier=num_hier,
# 				num_clips=num_clips
# 		)

# 		self.conv_in_1x1 = nn.Conv3d(in_channels=1024, out_channels=transformer_in_channel, kernel_size=1, stride=1, bias=True)
# 		self.transformer =  Transformer(
# 								transformer_in_channel, 
# 								hidden_size=transformer_in_channel, 
# 								nhead=nhead,
# 								num_encoder_layers=num_encoder_layers,
# 								num_decoder_layers=-1,
# 								max_len=4*7*12+3,
# 							)

# 		self.audionet = SoundNet()
# 		self.audio_conv_1x1 = nn.Conv2d(in_channels=1024, out_channels=transformer_in_channel, kernel_size=1, stride=1, bias=True)
# 		self.audionet.load_state_dict(torch.load('./soundnet8_final.pth'))
# 		print("Loaded SoundNet Weights")
# 		for param in self.audionet.parameters():
# 			param.requires_grad = True

# 		self.maxpool = nn.MaxPool3d((4,1,1),stride=(2,1,2),padding=(0,0,0))
# 		self.bilinear = nn.Bilinear(42, 3, 4*7*12)

# 	def forward(self, x, audio):
# 		audio = self.audionet(audio)
# 		# print(audio.size())
# 		audio = self.audio_conv_1x1(audio)
# 		audio = audio.flatten(2)
# 		# print("audio", audio.shape)

# 		[y0, y1, y2, y3] = self.visual_model.backbone(x)
# 		y0 = self.conv_in_1x1(y0)
# 		y0 = y0.flatten(2)
# 		# print("video", y0.shape)

# 		fused_out = torch.cat((y0, audio), 2)
# 		# print("fused_out", fused_out.size())
# 		fused_out = fused_out.permute((2,0,1))
# 		fused_out = self.transformer(fused_out, -1)

# 		fused_out = fused_out.permute((1, 2, 0))

# 		video_features = fused_out[..., :4*7*12]
# 		audio_features = fused_out[..., 4*7*12:]

# 		# print("separate", video_features.shape, audio_features.shape)

# 		video_features = video_features.view(video_features.size(0), video_features.size(1), 4, 7, 12)
# 		audio_features = torch.mean(audio_features, dim=2)

# 		audio_features = audio_features.view(audio_features.size(0), audio_features.size(1), 1,1,1).repeat(1,1,4,7,12)

# 		final_out = torch.cat((video_features, audio_features), 1)

# 		# print(final_out.size())	

# 		return self.visual_model.decoder(final_out, y1, y2, y3)

# class VideoAudioSaliencyModel(nn.Module):
# 	def __init__(self, 
# 				use_transformer=False,
# 				transformer_in_channel=32,
# 				num_encoder_layers=3,
# 				nhead=4,
# 				use_upsample=True,
# 				num_hier=3,
# 				num_clips=32
# 			):
# 		super(VideoAudioSaliencyModel, self).__init__()
# 		self.use_transformer = use_transformer
# 		self.visual_model = VideoSaliencyModel(
# 				transformer_in_channel=transformer_in_channel,
# 				nhead=nhead,
# 				use_upsample=use_upsample,
# 				num_hier=num_hier,
# 				num_clips=num_clips
# 		)

# 		if self.use_transformer:
# 			self.conv_in_1x1 = nn.Conv3d(in_channels=1024, out_channels=transformer_in_channel, kernel_size=1, stride=1, bias=True)
# 			self.conv_out_1x1 = nn.Conv3d(in_channels=32, out_channels=1024, kernel_size=1, stride=1, bias=True)
# 			self.transformer =  Transformer(
# 									4*7*12, 
# 									hidden_size=4*7*12, 
# 									nhead=nhead,
# 									num_encoder_layers=num_encoder_layers,
# 									num_decoder_layers=-1,
# 									max_len=transformer_in_channel,
# 								)

# 		self.audionet = SoundNet()
# 		self.audionet.load_state_dict(torch.load('./soundnet8_final.pth'))
# 		print("Loaded SoundNet Weights")
# 		for param in self.audionet.parameters():
# 			param.requires_grad = True

# 		self.maxpool = nn.MaxPool3d((4,1,1),stride=(2,1,2),padding=(0,0,0))
# 		self.bilinear = nn.Bilinear(42, 3, 4*7*12)

# 	def forward(self, x, audio):
# 		audio = self.audionet(audio)
# 		[y0, y1, y2, y3] = self.visual_model.backbone(x)
# 		y0 = self.maxpool(y0)
# 		fused_out = self.bilinear(y0.flatten(2), audio.flatten(2))
# 		fused_out = fused_out.view(fused_out.size(0), fused_out.size(1), 4, 7, 12)

# 		if self.use_transformer:
# 			fused_out = self.conv_in_1x1(fused_out)
# 			fused_out = fused_out.flatten(2)
# 			fused_out = fused_out.permute((1,0,2))
# 			# print("fused_out", fused_out.shape)
# 			fused_out = self.transformer(fused_out, -1)
# 			fused_out = fused_out.permute((1,0,2))
# 			fused_out = fused_out.view(fused_out.size(0), fused_out.size(1), 4, 7, 12)
# 			fused_out = self.conv_out_1x1(fused_out)

# 		return self.visual_model.decoder(fused_out, y1, y2, y3)



# class DecoderConvUp(nn.Module):
# 	def __init__(self, BiCubic = True):
# 		super(DecoderConvUp, self).__init__()
# 		if BiCubic:
# 			self.upsampling = Interpolate(scale_factor=(2,2), mode='bicubic' , align_corners=True)
# 		else:
# 			self.upsampling = nn.Upsample(scale_factor=(1,2,2), mode='trilinear')
		
# 		self.convtsp1 = nn.Sequential(
# 			nn.Conv3d(1024, 832, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling
# 		)
# 		self.convtsp2 = nn.Sequential(
# 			nn.Conv3d(832, 480, kernel_size=(3,3,3), stride=(3,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling
# 		)
# 		self.convtsp3 = nn.Sequential(
# 			nn.Conv3d(480, 192, kernel_size=(5,3,3), stride=(5,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling
# 		)
# 		self.convtsp4 = nn.Sequential(
# 			nn.Conv3d(192, 64, kernel_size=(5,3,3), stride=(5,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling, # 112 x 192

# 			nn.Conv3d(64, 32, kernel_size=(2,3,3), stride=(2,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling, # 224 x 384

# 			# 4 time dimension
# 			nn.Conv3d(32, 32, kernel_size=(2,1,1), stride=(2,1,1), bias=False),
# 			nn.ReLU(),            
# 			nn.Conv3d(32, 1, kernel_size=1, stride=1, bias=True),
# 			nn.Sigmoid(),
# 		)

# 	def forward(self, y0, y1, y2, y3):
# 		z = self.convtsp1(y0)
# 		# print('convtsp1', z.shape)

# 		z = torch.cat((z,y1), 2)
# 		# print('cat_convtsp1', z.shape)
		
# 		z = self.convtsp2(z)
# 		# print('convtsp2', z.shape)

# 		z = torch.cat((z,y2), 2)
# 		# print('cat_convtsp2', z.shape)
		
# 		z = self.convtsp3(z)
# 		# print('convtsp3', z.shape)

# 		z = torch.cat((z,y3), 2)
# 		# print("cat_convtsp3", z.shape)
		
# 		z = self.convtsp4(z)
# 		# print('convtsp4', z.shape)
		
# 		z = z.view(z.size(0), z.size(3), z.size(4))
# 		# print('output', z.shape)

# 		return z
	


# TODO: Add Efficient Net equivalent of the decoder
# class DecoderConvUpEfficientNet(nn.Module ):
# 	def __init__(self, BiCubic = True):
# 		super(DecoderConvUpEfficientNet, self).__init__()

# 		if BiCubic:
# 			self.upsampling = Interpolate(scale_factor=(2,2), mode='bicubic' , align_corners=True)
# 		else:
# 			self.upsampling = nn.Upsample(scale_factor=(1,2,2), mode='trilinear')

# 		self.convtsp1 = nn.Sequential(
# 			nn.Conv3d(1024, 832, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling
# 		)
# 		self.convtsp2 = nn.Sequential(
# 			nn.Conv3d(832, 480, kernel_size=(3,3,3), stride=(3,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling
# 		)
# 		self.convtsp3 = nn.Sequential(
# 			nn.Conv3d(480, 192, kernel_size=(5,3,3), stride=(5,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling
# 		)
# 		self.convtsp4 = nn.Sequential(
# 			nn.Conv3d(192, 64, kernel_size=(5,3,3), stride=(5,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling, # 112 x 192

# 			nn.Conv3d(64, 32, kernel_size=(2,3,3), stride=(2,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling, # 224 x 384

# 			# 4 time dimension
# 			nn.Conv3d(32, 32, kernel_size=(2,1,1), stride=(2,1,1), bias=False),
# 			nn.ReLU(),            
# 			nn.Conv3d(32, 1, kernel_size=1, stride=1, bias=True),
# 			nn.Sigmoid(),
# 		)

# 	def forward(self, y0, y1, y2, y3):
# 		z = self.convtsp1(y0)
# 		# print('convtsp1', z.shape)

# 		z = torch.cat((z,y1), 2)
# 		# print('cat_convtsp1', z.shape)
		
# 		z = self.convtsp2(z)
# 		# print('convtsp2', z.shape)

# 		z = torch.cat((z,y2), 2)
# 		# print('cat_convtsp2', z.shape)
		
# 		z = self.convtsp3(z)
# 		# print('convtsp3', z.shape)

# 		z = torch.cat((z,y3), 2)
# 		# print("cat_convtsp3", z.shape)
		
# 		z = self.convtsp4(z)
# 		# print('convtsp4', z.shape)
		
# 		z = z.view(z.size(0), z.size(3), z.size(4))
# 		# print('output', z.shape)

# 		return z

# class DecoderConvUp16(nn.Module):
# 	def __init__(self):
# 		super(DecoderConvUp16, self).__init__()
# 		self.upsampling = nn.Upsample(scale_factor=(1,2,2), mode='trilinear')
# 		self.convtsp1 = nn.Sequential(
# 			nn.Conv3d(1024, 832, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling
# 		)
# 		self.convtsp2 = nn.Sequential(
# 			nn.Conv3d(832, 480, kernel_size=(3,3,3), stride=(3,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling
# 		)
# 		self.convtsp3 = nn.Sequential(
# 			nn.Conv3d(480, 192, kernel_size=(5,3,3), stride=(5,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling
# 		)
# 		self.convtsp4 = nn.Sequential(
# 			nn.Conv3d(192, 64, kernel_size=(5,3,3), stride=(5,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling, # 112 x 192

# 			nn.Conv3d(64, 32, kernel_size=(2,3,3), stride=(2,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling, # 224 x 384

# 			# 4 time dimension
# 			nn.Conv3d(32, 1, kernel_size=(1,1,1), stride=(1,1,1), bias=True),
# 			# nn.ReLU(),            
# 			# nn.Conv3d(32, 1, kernel_size=1, stride=1, bias=True),
# 			nn.Sigmoid(),
# 		)

# 	def forward(self, y0, y1, y2, y3):
# 		z = self.convtsp1(y0)
# 		# print('convtsp1', z.shape)

# 		z = torch.cat((z,y1), 2)
# 		# print('cat_convtsp1', z.shape)
		
# 		z = self.convtsp2(z)
# 		# print('convtsp2', z.shape)

# 		z = torch.cat((z,y2), 2)
# 		# print('cat_convtsp2', z.shape)
		
# 		z = self.convtsp3(z)
# 		# print('convtsp3', z.shape)

# 		z = torch.cat((z,y3), 2)
# 		# print("cat_convtsp3", z.shape)
		
# 		z = self.convtsp4(z)
# 		# print('convtsp4', z.shape)
		
# 		z = z.view(z.size(0), z.size(3), z.size(4))
# 		# print('output', z.shape)

# 		return z

# class DecoderConvUp8(nn.Module):
# 	def __init__(self):
# 		super(DecoderConvUp8, self).__init__()
# 		self.upsampling = nn.Upsample(scale_factor=(1,2,2), mode='trilinear')
# 		self.convtsp1 = nn.Sequential(
# 			nn.Conv3d(1024, 832, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling
# 		)
# 		self.convtsp2 = nn.Sequential(
# 			nn.Conv3d(832, 480, kernel_size=(3,3,3), stride=(3,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling
# 		)
# 		self.convtsp3 = nn.Sequential(
# 			nn.Conv3d(480, 192, kernel_size=(5,3,3), stride=(5,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling
# 		)
# 		self.convtsp4 = nn.Sequential(
# 			nn.Conv3d(192, 64, kernel_size=(5,3,3), stride=(5,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling, # 112 x 192

# 			nn.Conv3d(64, 32, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling, # 224 x 384

# 			# 4 time dimension
# 			nn.Conv3d(32, 1, kernel_size=(1,1,1), stride=(1,1,1), bias=True),
# 			# nn.ReLU(),            
# 			# nn.Conv3d(32, 1, kernel_size=1, stride=1, bias=True),
# 			nn.Sigmoid(),
# 		)

# 	def forward(self, y0, y1, y2, y3):
# 		z = self.convtsp1(y0)
# 		# print('convtsp1', z.shape)

# 		z = torch.cat((z,y1), 2)
# 		# print('cat_convtsp1', z.shape)
		
# 		z = self.convtsp2(z)
# 		# print('convtsp2', z.shape)

# 		z = torch.cat((z,y2), 2)
# 		# print('cat_convtsp2', z.shape)
		
# 		z = self.convtsp3(z)
# 		# print('convtsp3', z.shape)

# 		z = torch.cat((z,y3), 2)
# 		# print("cat_convtsp3", z.shape)
		
# 		z = self.convtsp4(z)
# 		# print('convtsp4', z.shape)
		
# 		z = z.view(z.size(0), z.size(3), z.size(4))
# 		# print('output', z.shape)

# 		return z

# class DecoderConvUp48(nn.Module):
# 	def __init__(self):
# 		super(DecoderConvUp48, self).__init__()
# 		self.upsampling = nn.Upsample(scale_factor=(1,2,2), mode='trilinear')
# 		self.convtsp1 = nn.Sequential(
# 			nn.Conv3d(1024, 832, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling
# 		)
# 		self.convtsp2 = nn.Sequential(
# 			nn.Conv3d(832, 480, kernel_size=(3,3,3), stride=(3,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling
# 		)
# 		self.convtsp3 = nn.Sequential(
# 			nn.Conv3d(480, 192, kernel_size=(5,3,3), stride=(5,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling
# 		)
# 		self.convtsp4 = nn.Sequential(
# 			nn.Conv3d(192, 64, kernel_size=(5,3,3), stride=(5,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling, # 112 x 192

# 			nn.Conv3d(64, 32, kernel_size=(2,3,3), stride=(2,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling, # 224 x 384

# 			# 4 time dimension
# 			nn.Conv3d(32, 32, kernel_size=(3,1,1), stride=(3,1,1), bias=True),
# 			nn.ReLU(),            
# 			nn.Conv3d(32, 1, kernel_size=1, stride=1, bias=True),
# 			nn.Sigmoid(),
# 		)

# 	def forward(self, y0, y1, y2, y3):
# 		# print(y0.shape)
# 		z = self.convtsp1(y0)
# 		# print('convtsp1', z.shape)

# 		z = torch.cat((z,y1), 2)
# 		# print('cat_convtsp1', z.shape)
		
# 		z = self.convtsp2(z)
# 		# print('convtsp2', z.shape)

# 		z = torch.cat((z,y2), 2)
# 		# print('cat_convtsp2', z.shape)
		
# 		z = self.convtsp3(z)
# 		# print('convtsp3', z.shape)

# 		z = torch.cat((z,y3), 2)
# 		# print("cat_convtsp3", z.shape)
		
# 		z = self.convtsp4(z)
# 		# print('convtsp4', z.shape)
		
# 		z = z.view(z.size(0), z.size(3), z.size(4))
# 		# print('output', z.shape)

# 		return z


# class DecoderConvUpNoHier(nn.Module):
# 	def __init__(self):
# 		super(DecoderConvUpNoHier, self).__init__()
# 		self.upsampling = nn.Upsample(scale_factor=(1,2,2), mode='trilinear')
# 		self.convtsp1 = nn.Sequential(
# 			nn.Conv3d(1024, 832, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling
# 		)
# 		self.convtsp2 = nn.Sequential(
# 			nn.Conv3d(832, 480, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling
# 		)
# 		self.convtsp3 = nn.Sequential(
# 			nn.Conv3d(480, 192, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling
# 		)
# 		self.convtsp4 = nn.Sequential(
# 			nn.Conv3d(192, 64, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling, # 112 x 192

# 			nn.Conv3d(64, 32, kernel_size=(2,3,3), stride=(2,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling, # 224 x 384

# 			# 4 time dimension
# 			nn.Conv3d(32, 32, kernel_size=(2,1,1), stride=(2,1,1), bias=False),
# 			nn.ReLU(),            
# 			nn.Conv3d(32, 1, kernel_size=1, stride=1, bias=True),
# 			nn.Sigmoid(),
# 		)

# 	def forward(self, y0):
		
# 		z = self.convtsp1(y0)
# 		# print('convtsp1', z.shape)

# 		# z = torch.cat((z,y1), 2)
# 		# print('cat_convtsp1', z.shape)
		
# 		z = self.convtsp2(z)
# 		# print('convtsp2', z.shape)

# 		# z = torch.cat((z,y2), 2)
# 		# print('cat_convtsp2', z.shape)
		
# 		z = self.convtsp3(z)
# 		# print('convtsp3', z.shape)

# 		# z = torch.cat((z,y3), 2)
# 		# print("cat_convtsp3", z.shape)
		
# 		z = self.convtsp4(z)
# 		# print('convtsp4', z.shape)
		
# 		z = z.view(z.size(0), z.size(3), z.size(4))
# 		# print('output', z.shape)

# 		return z

# class DecoderConvUp1Hier(nn.Module):
# 	def __init__(self):
# 		super(DecoderConvUp1Hier, self).__init__()
# 		self.upsampling = nn.Upsample(scale_factor=(1,2,2), mode='trilinear')
# 		self.convtsp1 = nn.Sequential(
# 			nn.Conv3d(1024, 832, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling
# 		)
# 		self.convtsp2 = nn.Sequential(
# 			nn.Conv3d(832, 480, kernel_size=(3,3,3), stride=(3,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling
# 		)
# 		self.convtsp3 = nn.Sequential(
# 			nn.Conv3d(480, 192, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling
# 		)
# 		self.convtsp4 = nn.Sequential(
# 			nn.Conv3d(192, 64, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling, # 112 x 192

# 			nn.Conv3d(64, 32, kernel_size=(2,3,3), stride=(2,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling, # 224 x 384

# 			# 4 time dimension
# 			nn.Conv3d(32, 32, kernel_size=(2,1,1), stride=(2,1,1), bias=False),
# 			nn.ReLU(),            
# 			nn.Conv3d(32, 1, kernel_size=1, stride=1, bias=True),
# 			nn.Sigmoid(),
# 		)

# 	def forward(self, y0, y1):
		
# 		z = self.convtsp1(y0)
# 		# print('convtsp1', z.shape, y1.shape)

# 		z = torch.cat((z,y1), 2)
# 		# print('cat_convtsp1', z.shape)
		
# 		z = self.convtsp2(z)
# 		# print('convtsp2', z.shape)

# 		# z = torch.cat((z,y2), 2)
# 		# print('cat_convtsp2', z.shape)
		
# 		z = self.convtsp3(z)
# 		# print('convtsp3', z.shape)

# 		# z = torch.cat((z,y3), 2)
# 		# print("cat_convtsp3", z.shape)
		
# 		z = self.convtsp4(z)
# 		# print('convtsp4', z.shape)
		
# 		z = z.view(z.size(0), z.size(3), z.size(4))
# 		# print('output', z.shape)

# 		return z

# class DecoderConvUp2Hier(nn.Module):
# 	def __init__(self):
# 		super(DecoderConvUp2Hier, self).__init__()
# 		self.upsampling = nn.Upsample(scale_factor=(1,2,2), mode='trilinear')
# 		self.convtsp1 = nn.Sequential(
# 			nn.Conv3d(1024, 832, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling
# 		)
# 		self.convtsp2 = nn.Sequential(
# 			nn.Conv3d(832, 480, kernel_size=(3,3,3), stride=(3,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling
# 		)
# 		self.convtsp3 = nn.Sequential(
# 			nn.Conv3d(480, 192, kernel_size=(5,3,3), stride=(5,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling
# 		)
# 		self.convtsp4 = nn.Sequential(
# 			nn.Conv3d(192, 64, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling, # 112 x 192

# 			nn.Conv3d(64, 32, kernel_size=(2,3,3), stride=(2,1,1), padding=(0,1,1), bias=False),
# 			nn.ReLU(),
# 			self.upsampling, # 224 x 384

# 			# 4 time dimension
# 			nn.Conv3d(32, 32, kernel_size=(2,1,1), stride=(2,1,1), bias=False),
# 			nn.ReLU(),            
# 			nn.Conv3d(32, 1, kernel_size=1, stride=1, bias=True),
# 			nn.Sigmoid(),
# 		)

# 	def forward(self, y0, y1, y2):
		
# 		z = self.convtsp1(y0)
# 		# print('convtsp1', z.shape)

# 		z = torch.cat((z,y1), 2)
# 		# print('cat_convtsp1', z.shape)
		
# 		z = self.convtsp2(z)
# 		# print('convtsp2', z.shape)

# 		z = torch.cat((z,y2), 2)
# 		# print('cat_convtsp2', z.shape)
		
# 		z = self.convtsp3(z)
# 		# print('convtsp3', z.shape)

# 		# z = torch.cat((z,y3), 2)
# 		# print("cat_convtsp3", z.shape)
		
# 		z = self.convtsp4(z)
# 		# print('convtsp4', z.shape)
		
# 		z = z.view(z.size(0), z.size(3), z.size(4))
# 		# print('output', z.shape)

# 		return z




# class SoundNet(nn.Module):
#     def __init__(self):
#         super(SoundNet, self).__init__()

#         self.conv1 = nn.Conv2d(1, 16, kernel_size=(64, 1), stride=(2, 1),
#                                padding=(32, 0))
#         self.batchnorm1 = nn.BatchNorm2d(16, eps=1e-5, momentum=0.1)
#         self.relu1 = nn.ReLU(True)
#         self.maxpool1 = nn.MaxPool2d((8, 1), stride=(8, 1))

#         self.conv2 = nn.Conv2d(16, 32, kernel_size=(32, 1), stride=(2, 1),
#                                padding=(16, 0))
#         self.batchnorm2 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1)
#         self.relu2 = nn.ReLU(True)
#         self.maxpool2 = nn.MaxPool2d((8, 1), stride=(8, 1))

#         self.conv3 = nn.Conv2d(32, 64, kernel_size=(16, 1), stride=(2, 1),
#                                padding=(8, 0))
#         self.batchnorm3 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
#         self.relu3 = nn.ReLU(True)

#         self.conv4 = nn.Conv2d(64, 128, kernel_size=(8, 1), stride=(2, 1),
#                                padding=(4, 0))
#         self.batchnorm4 = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)
#         self.relu4 = nn.ReLU(True)

#         self.conv5 = nn.Conv2d(128, 256, kernel_size=(4, 1), stride=(2, 1),
#                                padding=(2, 0))
#         self.batchnorm5 = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
#         self.relu5 = nn.ReLU(True)
#         self.maxpool5 = nn.MaxPool2d((4, 1), stride=(4, 1))

#         self.conv6 = nn.Conv2d(256, 512, kernel_size=(4, 1), stride=(2, 1),
#                                padding=(2, 0))
#         self.batchnorm6 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
#         self.relu6 = nn.ReLU(True)

#         self.conv7 = nn.Conv2d(512, 1024, kernel_size=(4, 1), stride=(2, 1),
#                                padding=(2, 0))
#         self.batchnorm7 = nn.BatchNorm2d(1024, eps=1e-5, momentum=0.1)
#         self.relu7 = nn.ReLU(True)

#         self.conv8_objs = nn.Conv2d(1024, 1000, kernel_size=(8, 1),
#                                     stride=(2, 1))
#         self.conv8_scns = nn.Conv2d(1024, 401, kernel_size=(8, 1),
#                                     stride=(2, 1))

#     def forward(self, waveform):
#         x = self.conv1(waveform)
#         x = self.batchnorm1(x)
#         x = self.relu1(x)
#         x = self.maxpool1(x)

#         x = self.conv2(x)
#         x = self.batchnorm2(x)
#         x = self.relu2(x)
#         x = self.maxpool2(x)

#         x = self.conv3(x)
#         x = self.batchnorm3(x)
#         x = self.relu3(x)

#         x = self.conv4(x)
#         x = self.batchnorm4(x)
#         x = self.relu4(x)

#         x = self.conv5(x)
#         x = self.batchnorm5(x)
#         x = self.relu5(x)
#         x = self.maxpool5(x)

#         x = self.conv6(x)
#         x = self.batchnorm6(x)
#         x = self.relu6(x)

#         x = self.conv7(x)
#         x = self.batchnorm7(x)
#         x = self.relu7(x)

#         return x