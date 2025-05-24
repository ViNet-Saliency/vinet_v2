
import scipy.io as sio 
import os
from os.path import join
import csv
import cv2, copy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
# import torchaudio
import sys
from scipy.io import loadmat, wavfile
import json

import pdb
import pickle
import random


import torch.utils.data as data

from utils_sal import get_frame_indices

class DHF1K_Dataset(Dataset):
	def __init__(self, args,mode='train'):
		''' mode: train, val, test'''
		if mode == 'train':
			self.path_data = join(args.videos_root_path,'annotation') if os.path.exists(join(args.videos_root_path,'annotation')) else join(args.videos_root_path,'train')
		if mode == 'val':
			self.path_data = join(args.videos_root_path,'val')
		elif mode == 'test' :
			self.path_data = join(args.videos_root_path,'val')
			
		self.len_snippet = args.len_snippet
		self.mode = mode

		# added
		self.use_skip = args.use_skip
		self.img_transform = transforms.Compose([
			transforms.Resize((256, 456) if self.use_skip else (224,448)),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.450, 0.450, 0.450],#[0.485, 0.456, 0.406],
				[0.225, 0.225, 0.225]#[0.229, 0.224, 0.225]
			)
		])

		if self.mode == "train":
			self.video_names = os.listdir(self.path_data)


			print("Number of videos in training set: ", len(self.video_names))
			

		elif self.mode=="val":
			self.list_num_frame = []

			self.video_names = os.listdir(self.path_data)

			for v in self.video_names:

				if len(os.listdir(os.path.join(self.path_data ,v,'images')))<= int(self.len_snippet):
					self.list_num_frame.append((v, len(os.listdir(os.path.join(self.path_data ,v,'images'))) // 2))
				else:
					for i in range(0, len(os.listdir(os.path.join(self.path_data ,v,'images'))) - self.len_snippet, int(0.5*self.len_snippet)):
						self.list_num_frame.append((v, i))
			print("Number of val video names are : ",len(self.video_names))
			print("Number of val samples are : ",len(self.list_num_frame))

		else:
			self.list_num_frame = []
			self.video_names = sorted(os.listdir(self.path_data))
			
			if args.video_names_list != '':

				self.video_names = [i for i in self.video_names if i in args.video_names_list]

			print("test video names are : ",self.video_names)


			for video_name in self.video_names:
				num_frames = len(os.listdir(os.path.join(self.path_data, video_name, 'images')))
				for mid_frame in range(0,num_frames):
					self.list_num_frame.append((video_name,num_frames,mid_frame))

			print("Number of test video names are : ",len(self.video_names))
			print("Number of test samples are : ",len(self.list_num_frame))

	def __len__(self):
		if self.mode == 'train':
			return len(self.video_names)
		else:
			return len(self.list_num_frame)

	def __getitem__(self, idx):
		# print(self.mode)
		if self.mode == "train":
			video_name = self.video_names[idx]
			# video_name,mid_frame = self.list_num_frame[idx]
			
			num_frames = len(os.listdir(os.path.join(self.path_data, video_name, 'images')))


			mid_frame = np.random.randint(0,num_frames)
	

			frame_indices = get_frame_indices(mid_frame,self.len_snippet,num_frames)
			

		elif self.mode == "val":
			(video_name, mid_frame) = self.list_num_frame[idx]

			num_frames = len(os.listdir(os.path.join(self.path_data ,video_name,'images')))

			frame_indices = get_frame_indices(mid_frame,self.len_snippet,num_frames)

			
		elif self.mode == 'test':
			(video_name,num_frames, mid_frame) = self.list_num_frame[idx]

			frame_indices = get_frame_indices(mid_frame,self.len_snippet,num_frames)
		
		
		clip_img = []
		clip_gt = []

		list_clips,list_sal_clips = [],[]

			
		# for file_name in sorted(self.video_names):

		path_clip = os.path.join(self.path_data, video_name, 'images')
		path_annt = os.path.join(self.path_data, video_name, 'maps')		
		
		temp1 = sorted([join(path_clip,image_file) for image_file in os.listdir(path_clip)])
		temp2 = sorted([join(path_annt,image_file) for image_file in os.listdir(path_annt)])
		
		list_clips.extend(temp1)
		list_sal_clips.extend(temp2)

		

		frame_indices = [frame_indices[i] for i in range(len(frame_indices)) if i % 2 != 0] if self.len_snippet == 64 else frame_indices

		for frame_idx in frame_indices:
			img = Image.open(list_clips[frame_idx]).convert('RGB')
			clip_img.append(self.img_transform(img))	

		clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))			
			
		gt = np.array(Image.open(list_sal_clips[mid_frame]).convert('L'))
		gt = gt.astype('float')

		if self.mode != "test":

			if self.mode == "train":
				if self.use_skip:
					gt = cv2.resize(gt, (456, 256))
				else:
					gt = cv2.resize(gt, (448, 224))

			if np.max(gt) > 1.0:
				gt = gt / 255.0

			binary_img = np.zeros_like(gt)

		else:

			if self.use_skip:
				gt = cv2.resize(gt, (456, 256))
			else:
				gt = cv2.resize(gt, (448, 224))

			if np.max(gt) > 1.0:
				gt = gt / 255.0



			fix_map_filename = list_sal_clips[mid_frame].split('/')[-1].split('.')[0]
			video_clip_name = list_sal_clips[mid_frame].split('/')[-3]

			fix = sio.loadmat(join(self.path_data,video_clip_name,'fixation','maps','{}.mat'.format(fix_map_filename)))

			if self.mode == "test":
				binary_img = torch.FloatTensor(fix['I']).numpy()

		if self.mode == "train" or self.mode == "val":
			return clip_img, gt, binary_img
		else:
			return clip_img, gt, binary_img, video_name, mid_frame


class Hollywood2_Dataset(Dataset):
	def __init__(self, args,mode = 'train'):
		''' mode: train, val, perframe 
			frame_no: last, middle
		'''
		if mode == 'train':
			self.path_data = join(args.videos_root_path,'training')
		else:
			self.path_data = join(args.videos_root_path,'testing')
			
		self.len_snippet = args.len_snippet
		self.dataset_name = args.dataset
		self.mode = mode

		self.use_skip = args.use_skip


		self.img_transform = transforms.Compose([
			transforms.Resize((256, 456) if self.use_skip else (224,448)),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.450, 0.450, 0.450],#[0.485, 0.456, 0.406],
				[0.225, 0.225, 0.225]#[0.229, 0.224, 0.225]
			)
		])

		if self.mode == "train":
			self.video_clips = os.listdir(self.path_data)
			self.list_num_frame = [len(os.listdir(os.path.join(self.path_data, d, 'images'))) for d in self.video_clips]
			self.video_names = np.unique([video_name.split('_')[0] for video_name in self.video_clips])

			print("Number of train video names are : ",len(self.video_names))
			print("Number of train samples are : ",len(self.list_num_frame))
		
		elif self.mode == "val":
			self.list_num_frame = []
			self.video_clips = os.listdir(self.path_data)
			self.video_names = np.unique([file_name.split('_')[0] for file_name in self.video_clips])
			stride = 2*self.len_snippet if self.len_snippet == 32 else self.len_snippet
			for video_name in self.video_names:
				num_frames = sum([len(os.listdir(os.path.join(self.path_data, file_name, 'images'))) for file_name in self.video_clips if file_name.split('_')[0] == video_name])
				for mid_frame in range(0, num_frames-self.len_snippet, stride):
					self.list_num_frame.append((video_name, num_frames, mid_frame))

			print("Number of val video names are : ",len(self.video_names))
			print("Number of val samples are : ",len(self.list_num_frame))

		elif self.mode == "test":
			self.list_num_frame = []
			self.video_clips = os.listdir(self.path_data)
			self.video_names = np.unique([file_name.split('_')[0] for file_name in self.video_clips])

			if args.video_names_list != '':

				self.video_names = [i for i in self.video_names if i in args.video_names_list]

			for video_name in self.video_names:
				num_frames = sum([len(os.listdir(os.path.join(self.path_data,file_name,'images'))) for file_name in self.video_clips if file_name.split('_')[0] == video_name])
				# print('num_frames: ', num_frames)
				for mid_frame in range(0,num_frames):
					self.list_num_frame.append((video_name,num_frames,mid_frame))

			print("Number of test video names are : ",len(self.video_names))
			print("Number of test samples are : ",len(self.list_num_frame))

	def __len__(self):
		if self.mode == 'train':
			return len(self.video_names)
		return len(self.list_num_frame)
	
	def __getitem__(self, idx):
		if self.mode == "train":
			video_name = self.video_names[idx]
			video_clips = [video_clip for video_clip in self.video_clips if video_clip.split('_')[0] == video_name]
			num_frames = sum([len(os.listdir(os.path.join(self.path_data,file_name,'images'))) for file_name in video_clips])

			mid_frame = np.random.randint(0,max(num_frames - self.len_snippet,num_frames))

			frame_indices = get_frame_indices(mid_frame,self.len_snippet,num_frames)

		elif self.mode == "val":
			(video_name,num_frames, mid_frame) = self.list_num_frame[idx]
			
			video_clips = [video_clip for video_clip in self.video_clips if video_clip.split('_')[0] == video_name]
		
			frame_indices = get_frame_indices(mid_frame,self.len_snippet,num_frames)

		elif self.mode == 'test':
			(video_name,num_frames, mid_frame) = self.list_num_frame[idx]
			
			video_clips = [video_clip for video_clip in self.video_clips if video_clip.split('_')[0] == video_name]
		
			frame_indices = get_frame_indices(mid_frame,self.len_snippet,num_frames)

		
		
		clip_img = []
		clip_gt = []

		list_clips,list_sal_clips = [],[]
			
		for file_name in sorted(video_clips):

			path_clip = os.path.join(self.path_data, file_name, 'images')
			path_annt = os.path.join(self.path_data, file_name, 'maps')		
			
			temp1 = sorted([join(path_clip,image_file) for image_file in os.listdir(path_clip)])
			temp2 = sorted([join(path_annt,image_file) for image_file in os.listdir(path_annt)])
			
			list_clips.extend(temp1)
			list_sal_clips.extend(temp2)



		frame_indices = [frame_indices[i] for i in range(len(frame_indices)) if i % 2 != 0] if self.len_snippet == 64 else frame_indices

		for frame_idx in  frame_indices:#[frame_indices[i] for i in range(len(frame_indices)) if i % 2 != 0]:
			img = Image.open(list_clips[frame_idx]).convert('RGB')
			clip_img.append(self.img_transform(img))	

		clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))			
			
		gt = np.array(Image.open(list_sal_clips[mid_frame]).convert('L'))
		gt = gt.astype('float')

		if self.mode != "test":
			if self.mode == "train":
				if self.use_skip:
					gt = cv2.resize(gt, (456, 256))
				else:
					gt = cv2.resize(gt, (448, 224))

			if np.max(gt) > 1.0:
				gt = gt / 255.0

			binary_img = []

		else:

			if self.use_skip:
				gt = cv2.resize(gt, (456, 256))
			else:
				gt = cv2.resize(gt, (448, 224))

			if np.max(gt) > 1.0:
				gt = gt / 255.0


			fix_map_filename = list_sal_clips[mid_frame].split('/')[-1].split('.')[0]
			video_clip_name = list_sal_clips[mid_frame].split('/')[-3]


			fix = sio.loadmat(join(self.path_data,video_clip_name,'fixation','maps','{}.mat'.format(fix_map_filename)))

			binary_img = torch.FloatTensor(fix['I']).numpy()

		if self.mode == "train" or self.mode == "val":
			return clip_img, gt, binary_img
		else:
			return clip_img, gt, binary_img, video_name, mid_frame
		

class UCF_Dataset(Dataset):
	def __init__(self, args,mode = 'train'):

		if mode == 'train':
			self.path_data = join(args.videos_root_path,'training')
		else:
			self.path_data = join(args.videos_root_path,'testing')
			
		self.len_snippet = args.len_snippet
		self.dataset_name = args.dataset
		self.mode = mode

		self.use_skip = args.use_skip
		self.img_transform = transforms.Compose([
			transforms.Resize((256, 456) if self.use_skip else (224,448)),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.450, 0.450, 0.450],#[0.485, 0.456, 0.406],
				[0.225, 0.225, 0.225]#[0.229, 0.224, 0.225]
			)
		])

		if self.mode == "train":

			self.video_names = os.listdir(self.path_data)
			self.list_num_frame = [len(os.listdir(os.path.join(self.path_data, d, 'images'))) for d in self.video_names]
			
			print("Train video names are : ",self.video_names)


			print("Number of train video names are : ",len(self.video_names))
			print("Number of train samples are : ",len(self.list_num_frame))
		
		elif self.mode == "val":
			self.list_num_frame = []

			self.video_names = os.listdir(self.path_data)

			for v in os.listdir(self.path_data ):

				if len(os.listdir(os.path.join(self.path_data ,v,'images')))<= int(self.len_snippet):
					self.list_num_frame.append((v, len(os.listdir(os.path.join(self.path_data ,v,'images'))) // 2))
				else:
					for i in range(0, len(os.listdir(os.path.join(self.path_data ,v,'images'))), int(0.5*self.len_snippet)):
						self.list_num_frame.append((v, i))

			print("Number of val video names are : ",len(self.video_names))
			print("Number of val samples are : ",len(self.list_num_frame))

		elif self.mode == "test":
			self.list_num_frame = []
			self.video_names = os.listdir(self.path_data)

			

			print("test video names are : ",self.video_names)

			if args.video_names_list != '':

				self.video_names = [i for i in self.video_names if i in args.video_names_list]

			print("test video names are : ",self.video_names)

			

			for video_name in self.video_names:
				num_frames = len(os.listdir(os.path.join(self.path_data, video_name, 'images')))
				for mid_frame in range(0,num_frames):
					self.list_num_frame.append((video_name,num_frames,mid_frame))

			print("Number of test video names are : ",len(self.video_names))
			print("Number of test samples are : ",len(self.list_num_frame))

	def __len__(self):
		if self.mode == 'train':
			return len(self.video_names)
		return len(self.list_num_frame)
	
	def __getitem__(self, idx):
		if self.mode == "train":
			video_name = self.video_names[idx]

			num_frames = len(os.listdir(os.path.join(self.path_data, video_name, 'images')))

			mid_frame = np.random.randint(0,num_frames)


			frame_indices = get_frame_indices(mid_frame,self.len_snippet,num_frames)

		elif self.mode == "val":
			(video_name, mid_frame) = self.list_num_frame[idx]

			num_frames = len(os.listdir(os.path.join(self.path_data ,video_name,'images')))
		
			frame_indices = get_frame_indices(mid_frame,self.len_snippet,num_frames)


		elif self.mode == 'test':

			(video_name,num_frames, mid_frame) = self.list_num_frame[idx]
			
	
			frame_indices = get_frame_indices(mid_frame,self.len_snippet,num_frames)
		
		
		clip_img = []
		clip_gt = []

		list_clips,list_sal_clips = [],[]

		path_clip = os.path.join(self.path_data, video_name, 'images')
		path_annt = os.path.join(self.path_data, video_name, 'maps')		
		
		temp1 = sorted([join(path_clip,image_file) for image_file in os.listdir(path_clip)])
		temp2 = sorted([join(path_annt,image_file) for image_file in os.listdir(path_annt)])
		
		list_clips.extend(temp1)
		list_sal_clips.extend(temp2)



		frame_indices = [frame_indices[i] for i in range(len(frame_indices)) if i % 2 != 0] if self.len_snippet == 64 else frame_indices
		for frame_idx in  frame_indices:#[frame_indices[i] for i in range(len(frame_indices)) if i % 2 != 0]:
			img = Image.open(list_clips[frame_idx]).convert('RGB')
			clip_img.append(self.img_transform(img))	

		clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))			
			
		gt = np.array(Image.open(list_sal_clips[mid_frame]).convert('L'))
		gt = gt.astype('float')

		if self.mode != "test":

			if self.mode == "train":
				if self.use_skip:
					gt = cv2.resize(gt, (456, 256))
				else:
					gt = cv2.resize(gt, (448, 224))

			if np.max(gt) > 1.0:
				gt = gt / 255.0

			binary_img = np.zeros_like(gt)


		else:

			if self.use_skip:
				gt = cv2.resize(gt, (456, 256))
			else:
				gt = cv2.resize(gt, (448, 224))

			if np.max(gt) > 1.0:
				gt = gt / 255.0

			fix_map_filename = list_sal_clips[mid_frame].split('/')[-1].split('.')[0]
			video_clip_name = list_sal_clips[mid_frame].split('/')[-3]

			fix = sio.loadmat(join(self.path_data,video_clip_name,'fixation','maps','{}.mat'.format(fix_map_filename)))

			binary_img = torch.FloatTensor(fix['I']).numpy()

		if self.mode == "train" or self.mode == "val":
			return clip_img, gt, binary_img
		else:
			return clip_img, gt, binary_img, video_name, mid_frame
