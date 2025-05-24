
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


class MVVA_Dataset(data.Dataset):
	def __init__(self,
				 args,
				 mode='train'
				 ):


		self.args = args
		self.mode = mode

		self.use_skip = args.use_skip

		self.videos_root_path = args.videos_root_path

		self.videos_frames_root_path = args.videos_frames_root_path

		self.gt_sal_maps_path = args.gt_sal_maps_path

		self.fixation_data_path = args.fixation_data_path


		self.mode = mode

		self.dataset = args.dataset

		self.len_snippet = args.len_snippet


				

		if (self.mode == 'val' or self.mode == 'test') and args.dataset == 'mvva':
			mode = 'test'
		file_name = '{}_list_{}_{}_fps.txt'.format(args.dataset.lower(), mode, args.split)
		print("Taking the split file: ", file_name)
		
		self.list_indata = []
		with open(join(args.fold_lists_path, file_name), 'r') as f:
			for line in f.readlines():
				name = line.split(' ')[0].strip()

				self.list_indata.append(name)


		self.video_names = [video_name for video_name in os.listdir(self.videos_frames_root_path)]

		self.video_names = [i for i in self.video_names if i in self.list_indata]
		

		self.img_transform = transforms.Compose([
			transforms.Resize((256, 456) if self.use_skip else (224,448)),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.450, 0.450, 0.450],#[0.485, 0.456, 0.406],
				[0.225, 0.225, 0.225]#[0.229, 0.224, 0.225]
			)
		])


		if self.mode == 'val':

			self.video_names = [video_name for video_name in os.listdir(self.videos_frames_root_path)]
			
			self.video_names = [i for i in self.video_names if i in self.list_indata]

			self.list_num_frame = []

			for p in self.video_names:

				num_frames = len(os.listdir(join(self.videos_frames_root_path, p)))

				for i in range(0, num_frames - self.len_snippet,int(self.len_snippet)):

					self.list_num_frame.append((p, i))
			print("Number of validation samples: ",len(self.list_num_frame))

		if self.mode == 'test':

			self.video_names = [i for i in self.video_names if i in self.list_indata]

			print("Test Video names are ",self.video_names)

			self.list_num_frame = []

			for p in self.video_names :
				num_frames = len(os.listdir(join(self.videos_frames_root_path, p)))


			for video_name in self.video_names:
				num_frames = len(os.listdir(join(self.videos_frames_root_path, video_name)))
				for mid_frame in range(0,num_frames):
					self.list_num_frame.append((video_name,num_frames,mid_frame))

			print("Number of test frames is : ", len(self.list_num_frame))



	def __getitem__(self, index):

		if self.mode == 'train':
			video_name = self.video_names[index] 
			num_frames = len(os.listdir(join(self.videos_frames_root_path, video_name)))

		
		if self.mode == 'val':

			video_file, start_frame = self.list_num_frame[index]

			video_name = video_file.split('.')[0]

			num_frames = len(os.listdir(join(self.videos_frames_root_path, video_name)))
		
		if self.mode == 'train':

			start_frame = np.random.randint(
					0, num_frames - self.len_snippet)

			# num_rois = self.frame_class_data[video_name]['num_rois']


		if self.mode == 'train' or self.mode == 'val':
			clip = []
			for i in range(0,self.len_snippet,2):
				img = Image.open(os.path.join(join(self.videos_frames_root_path, video_name), 'img_%05d.jpg' % (
					start_frame + i+1))).convert('RGB')
				
				img = img.resize((1280,720))
				W,H = img.size
				img = self.img_transform(img)
				clip.append(img)
				
			clip_img = torch.FloatTensor(torch.stack(clip, dim=0))

	
			######## Getting ground truth saliency map ###############
			path_annt = os.path.join(self.gt_sal_maps_path, video_name,'maps')

			gt = np.array(Image.open(join(path_annt, 'eyeMap_%05d.jpg'%(start_frame+self.len_snippet))).convert('L'))
			gt = gt.astype('float')

			if self.mode == "train":
				if self.use_skip:
					gt = cv2.resize(gt, (456, 256))
				else:
					gt = cv2.resize(gt, (448, 224))

			if np.max(gt) > 1.0:
				gt = gt / 255.0

	
			########### Getting the fixation map	#######################
			count = 0
			fix_data = np.load(join(self.fixation_data_path,video_name + '.npy'))
			binary_img = np.zeros((H,W),dtype=np.uint8)


			assert num_frames == fix_data.shape[0]

			for j in range(fix_data.shape[2]):
				x = fix_data[start_frame+self.len_snippet - 1,0,j]
				y = fix_data[start_frame+self.len_snippet - 1,1,j]
				try:           
					binary_img[int(y),int(x)] = 1
				except:
					count+=1

				# print("Binary image shape after concate : ",binary_img.shape)

		### Getting test data			
		if self.mode == 'test':

			# video_name, clip_frameindices,mid_frame = self.list_num_frame[index]

			(video_name,num_frames, mid_frame) = self.list_num_frame[index]
			

			frame_indices = get_frame_indices(mid_frame,self.len_snippet,num_frames)

			# num_frames = len(os.listdir(join(self.videos_frames_root_path, video_name)))

			frame_indices = [frame_indices[i] for i in range(len(frame_indices)) if i % 2 != 0] if self.len_snippet == 64 else frame_indices

			clip = []
			for i in frame_indices:#[clip_frameindices[i] for i in range(len(clip_frameindices)) if i % self.alternate == 0]:
				img = Image.open(os.path.join(join(self.videos_frames_root_path, video_name), 'img_%05d.jpg' % (i+1))).convert('RGB')
				
				img = img.resize((1280,720))

				W,H = img.size

				img = self.img_transform(img)				
				clip.append(img)
			clip_img = torch.FloatTensor(torch.stack(clip, dim=0))


			###################### Getting the  ground truth ###################
			path_annt = os.path.join(self.gt_sal_maps_path, video_name,'maps')
			gt_sal = np.array(Image.open(join(path_annt, 'eyeMap_%05d.jpg'%(mid_frame+1))).convert('L'))
			gt = gt_sal.astype('float')


			########### Getting the fixation map
			count = 0
			fix_data = np.load(join(self.fixation_data_path,video_name + '.npy'))
			binary_img = np.zeros((H // 2,W // 2),dtype=np.uint8)

			assert num_frames == fix_data.shape[0]

			for j in range(fix_data.shape[2]):
				x = fix_data[mid_frame,0,j]
				y = fix_data[mid_frame,1,j]
				try:           
					binary_img[int(y/2),int(x/2)] = 1
				except:
					count+=1
			# gt = []


		if self.mode != 'test':
			return clip_img,gt,binary_img
		return clip_img,gt,binary_img,video_name,mid_frame

	def __len__(self):
		if self.mode == 'train':
			return len(self.video_names)
		else:
		   return(len(self.list_num_frame))



class Other_AudioVisual_Dataset(Dataset): #coutrot2,coutrot1,DIEM,AVAD, ETMD,
	def __init__(self, args,dataset, mode='train',stride=0.5,use_extra_data=False):
		self.len_snippet = args.len_snippet

		self.dataset = dataset
		self.mode = mode
		# self.use_skip = bool(args.use_skip)
		self.split = args.split
		self.path_data = '~/Audio-Visual-SaliencyDatasets'
		self.use_extra_data = use_extra_data
		self.stride = stride


		self.img_transform = transforms.Compose([
			transforms.Resize((256, 456) if self.use_skip else (224, 448)),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.450, 0.450, 0.450],
				[0.225, 0.225, 0.225]
			)
		])

		
		self.video_names = [v for v in os.listdir(join(self.path_data, 'video_frames', self.dataset))]

		if self.dataset=='DIEM':
			mode = 'test' if self.mode == 'val' or self.mode == 'test' else 'train'
			file_name = 'DIEM_list_{}_fps.txt'.format(mode)
		else:
			if (self.mode == 'val' or self.mode == 'test'):
				mode = 'test'
			file_name = '{}_list_{}_{}_fps.txt'.format(self.dataset, mode, self.split)
			print("Taking the split file: ", file_name)

		self.list_indata = []
		with open(join(self.path_data, 'fold_lists', file_name), 'r') as f:
			for line in f.readlines():
				name = line.split(' ')[0].strip()
				self.list_indata.append(name)

		self.video_names = [i for i in self.video_names if i in self.list_indata]

		print("Video names are : ",self.video_names)

		if self.mode == 'train':

			if self.use_extra_data:

				print("Using extra data-----------------",self.dataset)

				self.list_num_frame = []

				for video_name in self.video_names:

					num_frames = len(os.listdir(join(self.path_data,'video_frames',self.dataset,video_name)))

					scene_start = 0
					scene_end = num_frames // 2

					self.list_num_frame.append((video_name, scene_start,scene_end))

					scene_start = num_frames // 2
					scene_end = num_frames

					self.list_num_frame.append((video_name, scene_start,scene_end))


			else:

				self.list_num_frame = []


				for video_name in self.video_names:

					num_frames = len(os.listdir(join(self.path_data,'video_frames',self.dataset,video_name)))

					scene_start = 0
					scene_end = num_frames
					self.list_num_frame.append((video_name, scene_start,scene_end))

			print("Number of training videos are : ", len(self.video_names))
			print("Number of training samples are : ", len(self.list_num_frame))

		if self.mode == 'val':
			self.list_num_frame = []
			for video_name in self.video_names:
				num_frames = len(os.listdir(join(self.path_data,'video_frames',self.dataset,video_name)))
				for i in range(0, num_frames - self.len_snippet, int(self.stride*self.len_snippet)):
					self.list_num_frame.append((video_name, i))
			print("number of validation videos are : ", len(self.video_names))
			print("number of validation samples are : ", len(self.list_num_frame))

		elif self.mode == 'test':
			self.list_num_frame = []
			self.video_names = sorted(self.video_names)
			# self.video_names = self.video_names[:1]
			# print("Video name is ",self.video_names)

			for video_name in self.video_names:
				num_frames = len(os.listdir(os.path.join(self.path_data,'video_frames',self.dataset,video_name)))
				print("Number of frames in video ",video_name," are : ",num_frames)
				for mid_frame in range(0,num_frames):
					self.list_num_frame.append((video_name,num_frames,mid_frame))

			print("Number of test video names are : ",len(self.video_names))
			print("Number of test samples are : ",len(self.list_num_frame))

	def __len__(self):
		if self.mode == 'train':
			return len(self.list_num_frame)
		else:
			return len(self.list_num_frame)
		
	def __getitem__(self, index):
		if self.mode == 'train':
			video_name,scene_start,scene_end = self.list_num_frame[index]
			num_frames = len(os.listdir(join(self.path_data,'video_frames',self.dataset,video_name)))
			
			mid_frame = np.random.randint(scene_start,scene_end)


			frame_indices = get_frame_indices(mid_frame,self.len_snippet,num_frames)

		
		if self.mode == 'val':
			video_file, mid_frame = self.list_num_frame[index]
			video_name = video_file.split('.')[0]
			num_frames = len(os.listdir(join(self.path_data,'video_frames',self.dataset,video_name)))


			frame_indices = get_frame_indices(mid_frame,self.len_snippet,num_frames)


		if self.mode == 'train' or self.mode == 'val':
			clip_img = []


			frame_indices = frame_indices if self.len_snippet == 32 else frame_indices[::2]

			for i in frame_indices:

				img = Image.open(os.path.join(join(self.path_data,'video_frames',self.dataset,video_name), 'img_%05d.jpg' % (i+1))).convert('RGB')

				W, H = img.size 
				img = self.img_transform(img)

				clip_img.append(img)
			
			clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))


			########### Getting Ground truth saliency map

			path_annt = os.path.join(self.path_data,'annotations',self.dataset,video_name,'maps')
			gt = np.array(Image.open(join(path_annt,'eyeMap_%05d.jpg'%(mid_frame+1))).convert('L'))
			gt = gt.astype('float')

			if self.mode == 'train':
				if self.use_skip:
					gt = cv2.resize(gt, (456, 256))
				else:
					gt = cv2.resize(gt, (448, 224))

			if np.max(gt) > 1.0:
				gt = gt / 255.0


			########## Getting binary fixation map
			binary_img = np.array(loadmat(join(self.path_data, 'annotations', self.dataset, video_name, 'fixMap_%05d.mat')%(mid_frame+1))['eyeMap'])


		if self.mode == 'test':
			video_name, num_frames, mid_frame = self.list_num_frame[index]
			num_frames = len(os.listdir(join(self.path_data,'video_frames',self.dataset,video_name)))



			frame_indices = get_frame_indices(mid_frame,self.len_snippet,num_frames)

			frame_indices = frame_indices if self.len_snippet == 32 else frame_indices[::2]


			clip_img = []

			for i in frame_indices:
				img = Image.open(os.path.join(join(self.path_data,'video_frames',self.dataset,video_name), 'img_%05d.jpg' % (i+1))).convert('RGB')

				W, H = img.size
				img = self.img_transform(img)

				clip_img.append(img)


			clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))


			########### Getting Ground truth saliency map

			gt, binary_img = [], []

			path_annt = os.path.join(self.path_data,'annotations',self.dataset,video_name,'maps')
			gt = np.array(Image.open(join(path_annt,'eyeMap_%05d.jpg'%(mid_frame+1))).convert('L'))
			gt = gt.astype('float')
			if np.max(gt) > 1.0:
				gt = gt / 255.0
	

			########## Getting binary fixation map
			binary_img = np.array(loadmat(join(self.path_data, 'annotations', self.dataset, video_name, 'fixMap_%05d.mat')%(mid_frame+1))['eyeMap'])
			
			

		if self.mode == "train" or self.mode == "val":
			
			return clip_img, gt, binary_img
		else:
			return clip_img, gt, binary_img, video_name, mid_frame
