import os
from os.path import join
import cv2, copy
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torchaudio
import sys
import json

import scipy.io as sio


class OtherAudioVisualDataset(Dataset):
	def __init__(self, len_snippet, dataset_name='DIEM', split=1, mode='train',args=None):
		''' mode: train, val, save '''

		self.path_data = args.videos_root_path
		self.fold_lists_path = args.fold_lists_path

		self.mode = mode
		self.len_snippet = len_snippet
		self.img_transform = transforms.Compose([
			transforms.Resize((224, 384)),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.485, 0.456, 0.406],
				[0.229, 0.224, 0.225]
			)
		])
		self.list_num_frame = []
		self.dataset_name = dataset_name
		if (self.mode == 'val' or self.mode == 'test'):
			mode = 'test'
		if dataset_name=='DIEM':
			file_name = 'DIEM_list_{}_fps.txt'.format(mode)
		else:
			file_name = '{}_list_{}_{}_fps.txt'.format(dataset_name, mode, split)
		
		self.list_indata = []
		with open(join(self.fold_lists_path, file_name), 'r') as f:

			for line in f.readlines():
				name = line.split(' ')[0].strip()
				self.list_indata.append(name)

		self.list_indata.sort()	
		print(self.mode, len(self.list_indata))
		if self.mode=='train':
			self.list_num_frame = [len(os.listdir(os.path.join(self.path_data,'annotations', dataset_name, v, 'maps'))) for v in self.list_indata]
		
		elif self.mode == 'val': 
			print("val set")
			for v in self.list_indata:
				frames = os.listdir(join(self.path_data, 'annotations', dataset_name, v, 'maps'))
				frames.sort()
				for i in range(0, len(frames)-self.len_snippet,  2*self.len_snippet):
					if self.check_frame(join(self.path_data, 'annotations', dataset_name, v, 'maps', 'eyeMap_%05d.jpg'%(i+self.len_snippet))):
						self.list_num_frame.append((v, i))

		elif self.mode == 'test':
			print("test set")
			if args.video_names_list != '':

				self.video_names = [i for i in self.list_indata if i in args.video_names_list]

			print("test video names are : ",self.video_names)

			for v in self.video_names:
				frames = os.listdir(join(self.path_data, 'annotations', dataset_name, v, 'maps'))
				frames.sort()
				num_frames = len(frames)

				for i in range(0, num_frames):
					self.list_num_frame.append((v,num_frames, i))

	def check_frame(self, path):
		img = cv2.imread(path, 0)
		return img.max()!=0

	def __len__(self):
		return len(self.list_num_frame)

	def __getitem__(self, idx):
		# print(self.mode)
		if self.mode == "train":
			video_name = self.list_indata[idx]
			while 1:
				start_idx = np.random.randint(0, self.list_num_frame[idx]-self.len_snippet+1)
				if self.check_frame(join(self.path_data, 'annotations', self.dataset_name, video_name, 'maps', 'eyeMap_%05d.jpg'%(start_idx+self.len_snippet))):
					break
				else:
					print("No saliency defined in train dataset")
					sys.stdout.flush()

		elif self.mode == "val":
			(video_name, start_idx) = self.list_num_frame[idx]

		elif self.mode == "test":
			(video_name, num_frames, end_idx) = self.list_num_frame[idx]

		path_clip = os.path.join(self.path_data, 'video_frames', self.dataset_name, video_name)
		path_annt = os.path.join(self.path_data, 'annotations', self.dataset_name, video_name, 'maps')


		if self.mode != "test":
			clip_img = []
			
			for i in range(self.len_snippet):
				img = Image.open(join(path_clip, 'img_%05d.jpg'%(start_idx+i+1))).convert('RGB')
				sz = img.size		
				clip_img.append(self.img_transform(img))
				
			clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))
			
			gt = np.array(Image.open(join(path_annt, 'eyeMap_%05d.jpg'%(start_idx+self.len_snippet))).convert('L'))
			gt = gt.astype('float')
			
			if self.mode == "train":
				gt = cv2.resize(gt, (384, 224))

			if np.max(gt) > 1.0:
				gt = gt / 255.0
			assert gt.max()!=0, (start_idx, video_name)

		else:
			frame_indices = np.arange(end_idx-self.len_snippet + 1, end_idx + 1)

			temp = []
			for i in frame_indices:
				if i < 0:
					temp.append(0)
				else:
					temp.append(i)
			frame_indices = temp

			clip_img = []

			for i in frame_indices:
				img = Image.open(join(path_clip, 'img_%05d.jpg'%(i+1))).convert('RGB')
				sz = img.size
				clip_img.append(self.img_transform(img))

			clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))

			# added
			if end_idx == 125 and video_name == 'V11_News4':
				end_idx = 124
			# X----X----X
			gt = np.array(Image.open(join(path_annt, 'eyeMap_%05d.jpg'%(end_idx+1))).convert('L'))
			gt = gt.astype('float')

			if np.max(gt) > 1.0:
				gt = gt / 255.0
			gt = torch.FloatTensor(gt)

			binary_img = np.array(sio.loadmat(join(self.path_data, 'annotations', self.dataset_name, video_name, 'fixMap_%05d.mat')%(end_idx+1))['eyeMap'])

		if self.mode == "test":
			return clip_img, gt, binary_img, video_name, end_idx
		return clip_img, gt

class DHF1KDataset(Dataset):
	def __init__(self, len_snippet, mode="train", alternate=1, frames_path='images',args=None):
		''' mode: train, val, save '''


		if mode == 'train':
			self.path_data = join(args.videos_root_path,'annotation') if os.path.exists(join(args.videos_root_path,'annotation')) else join(args.videos_root_path,'train')
		if mode == 'val':
			self.path_data = join(args.videos_root_path,'val')
		elif mode == 'test' :
			self.path_data = join(args.videos_root_path,'val')
			
		self.frames_path = frames_path
		self.len_snippet = len_snippet
		self.mode = mode

		self.alternate = alternate
		self.img_transform = transforms.Compose([
			transforms.Resize((224, 384)),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.485, 0.456, 0.406],
				[0.229, 0.224, 0.225]
			)
		])
		if self.mode == "train":
			self.video_names = os.listdir(self.path_data)
			self.list_num_frame = [len(os.listdir(os.path.join(self.path_data,d,frames_path))) for d in self.video_names]
		elif self.mode=="val":
			self.list_num_frame = []
			for v in os.listdir(self.path_data):
				for i in range(0, len(os.listdir(os.path.join(self.path_data,v,frames_path)))- self.alternate * self.len_snippet, 4*self.len_snippet):
					self.list_num_frame.append((v, i))
		else:
			self.list_num_frame = []

			self.video_names = os.listdir(self.path_data)

			if args.video_names_list != '':
				self.video_names = [i for i in self.video_names if i in args.video_names_list]

			print("Number of test videos: ", len(self.video_names))

			
			for v in self.video_names:
				num_frames = len(os.listdir(os.path.join(self.path_data,v,frames_path)))
				for i in range(0,num_frames):
					self.list_num_frame.append((v,i))

			print("Number of test samples: ", len(self.list_num_frame))

	def __len__(self):
		return len(self.list_num_frame)

	def __getitem__(self, idx):

		if self.mode == "train":
			file_name = self.video_names[idx]
			start_idx = np.random.randint(0, self.list_num_frame[idx]-self.alternate * self.len_snippet+1)

		elif self.mode == "val":
			(file_name, start_idx) = self.list_num_frame[idx]

		elif self.mode == "test":
			(file_name, end_idx) = self.list_num_frame[idx]

		path_clip = os.path.join(self.path_data, file_name, self.frames_path)
		path_annt = os.path.join(self.path_data, file_name, 'maps')

		clip_img = []
		clip_gt = []

		if self.mode != 'test':
		
			for i in range(self.len_snippet):
				img = Image.open(os.path.join(path_clip, '%04d.png'%(start_idx+self.alternate*i+1))).convert('RGB')
				sz = img.size

				gt = np.array(Image.open(os.path.join(path_annt, '%04d.png'%(start_idx+self.alternate*i+1))).convert('L'))
				gt = gt.astype('float')
				
				if self.mode == "train":
					gt = cv2.resize(gt, (384, 224))
				
				if np.max(gt) > 1.0:
					gt = gt / 255.0
				clip_gt.append(torch.FloatTensor(gt))

				clip_img.append(self.img_transform(img))
				
			clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))

		else:

			frame_indices = np.arange(end_idx-self.len_snippet + 1, end_idx + 1)

			temp = []
			for i in frame_indices:
				if i < 0:
					temp.append(0)
				else:
					temp.append(i)
			frame_indices = temp


			for i in frame_indices:
				img = Image.open(os.path.join(path_clip, '%04d.png'%(i+1))).convert('RGB')
				sz = img.size
				clip_img.append(self.img_transform(img))

			clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))
			gt = np.array(Image.open(os.path.join(path_annt, '%04d.png'%(end_idx+1))).convert('L'))
			gt = gt.astype('float')
			if np.max(gt) > 1.0:
				gt = gt / 255.0
			clip_gt = torch.FloatTensor(gt)


			fix = sio.loadmat(join(self.path_data,file_name,'fixation','maps','%04d.mat'%(end_idx+1)))

			if self.mode == "test":
				binary_img = torch.FloatTensor(fix['I']).numpy()


		if self.mode == "train" or self.mode == "val":
			clip_gt = torch.FloatTensor(torch.stack(clip_gt, dim=0))
			return clip_img, clip_gt[-1]#,binary_img
		elif self.mode=="test":
			return clip_img, clip_gt,binary_img, file_name, end_idx

class UCFDataset(Dataset):
	def __init__(self, len_snippet, mode="train", frames_path='images',args=None):
		''' mode: train, val, perframe 
			frame_no: last, middle
		'''
		# self.path_data = path_data
		if mode == 'train':
			self.path_data = join(args.videos_root_path,'training')
		else:
			self.path_data = join(args.videos_root_path,'testing')

		self.frames_path = frames_path
		self.len_snippet = len_snippet
		self.mode = mode
		# self.frame_no = frame_no
		# self.multi_frame = multi_frame
		self.img_transform = transforms.Compose([
			transforms.Resize((224, 384)),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.485, 0.456, 0.406],
				[0.229, 0.224, 0.225]
			)
		])
		if self.mode == "train":
			self.video_names = os.listdir(self.path_data)
			self.list_num_frame = [len(os.listdir(os.path.join(self.path_data,d,frames_path))) for d in self.video_names]

			print("Number of training samples are: ",len(self.list_num_frame))


		elif self.mode=="val":
			self.list_num_frame = []
			for v in os.listdir(self.path_data):
				for i in range(0, len(os.listdir(os.path.join(self.path_data,v,frames_path)))-self.len_snippet, self.len_snippet):
					self.list_num_frame.append((v, i))
				if len(os.listdir(os.path.join(self.path_data,v,frames_path)))<=self.len_snippet:
					self.list_num_frame.append((v, 0))
			print("Number of validation samples are: ",len(self.list_num_frame))
		# added
		else:
			self.list_num_frame = []

			self.video_names = os.listdir(self.path_data)

			if args.video_names_list != '':

				self.video_names = [i for i in self.video_names if i in args.video_names_list]

			print("test video names are : ",self.video_names)

			for v in self.video_names:
				num_frames = len(os.listdir(os.path.join(self.path_data,v,frames_path)))
				for i in range(0,num_frames):
					self.list_num_frame.append((v,i))
		
	def __len__(self):
		return len(self.list_num_frame)

	def __getitem__(self, idx):
		if self.mode == "train":
			file_name = self.video_names[idx]
			start_idx = np.random.randint(0, max(1, self.list_num_frame[idx]-self.len_snippet+1))
		elif self.mode == "val":
			(file_name, start_idx) = self.list_num_frame[idx]
		# added
		elif self.mode == "test":
			(file_name, end_idx) = self.list_num_frame[idx]

			clip_img = []
			clip_gt = []

			list_clips,list_sal_clips = [],[]

			path_clip = os.path.join(self.path_data, file_name, 'images')
			path_annt = os.path.join(self.path_data, file_name, 'maps')		
			
			temp1 = sorted([join(path_clip,image_file) for image_file in os.listdir(path_clip)])
			temp2 = sorted([join(path_annt,image_file) for image_file in os.listdir(path_annt)])
			
			list_clips.extend(temp1)
			list_sal_clips.extend(temp2)

		if self.mode != "test":

			path_clip = os.path.join(self.path_data, file_name, self.frames_path)
			path_annt = os.path.join(self.path_data, file_name, 'maps')

			clip_img = []
			clip_gt = []

			list_clips = os.listdir(path_clip)
			list_clips.sort()
			list_sal_clips = os.listdir(path_annt)
			list_sal_clips.sort()
			
			if len(list_sal_clips)<self.len_snippet:
				temp = [list_clips[0] for _ in range(self.len_snippet-len(list_clips))]
				temp.extend(list_clips)
				list_clips = copy.deepcopy(temp)

				temp = [list_sal_clips[0] for _ in range(self.len_snippet-len(list_sal_clips))]
				temp.extend(list_sal_clips)
				list_sal_clips = copy.deepcopy(temp)

				assert len(list_sal_clips) == self.len_snippet and len(list_clips)==self.len_snippet
			for i in range(self.len_snippet):
				try:
					img = Image.open(os.path.join(path_clip, list_clips[start_idx+i])).convert('RGB')
				except:
					print(len(list_clips), start_idx,i, len(list_sal_clips))
				clip_img.append(self.img_transform(img))

				try:
					gt = np.array(Image.open(os.path.join(path_annt, list_sal_clips[start_idx+i])).convert('L'))
					gt = gt.astype('float')
				except:
					print(len(list_clips), start_idx,i, len(list_sal_clips))
				
				if self.mode == "train":
					gt = cv2.resize(gt, (384, 224))
				
				if np.max(gt) > 1.0:
					gt = gt / 255.0
				clip_gt.append(torch.FloatTensor(gt))
				
			clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))

		# added
		else:
			frame_indices = np.arange(end_idx-self.len_snippet + 1, end_idx + 1)

			temp = []
			for i in frame_indices:
				if i < 0:
					temp.append(0)
				else:
					temp.append(i)
			frame_indices = temp

			for i in frame_indices:
				img = Image.open(os.path.join(path_clip, list_clips[i])).convert('RGB')
				sz = img.size
				clip_img.append(self.img_transform(img))

			clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))
			gt = np.array(Image.open(os.path.join(path_annt, list_sal_clips[end_idx])).convert('L'))
			gt = gt.astype('float')
			if np.max(gt) > 1.0:
				gt = gt / 255.0
			clip_gt = torch.FloatTensor(gt)

			fix_map_filename = list_sal_clips[end_idx].split('/')[-1].split('.')[0]
			video_clip_name = list_sal_clips[end_idx].split('/')[-3]

			fix = sio.loadmat(join(self.path_data,video_clip_name,'fixation','maps','{}.mat'.format(fix_map_filename)))

			if self.mode == "test":
				binary_img = torch.FloatTensor(fix['I']).numpy()

		if self.mode == "train" or self.mode == "val":
			gt = torch.FloatTensor(torch.stack(clip_gt, dim=0))[-1]
			return clip_img, gt#, binary_img
		elif self.mode == "test":
			return clip_img, gt, binary_img, file_name, end_idx
			
class HollywoodDataset(Dataset):
	def __init__(self,len_snippet, mode="train",  frames_path='images',args=None):

		if mode == 'train':
			self.path_data = join(args.videos_root_path,'training')
		else:
			self.path_data = join(args.videos_root_path,'testing')

		self.frames_path = frames_path
		self.len_snippet = len_snippet
		self.mode = mode

		self.img_transform = transforms.Compose([
			transforms.Resize((224, 384)),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.485, 0.456, 0.406],
				[0.229, 0.224, 0.225]
			)
		])
		if self.mode == "train":
			self.video_names = os.listdir(self.path_data)
			self.list_num_frame = [len(os.listdir(os.path.join(self.path_data,d,frames_path))) for d in self.video_names]

		
		elif self.mode=="val":
			self.list_num_frame = []
			for v in os.listdir(self.path_data):
				for i in range(0, len(os.listdir(os.path.join(self.path_data,v,frames_path)))-self.len_snippet, self.len_snippet):
					self.list_num_frame.append((v, i))
				if len(os.listdir(os.path.join(self.path_data,v,frames_path)))<=self.len_snippet:
					self.list_num_frame.append((v, 0))
		# added
	
		
		else:
			self.list_num_frame = []
			self.video_clips = os.listdir(self.path_data)
			self.video_names = np.unique([file_name.split('_')[0] for file_name in self.video_clips])

			if args.video_names_list != '':

				self.video_names = [i for i in self.video_names if i in args.video_names_list]

			for video_name in self.video_names:
				num_frames = sum([len(os.listdir(os.path.join(self.path_data,file_name,'images'))) for file_name in self.video_clips if file_name.split('_')[0] == video_name])

				for end_idx in range(0,num_frames):
					self.list_num_frame.append((video_name,num_frames,end_idx))

			print("Number of test video names are : ",len(self.video_names))
			print("Number of test samples are : ",len(self.list_num_frame))
		
	def __len__(self):
		# return len(self.list_num_frame)
		if self.mode == 'train':
			return len(self.video_names)
		return len(self.list_num_frame)
	
	def __getitem__(self, idx):
		if self.mode == "train":
			file_name = self.video_names[idx]
			start_idx = np.random.randint(0, max(1, self.list_num_frame[idx]-self.len_snippet))

			# video_name = self.video_names[idx]
			# video_clips = [video_clip for video_clip in self.video_clips if video_clip.split('_')[0] == video_name]
			# num_frames = sum([len(os.listdir(os.path.join(self.path_data,file_name,'images'))) for file_name in video_clips])

		elif self.mode == "val":
			(file_name, start_idx) = self.list_num_frame[idx]
		# added
			# (video_name,num_frames, mid_frame) = self.list_num_frame[idx]
			
			# video_clips = [video_clip for video_clip in self.video_clips if video_clip.split('_')[0] == video_name]


		elif self.mode == "test":
			(file_name,num_frames, end_idx) = self.list_num_frame[idx]
			video_name = file_name

			video_clips = [video_clip for video_clip in self.video_clips if video_clip.split('_')[0] == file_name]


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

		if self.mode != "test":
			path_clip = os.path.join(self.path_data, file_name, self.frames_path)
			path_annt = os.path.join(self.path_data, file_name, 'maps')

			clip_img = []
			clip_gt = []

			list_clips = os.listdir(path_clip)
			list_clips.sort()
			list_sal_clips = os.listdir(path_annt)
			list_sal_clips.sort()

			if len(list_sal_clips)<self.len_snippet:
				temp = [list_clips[0] for _ in range(self.len_snippet-len(list_clips))]
				temp.extend(list_clips)
				list_clips = copy.deepcopy(temp)

				temp = [list_sal_clips[0] for _ in range(self.len_snippet-len(list_sal_clips))]
				temp.extend(list_sal_clips)
				list_sal_clips = copy.deepcopy(temp)

				assert len(list_sal_clips) == self.len_snippet and len(list_clips)==self.len_snippet

				
			for i in range(self.len_snippet):
				img = Image.open(os.path.join(path_clip, list_clips[start_idx+i])).convert('RGB')
				clip_img.append(self.img_transform(img))

				gt = np.array(Image.open(os.path.join(path_annt, list_sal_clips[start_idx+i])).convert('L'))
				gt = gt.astype('float')
				
				if self.mode == "train":
					gt = cv2.resize(gt, (384, 224))
				
				if np.max(gt) > 1.0:
					gt = gt / 255.0
				clip_gt.append(torch.FloatTensor(gt))
				
			clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))

		# added
		else:
			frame_indices = np.arange(end_idx-self.len_snippet + 1, end_idx + 1)

			temp = []
			for i in frame_indices:
				if i < 0:
					temp.append(0)
				else:
					temp.append(i)
			frame_indices = temp

			for i in frame_indices:
				img = Image.open(os.path.join(path_clip, list_clips[i])).convert('RGB')
				sz = img.size
				clip_img.append(self.img_transform(img))

			clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))
			gt = np.array(Image.open(os.path.join(path_annt, list_sal_clips[end_idx])).convert('L'))
			gt = gt.astype('float')
			if np.max(gt) > 1.0:
				gt = gt / 255.0
			clip_gt = torch.FloatTensor(gt)


			fix_map_filename = list_sal_clips[end_idx].split('/')[-1].split('.')[0]
			video_clip_name = list_sal_clips[end_idx].split('/')[-3]

			fix = sio.loadmat(join(self.path_data,video_clip_name,'fixation','maps','{}.mat'.format(fix_map_filename)))

			if self.mode == "test":
				binary_img = torch.FloatTensor(fix['I']).numpy()

		if self.mode == "train" or self.mode == "val":
			gt = torch.FloatTensor(torch.stack(clip_gt, dim=0))[-1]
			return clip_img, gt#, binary_img
		elif self.mode == "test":
			return clip_img, gt, binary_img, video_name, end_idx


class MVVADataset(Dataset):
	def __init__(self, len_snippet, dataset_name='mvva', split=1, mode='train',use_skip=True,args=None):

		self.len_snippet = len_snippet
		self.mode = mode
		self.fold_lists_path = args.fold_lists_path
		
		self.dataset = dataset_name


		self.videos_root_path = args.videos_root_path

		self.videos_frames_root_path = args.videos_frames_root_path

		self.gt_sal_maps_path = args.gt_sal_maps_path

		self.fixation_data_path = args.fixation_data_path

		self.use_skip = use_skip

		if (self.mode == 'val' or self.mode == 'test') and self.dataset == 'mvva':
			mode = 'test'
		file_name = '{}_list_{}_{}_fps.txt'.format(self.dataset, mode, split)
		print("Taking the split file: ", file_name)
		
		self.list_indata = []
		with open(join(self.fold_lists_path, file_name), 'r') as f:
			for line in f.readlines():
				name = line.split(' ')[0].strip()

				self.list_indata.append(name)

		self.video_names = [video_name for video_name in os.listdir(self.videos_frames_root_path)]

		self.video_names = [i for i in self.video_names if i in self.list_indata]
		


		self.img_transform = transforms.Compose([
			transforms.Resize((224, 384)),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.485, 0.456, 0.406],
				[0.229, 0.224, 0.225]
			)
		])
		self.audio_transform = transforms.Compose([
			transforms.Resize((224, 224)),  
			transforms.ToTensor(),           
			transforms.Normalize(            
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225]
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

			if args.video_names_list != '':

				self.video_names = [i for i in self.video_names if i in args.video_names_list]

			print("Test Video names are ",self.video_names)

			self.list_num_frame = []

			for video_name in self.video_names :
				num_frames = len(os.listdir(join(self.videos_frames_root_path, video_name)))
				for end_idx in range(0,num_frames):
					self.list_num_frame.append((video_name,num_frames,end_idx))

			print("Number of test video names are : ",len(self.video_names))
			print("Number of test samples are : ",len(self.list_num_frame))


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
					0, num_frames-self.len_snippet)


		if self.mode == 'train' or self.mode == 'val':
			clip = []
			for i in range(self.len_snippet):
				img = Image.open(os.path.join(join(self.videos_frames_root_path, video_name), 'img_%05d.jpg' % (
					start_frame+i+1))).convert('RGB')
				
				img = img.resize((1280,720))
				W,H = img.size
				img = self.img_transform(img)
				clip.append(img)

			
			clip_img = torch.FloatTensor(torch.stack(clip, dim=0))

			path_annt = os.path.join(self.gt_sal_maps_path, video_name,'maps')

			gt = np.array(Image.open(join(path_annt, 'eyeMap_%05d.jpg'%(start_frame+self.len_snippet))).convert('L'))
			gt = gt.astype('float')

			if self.mode == "train":

				gt = cv2.resize(gt, (384,224))


			if np.max(gt) > 1.0:
				gt = gt / 255.0

			########### Getting the fixation map
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


		### Getting test data			
		if self.mode == 'test':

			video_name, num_frames,end_idx = self.list_num_frame[index]


			frame_indices = np.arange(end_idx-self.len_snippet + 1, end_idx + 1)

			temp = []
			for i in frame_indices:
				if i < 0:
					temp.append(0)
				else:
					temp.append(i)
			frame_indices = temp

			clip = []
			for i in frame_indices:
				img = Image.open(os.path.join(join(self.videos_frames_root_path, video_name), 'img_%05d.jpg' % (i+1))).convert('RGB')
				
				img = img.resize((1280,720))

				W,H = img.size

				img = self.img_transform(img)				
				clip.append(img)
			clip_img = torch.FloatTensor(torch.stack(clip, dim=0))


			###################### ground truth ###################
			path_annt = os.path.join(self.gt_sal_maps_path, video_name,'maps')
			gt_sal = np.array(Image.open(join(path_annt, 'eyeMap_%05d.jpg'%(end_idx+1))).convert('L'))
			gt = gt_sal.astype('float')


			########### Getting the fixation map
			count = 0
			fix_data = np.load(join(self.fixation_data_path,video_name + '.npy'))
			binary_img = np.zeros((H // 2,W // 2),dtype=np.uint8)

			assert num_frames == fix_data.shape[0]

			for j in range(fix_data.shape[2]):
				x = fix_data[end_idx,0,j]
				y = fix_data[end_idx,1,j]
				try:           
					binary_img[int(y/2),int(x/2)] = 1
				except:
					count+=1

		if self.mode != 'test':
			return clip_img,gt,binary_img
		return clip_img,gt,binary_img,video_name,end_idx

	def __len__(self):
		if self.mode == 'train':
			return len(self.video_names)
		else:
		   return(len(self.list_num_frame))
		
