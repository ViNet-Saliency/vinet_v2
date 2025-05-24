import argparse
import gc
import glob, os
import torch
torch.use_deterministic_algorithms(True,warn_only=True)

import sys
import time
import torch.nn as nn
import pickle
from torch.autograd import Variable
from torchvision import transforms, utils
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
# from SaliencyModel.vinet_dataloader import * 
from loss import *
import cv2
from model import *
# added
from vinet_model import *

from utils_sal import *

import pdb
from collections import OrderedDict
import wandb


from EEAA_dataloader import *
# from EEAA_audio_visual_dataloader import *
# added
from vinet_dataloader_new import DHF1KDataset as dhf1k
from vinet_dataloader_new import Hollywood_UCFDataset as hollywooducf
from vinet_dataloader_new import Hollywood_ViNet_Dataset as hollywoodvinet
from vinet_dataloader_new import SoundDatasetLoader
from vinet_dataloader_new import mvva_vinet_dataset

from tqdm import tqdm

from os.path import join
import random
from collections import defaultdict 


parser = argparse.ArgumentParser()

parser.add_argument('--no_epochs',default=120, type=int)
parser.add_argument('--lr',default=1e-4, type=float)

parser.add_argument('--kldiv',default=True, type=bool)
parser.add_argument('--cc',default=True, type=bool)
parser.add_argument('--nss',default=False, type=bool)
parser.add_argument('--sim',default=False, type=bool)
parser.add_argument('--nss_emlnet',default=False, type=bool)
parser.add_argument('--nss_norm',default=False, type=bool)
parser.add_argument('--l1',default=False, type=bool)

parser.add_argument('--lr_sched',default=False, type=bool)

parser.add_argument('--scheduler',default="ReduceLROnPlateau",type=str)

parser.add_argument('--enc_model',default="SlowFast", type=str)
parser.add_argument('--optim',default="Adam", type=str)


parser.add_argument('--step_size',default=10, type=int)

parser.add_argument('--kldiv_coeff',default=1.0, type=float)
parser.add_argument('--cc_coeff',default=-1, type=float)
parser.add_argument('--sim_coeff',default=0.0, type=float)
parser.add_argument('--nss_coeff',default=0.0, type=float)


# parser.add_argument('--nss_emlnet_coeff',default=1.0, type=float)
# parser.add_argument('--nss_norm_coeff',default=1.0, type=float)
# parser.add_argument('--l1_coeff',default=1.0, type=float)
parser.add_argument('--train_enc',default=1, type=int)

parser.add_argument('--batch_size',default=8, type=int)
parser.add_argument('--log_interval',default=5, type=int)
parser.add_argument('--no_workers',default=4, type=int)


# parser.add_argument('--clip_size',default=64, type=int)
# parser.add_argument('--nhead',default=4, type=int)
# parser.add_argument('--num_encoder_layers',default=3, type=int)
# parser.add_argument('--num_decoder_layers',default=3, type=int)

parser.add_argument('--decoder_groups', default=32, type=int)
# parser.add_argument('--transformer_in_channel',default=32, type=int)
parser.add_argument('--train_path_data',default="/home/sid/DHF1K/annotation", type=str)
parser.add_argument('--val_path_data',default="/home/sid/DHF1K/val", type=str)

parser.add_argument('--decoder_upsample',default=1, type=int)

# parser.add_argument('--frame_no',default="last", type=str)
# parser.add_argument('--load_weight',default="None", type=str)

# parser.add_argument('--num_hier',default=3, type=int)

parser.add_argument('--dataset',default="Hollywood2", type=str)

# parser.add_argument('--alternate',default=1, type=int)
# parser.add_argument('--spatial_dim',default=-1, type=int)
parser.add_argument('--split',default='random1', type=str)
parser.add_argument('--use_audio',default=False, type=bool)
# parser.add_argument('--use_transformer',default=False, type=bool)
# parser.add_argument('--use_vox',default=False, type=bool)

# added

parser.add_argument('--use_skip', default=1, type=int)

parser.add_argument('--neck_name', default='neck', type=str)

parser.add_argument('--videos_root_path', default='neck2', type=str)

parser.add_argument('--videos_frames_root_path', default='', type=str)

parser.add_argument('--audio_maps_path', default='', type=str)

parser.add_argument('--use_feature_masking',default=False, type=bool)

parser.add_argument('--use_decoder_feature_masking',default=False, type=bool)

parser.add_argument('--use_input_masking',default=False, type=bool)

parser.add_argument('--use_pairs',default=False, type=bool)

parser.add_argument('--len_snippet',default=32, type=int)

parser.add_argument('--window_length',default=64, type=int)

parser.add_argument('--video_frame_class_file_path', default='', type=str)

parser.add_argument('--fixation_data_path', default='', type=str)

parser.add_argument('--gt_sal_maps_path', default='', type=str)

parser.add_argument('--alternate',default=2, type=int)

parser.add_argument('--fold_lists_path', default='', type=str)

parser.add_argument('--model_save_root_path',default="/home/girmaji08/EEAA/SaliencyModel/saved_models/", type=str)

parser.add_argument('--videos_list_path', default='/home/girmaji08/EEAA/SaliencyModel/videos_list.npy', type=str)

parser.add_argument('--use_extra_data',default=0, type=int)

parser.add_argument('--load_weight',default="None", type=str)

parser.add_argument('--test_annotations_path', default='/home/sid/SaliencyModel/EEAA/SaliencyModel/annotations/mvva_train_annotations_64.pickle', type=str)

parser.add_argument('--use_triplet_attention',default=False, type=bool)

parser.add_argument('--use_cross_attention_in_decoder',default=False, type=bool)

# use_multiscale_cross_attention
parser.add_argument('--use_multiscale_cross_attention',default=False, type=bool)

#use_channel_shuffle
parser.add_argument('--use_channel_shuffle',default=True, type=bool)

#use_action
parser.add_argument('--use_action',default=False, type=bool)

#compute_metrics
parser.add_argument('--compute_metrics',default=True, type=bool)

#checkpoint path
parser.add_argument('--checkpoint_path_1',default='', type=str)
parser.add_argument('--checkpoint_path_2',default='', type=str)
# parser.add_argument('--checkpoint_path_3',default='', type=str)
# parser.add_argument('--checkpoint_path_4',default='', type=str)

#save_inferences
parser.add_argument('--save_inferences',default=False,type=bool)

#save_path
parser.add_argument('--save_path',default=None, type=str)

#model tag
parser.add_argument('--model_tag',default='16x29_withproj_TD_BT_2gpus', type=str)

#metrics_save_path
parser.add_argument('--metrics_save_path',default=None, type=str)

#BiFPN
parser.add_argument('--use_bifpn',default=False, type=bool)

#use_decoder_v2
parser.add_argument('--use_decoder_v2',default=0,type=int)

#subset_type
parser.add_argument('--subset_type',default='human', type=str)


#modelA_path
parser.add_argument('--modelA_path',default='', type=str)

#modelB_path
parser.add_argument('--modelB_path',default='', type=str)

#reload_data_every_epoch
parser.add_argument('--reload_data_every_epoch',default=0, type=int)

#use_image_saliency
parser.add_argument('--use_image_saliency',default=0, type=int)

#use_action_classification
parser.add_argument('--use_action_classification',default=0, type=int)

#video_names_list
parser.add_argument('--video_names_list',default='', type=str)
args = parser.parse_args()
print(args)

# added
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

if args.dataset == 'mvva':
	seed = 1100
	np.random.seed(seed)
	torch.manual_seed(seed)
	random.seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
if args.dataset == 'Hollywood2':
	seed = 867
	np.random.seed(seed)
	torch.manual_seed(seed)
	random.seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

flags_list = [(args.use_audio,'audio'),(args.use_input_masking,'input_masking'),(args.use_feature_masking,'feature_masking'),(args.use_pairs,'pairs'),(args.use_triplet_attention,'triplet_attention'),(args.use_cross_attention_in_decoder,'cross_attention_decoder'),(args.use_multiscale_cross_attention,'multiscale_cross_attention'),(args.use_decoder_feature_masking,'decoder_feature_masking'),(args.use_channel_shuffle,'channel_shuffle')]

flags_used = []
for i in range(len(flags_list)):
	if flags_list[i][0] == True:
		flags_used.append(flags_list[i][1])

if args.videos_list_path != '' and args.use_extra_data == 1:
	flags_used.append('extra_data')

flags_used = '_'.join(flags_used)



model_name_1 = args.checkpoint_path_1.split('/')[-1].split('.')[0]
model_name_2 = args.checkpoint_path_2.split('/')[-1].split('.')[0]
# model_name_3 = args.checkpoint_path_3.split('/')[-1].split('.')[0]
# model_name_4 = args.checkpoint_path_4.split('/')[-1].split('.')[0]
model_name = model_name_1 + model_name_2
wandb.login(key="947c6d8652e2600fbb5bc6f519588ea095620e6b")

# wandb.init()
wandb.init(
	project='EEAA-B',
	name= 'test_EEAA-B_%s' % (model_name),
	config={
		'batch_size': args.batch_size,
		'dataset': args.dataset,
		'decoder_groups': args.decoder_groups,
		'no_epochs': args.no_epochs,
		'lr': args.lr,
		'lr_sched': args.lr_sched,
		'scheduler': args.scheduler,
		'num_GPUs': torch.cuda.device_count(),
		'optim': args.optim,
		'neck_name': args.neck_name,
		'split': args.split,
		'use_audio': args.use_audio,
		'use_input_masking': args.use_input_masking,
		'use_feature_masking': args.use_feature_masking,
		'nss_coeff': args.nss_coeff,
		'cc_coeff': args.cc_coeff,
		'sim_coeff': args.sim_coeff,
		'kldiv_coeff': args.kldiv_coeff,
		'kldiv': args.kldiv,
		'nss': args.nss,
		'cc': args.cc,
		'sim': args.sim,
		'seed': seed
	}
)

model_1 = EEAA(args)
model_2 = VideoSaliencyModel()
# model_3 = VideoSaliencyModel()
# model_4 = VideoSaliencyModel()


if args.dataset == "DHF1KDataset":
	train_dataset = DHF1KDataset(args.train_path_data, args.clip_size, mode="train", alternate=args.alternate, use_skip=bool(args.use_skip))
	val_dataset = DHF1KDataset(args.val_path_data, args.clip_size, mode="val", alternate=args.alternate, use_skip=bool(args.use_skip))

elif args.dataset=="SoundDataset":
	train_dataset_diem = SoundDatasetLoader(args.clip_size, mode="train", dataset_name='DIEM', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)
	val_dataset_diem = SoundDatasetLoader(args.clip_size, mode="test", dataset_name='DIEM', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)

	train_dataset_coutrout1 = SoundDatasetLoader(args.clip_size, mode="train", dataset_name='Coutrot_db1', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)
	val_dataset_coutrout1 = SoundDatasetLoader(args.clip_size, mode="test", dataset_name='Coutrot_db1', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)

	train_dataset_coutrout2 = SoundDatasetLoader(args.clip_size, mode="train", dataset_name='Coutrot_db2', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)
	val_dataset_coutrout2 = SoundDatasetLoader(args.clip_size, mode="test", dataset_name='Coutrot_db2', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)
	
	train_dataset_avad = SoundDatasetLoader(args.clip_size, mode="train", dataset_name='AVAD', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)
	val_dataset_avad = SoundDatasetLoader(args.clip_size, mode="test", dataset_name='AVAD', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)

	train_dataset_etmd = SoundDatasetLoader(args.clip_size, mode="train", dataset_name='ETMD_av', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)
	val_dataset_etmd = SoundDatasetLoader(args.clip_size, mode="test", dataset_name='ETMD_av', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)

	train_dataset_summe = SoundDatasetLoader(args.clip_size, mode="train", dataset_name='SumMe', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)
	val_dataset_summe = SoundDatasetLoader(args.clip_size, mode="test", dataset_name='SumMe', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)
	
	train_dataset = torch.utils.data.ConcatDataset([
				train_dataset_diem, train_dataset_coutrout1,
				train_dataset_coutrout2, 
				train_dataset_avad, train_dataset_etmd,
				train_dataset_summe 
		])

	val_dataset = torch.utils.data.ConcatDataset([
				val_dataset_diem, val_dataset_coutrout1,
				val_dataset_coutrout2, 
				val_dataset_avad, val_dataset_etmd,
				val_dataset_summe 
		])
elif args.dataset == "AVAD":
	print("Using AVAD dataset")
	# train_dataset = AVAD_DataLoader(args, mode="train")
	# val_dataset = AVAD_DataLoader(args, mode="val")
	# test_dataset = AVAD_DataLoader(args, mode="test")
	train_dataset = Other_AudioVisual_DataLoader(args,'AVAD', mode="train")
	val_dataset = Other_AudioVisual_DataLoader(args,'AVAD', mode="val")
	test_dataset = Other_AudioVisual_DataLoader(args,'AVAD', mode="test")
	train_vinet = SoundDatasetLoader(args.len_snippet, dataset_name=args.dataset, split=args.split, mode="train")
	val_vinet = SoundDatasetLoader(args.len_snippet, dataset_name=args.dataset, split=args.split, mode="val")
	test_vinet = SoundDatasetLoader(32, dataset_name=args.dataset, split=args.split, mode="test")
elif args.dataset == "Coutrot_db1":
	print("Using Coutrot1 dataset")
	# train_dataset = Sound_DataLoader(args, dataset=args.dataset, mode="train")
	# val_dataset = Sound_DataLoader(args, dataset=args.dataset, mode="val")
	# test_dataset = Sound_DataLoader(args, dataset=args.dataset, mode="test")
	train_dataset = Other_AudioVisual_DataLoader(args,'Coutrot_db1', mode="train")
	val_dataset = Other_AudioVisual_DataLoader(args,'Coutrot_db1', mode="val")
	test_dataset = Other_AudioVisual_DataLoader(args,'Coutrot_db1', mode="test")
	train_vinet = SoundDatasetLoader(args.len_snippet, dataset_name=args.dataset, split=args.split, mode="train")
	val_vinet = SoundDatasetLoader(args.len_snippet, dataset_name=args.dataset, split=args.split, mode="val")
	test_vinet = SoundDatasetLoader(32, dataset_name=args.dataset, split=args.split, mode="test")
elif args.dataset == "Coutrot_db2":
	print("Using Coutrot2 dataset")
	# train_dataset = Sound_DataLoader(args, dataset=args.dataset, mode="train")
	# val_dataset = Sound_DataLoader(args, dataset=args.dataset, mode="val")
	# test_dataset = Sound_DataLoader(args, dataset=args.dataset, mode="test")
	train_dataset = Other_AudioVisual_DataLoader(args,'Coutrot_db2', mode="train")
	val_dataset = Other_AudioVisual_DataLoader(args,'Coutrot_db2', mode="val")
	test_dataset = Other_AudioVisual_DataLoader(args,'Coutrot_db2', mode="test")
	train_vinet = SoundDatasetLoader(args.len_snippet, dataset_name=args.dataset, split=args.split, mode="train")
	val_vinet = SoundDatasetLoader(args.len_snippet, dataset_name=args.dataset, split=args.split, mode="val")
	test_vinet = SoundDatasetLoader(32, dataset_name=args.dataset, split=args.split, mode="test")
elif args.dataset == "DHF1K":
	print("Using DHF1K dataset")
	train_dataset = DHF1KDataset(args, mode="train")
	val_dataset = DHF1KDataset(args, mode="val")
	test_dataset = DHF1KDataset(args, mode="test")
	train_vinet = dhf1k(args.train_path_data, args.len_snippet, mode="train",args=args)
	val_vinet = dhf1k(args.val_path_data, args.len_snippet, mode="val",args=args)
	test_vinet = dhf1k(args.val_path_data, 32, mode="test",args=args)
elif args.dataset == "DIEM":
	print("Using DIEM dataset")
	train_dataset = DIEM_DataLoader(args, mode="train")
	val_dataset = DIEM_DataLoader(args, mode="val")
	test_dataset = DIEM_DataLoader(args, mode="test")
	train_vinet = SoundDatasetLoader(32, dataset_name=args.dataset, split=args.split, mode="train",args=args)
	val_vinet = SoundDatasetLoader(32, dataset_name=args.dataset, split=args.split, mode="val",args=args)
	test_vinet = SoundDatasetLoader(32, dataset_name=args.dataset, split=args.split, mode="test",args=args)
elif args.dataset == "ETMD_av":
	print("Using ETMD dataset")
	# train_dataset = Sound_DataLoader(args, dataset=args.dataset, mode="train")
	# val_dataset = Sound_DataLoader(args, dataset=args.dataset, mode="val")
	# test_dataset = Sound_DataLoader(args, dataset=args.dataset, mode="test")
	train_dataset = ETMD_av_DataLoader(args,'ETMD_av', mode="train")
	val_dataset = ETMD_av_DataLoader(args,'ETMD_av', mode="val")
	test_dataset = ETMD_av_DataLoader(args,'ETMD_av', mode="test")
	train_vinet = SoundDatasetLoader(args.len_snippet, dataset_name=args.dataset, split=args.split, mode="train")
	val_vinet = SoundDatasetLoader(args.len_snippet, dataset_name=args.dataset, split=args.split, mode="val")
	test_vinet = SoundDatasetLoader(32, dataset_name=args.dataset, split=args.split, mode="test")
elif args.dataset == "mvva":
	print("Using MVVA dataset")
	train_dataset= mvva_dataset(args, mode="train")
	val_dataset = mvva_dataset(args, mode="val")
	test_dataset = mvva_dataset(args, mode="test")  

	train_vinet = mvva_vinet_dataset(32, dataset_name='mvva', split=1, mode='train',use_skip=True)
	val_vinet = mvva_vinet_dataset(32, dataset_name='mvva', split=1, mode='val',use_skip=True)
	test_vinet = mvva_vinet_dataset(32, dataset_name='mvva', split=1, mode='test',use_skip=True)  
elif args.dataset == "Hollywood2":
	print("Using Hollywood2 dataset")
	train_dataset = Hollywood2_Dataset(args, mode="train")
	val_dataset = Hollywood2_Dataset(args, mode="val")
	test_dataset = Hollywood2_Dataset(args, mode="test")
	# train_vinet = hollywooducf(args.train_path_data, args.len_snippet, mode="train")
	# val_vinet = hollywooducf(args.val_path_data, args.len_snippet, mode="val")
	# test_vinet = hollywooducf(args.val_path_data, 32, mode="test")

	train_vinet = hollywoodvinet(args.train_path_data,32, mode="train")
	val_vinet = hollywoodvinet(args.val_path_data, 32, mode="val")
	test_vinet = hollywoodvinet(args.val_path_data, 32, mode="test")
elif args.dataset == "UCF":
	print("Using UCF Sports dataset")
	train_dataset = UCF_DataLoader_orig(args, mode="train")
	val_dataset = UCF_DataLoader_orig(args, mode="val")
	test_dataset = UCF_DataLoader_orig(args, mode="test")
	train_vinet = hollywooducf(args.train_path_data, args.len_snippet, mode="train",args=args)
	val_vinet = hollywooducf(args.val_path_data, args.len_snippet, mode="val",args=args)
	test_vinet = hollywooducf(args.val_path_data, 32, mode="test",args=args)


print("Loading the dataset...")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.no_workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.no_workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)
train_vinet_loader = torch.utils.data.DataLoader(train_vinet, batch_size=args.batch_size, shuffle=True, num_workers=args.no_workers)
val_vinet_loader = torch.utils.data.DataLoader(val_vinet, batch_size=args.batch_size, shuffle=False, num_workers=args.no_workers)
test_vinet_loader = torch.utils.data.DataLoader(test_vinet, batch_size=1, shuffle=False, num_workers=args.no_workers)


if args.checkpoint_path_1 != '':
	print("Loading pretrained EEAA model weights if any: ",args.checkpoint_path_1)
	model_1.load_state_dict(torch.load(args.checkpoint_path_1))
	
if args.checkpoint_path_2 != '':
	print("Loading pretrained ViNet (36MB) weights if any: ",args.checkpoint_path_2)
	model_2.load_state_dict(torch.load(args.checkpoint_path_2))

# if args.checkpoint_path_3 != '':
# 	print("Loading pretrained ViNet (36MB) weights if any: ",args.checkpoint_path_3)
# 	model_1.load_state_dict(torch.load(args.checkpoint_path_3))
	
# if args.checkpoint_path_4 != '':
# 	print("Loading pretrained ViNet (36MB) weights if any: ",args.checkpoint_path_4)
# 	model_2.load_state_dict(torch.load(args.checkpoint_path_4))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
	print("Let's use", torch.cuda.device_count(), "GPUs!")
	model_1 = nn.DataParallel(model_1)
	model_2 = nn.DataParallel(model_2)
model_1.to(device)
model_2.to(device)
# model_3.to(device)
# model_4.to(device)

print(device)


def test(model_1, model_2, loader_1, loader_2, device, args):
	model_1.eval()
	model_2.eval()


	video_kldiv_loss = defaultdict(int)
	video_cc_loss = defaultdict(int)
	video_sim_loss = defaultdict(int)
	video_nss_loss = defaultdict(int)
	video_aucj_loss = defaultdict(int)
	video_num_frames = defaultdict(int)

	num_frames = 0
	total_time = 0
	
	nan_counts = 0

	for idx, sample in tqdm(enumerate(zip(loader_1, loader_2))):

		num_frames += 1

		# getting the input

		# if idx >= 100:
		# 	break
		img_clips_eeaa, img_clips_vinet = sample[0][0], sample[1][0]
		gt_sal_eeaa, gt_sal_vinet = sample[0][1], sample[1][1]
		binary_img_eeaa, binary_img_vinet = sample[0][2], sample[1][2]
		video_name_eeaa, video_name_vinet = sample[0][3][0], sample[1][3][0]
		mid_frame_eeaa, mid_frame_vinet = sample[0][4], sample[1][4]
		# pdb.set_trace()
		# if video_name_eeaa != video_name_vinet:
		# 	print(video_name_eeaa, video_name_vinet)
		assert gt_sal_eeaa.shape == gt_sal_vinet.shape
		assert binary_img_eeaa.shape == binary_img_vinet.shape
		# # print(video_name_eeaa, video_name_vinet)
		assert video_name_eeaa == video_name_vinet
		assert mid_frame_eeaa == mid_frame_vinet
		if args.use_audio:
			audio_feature = sample[3].to(device)
		img_clips_eeaa = img_clips_eeaa.to(device)
		img_clips_vinet = img_clips_vinet.to(device)
		img_clips_eeaa = img_clips_eeaa.permute((0,2,1,3,4))
		img_clips_vinet = img_clips_vinet.permute((0,2,1,3,4))

		# doing inference
		
		if args.use_audio:
			pred_sal_1 = model_1(img_clips_eeaa, audio_feature)
			pred_sal_2 = model_2(img_clips_vinet, audio_feature)
		else:
			# start_time = time.time()
			pred_sal_1 = model_1(img_clips_eeaa)
			pred_sal_2 = model_2(img_clips_vinet)
			# pred_sal_3 = model_1(img_clips)
			# pred_sal_4 = model_2(img_clips)
			# total_time += time.time() - start_time
			# continue

		# averaging the saliency maps
		pred_sal_2 = F.interpolate(pred_sal_2.unsqueeze(0),size=(256,456),mode = 'bicubic',align_corners = False).squeeze(0)
		stacked_sal_maps = torch.stack((pred_sal_1, pred_sal_2), dim=0)
		pred_sal = torch.mean(stacked_sal_maps, dim=0)

		# post processing of output
		gt_sal = gt_sal_eeaa.squeeze(0).numpy()

		pred_sal = pred_sal.cpu().squeeze(0).numpy()
		pred_sal = cv2.resize(pred_sal, (gt_sal.shape[1], gt_sal.shape[0]))
		pred_sal = blur(pred_sal).unsqueeze(0).cuda()


		
		if args.save_inferences and args.save_path is not None:
			os.makedirs(join(args.save_path, 'inferences','ViNet-E',video_name_eeaa),exist_ok=True)
			img_save(pred_sal, join(args.save_path,'inferences','ViNet-E', video_name_eeaa, 'img_%05d.png'%(mid_frame_eeaa+1)), normalize=True)
			continue
		gt_sal = torch.FloatTensor(gt_sal).unsqueeze(0).cuda()

		assert pred_sal.size() == gt_sal.size()

		# metrics computation

		if args.compute_metrics:
			cc_loss = cc(pred_sal, gt_sal)

		
			kldiv_loss = kldiv(pred_sal, gt_sal)

			sim_loss = similarity(pred_sal, gt_sal)

			# print('pred_sal: ', pred_sal.shape)
			# print('binary_img: ', binary_img.shape)
			aucj_loss = auc_judd(pred_sal.cpu(), binary_img_eeaa.cpu())

			if isinstance(aucj_loss, float):
				aucj_loss = torch.FloatTensor([aucj_loss])


			nss_loss = torch.FloatTensor([0.0]).cuda()
			for i in range(len(binary_img_eeaa)):
				# print('pred_sal: ', pred_sal[i,:,:].unsqueeze(0).shape)
				# print('binary_img: ', binary_img[i].unsqueeze(0).shape)
				nss_loss += nss(pred_sal[i,:,:].unsqueeze(0).detach().to(device), binary_img_eeaa[i].unsqueeze(0).to(device))
					
			nss_loss = nss_loss/len(binary_img_eeaa)


			if np.isnan(cc_loss.item()):
				cc_loss = torch.FloatTensor([0.0]).cuda()
				nan_counts += 1
			if np.isnan(kldiv_loss.item()):
				kldiv_loss = torch.FloatTensor([0.0]).cuda()
				nan_counts += 1
			if np.isnan(sim_loss.item()):
				sim_loss = torch.FloatTensor([0.0]).cuda()
				nan_counts += 1
			if np.isnan(nss_loss.item()):
				nss_loss = torch.FloatTensor([0.0]).cuda()
				nan_counts += 1
			if np.isnan(aucj_loss.item()):
				aucj_loss = torch.FloatTensor([0.0]).cuda()
				nan_counts += 1

			video_kldiv_loss[video_name_eeaa] += kldiv_loss.item()
			video_cc_loss[video_name_eeaa] += cc_loss.item()
			video_sim_loss[video_name_eeaa] += sim_loss.item()
			video_nss_loss[video_name_eeaa] += nss_loss.item()
			video_aucj_loss[video_name_eeaa] += aucj_loss.item()
			video_num_frames[video_name_eeaa] += 1

	# print("Total time taken to process the video with {} is {}".format(num_frames, total_time))

	return None, None, None, None, None, None


	# print("Total number of NaNs: ", nan_counts)
	# return video_kldiv_loss, video_cc_loss, video_sim_loss, video_nss_loss, video_aucj_loss, video_num_frames




with torch.no_grad():
	video_kldiv_loss, video_cc_loss, video_sim_loss, video_nss_loss, video_aucj_loss, video_num_frames = test(model_1, model_2, test_loader, test_vinet_loader, device, args)
	# pdb.set_trace()

# getting per video metrics
# video_metrics_dict = defaultdict(dict)

# for video_name in video_kldiv_loss.keys():

# 	video_metrics_dict[video_name]['kldiv_loss'] = video_kldiv_loss[video_name]/video_num_frames[video_name]
# 	video_metrics_dict[video_name]['cc_loss'] = video_cc_loss[video_name]/video_num_frames[video_name]
# 	video_metrics_dict[video_name]['sim_loss'] = video_sim_loss[video_name]/video_num_frames[video_name]
# 	video_metrics_dict[video_name]['nss_loss'] = video_nss_loss[video_name]/video_num_frames[video_name]
# 	video_metrics_dict[video_name]['aucj_loss'] = video_aucj_loss[video_name]/video_num_frames[video_name]


# # getting full test/val set metrics
# video_metrics_dict['full_metrics']['test_avg_loss']=np.sum(list(video_kldiv_loss.values()))/np.sum(list(video_num_frames.values()))
# video_metrics_dict['full_metrics']['test_cc_loss']=np.sum(list(video_cc_loss.values()))/np.sum(list(video_num_frames.values()))
# video_metrics_dict['full_metrics']['test_sim_loss']=np.sum(list(video_sim_loss.values()))/np.sum(list(video_num_frames.values()))
# video_metrics_dict['full_metrics']['test_nss_loss']=np.sum(list(video_nss_loss.values()))/np.sum(list(video_num_frames.values()))
# video_metrics_dict['full_metrics']['test_aucj_loss']=np.sum(list(video_aucj_loss.values()))/np.sum(list(video_num_frames.values()))


# # saving the video_metrics_dict
# print("model nmame is ", model_name)
# print("metrics save path is ", join(args.metrics_save_path,args.dataset))

# os.makedirs(join(args.metrics_save_path,args.dataset),exist_ok=True)
# #


# r = json.dumps(video_metrics_dict,indent=4)

# with open(join(args.metrics_save_path,args.dataset, model_name + '_split' + str(args.split) + '_' + 'video_metrics_dict.json'), 'w') as f:
# 	f.write(r)
	

# # logging the metrics to wandb

# wandb.log({'test_kldiv_loss':np.sum(list(video_kldiv_loss.values()))/np.sum(list(video_num_frames.values()))})
# wandb.log({'test_cc_loss':np.sum(list(video_cc_loss.values()))/np.sum(list(video_num_frames.values()))})
# wandb.log({'test_sim_loss':np.sum(list(video_sim_loss.values()))/np.sum(list(video_num_frames.values()))})
# wandb.log({'test_nss_loss':np.sum(list(video_nss_loss.values()))/np.sum(list(video_num_frames.values()))})
# wandb.log({'test_aucj_loss':np.sum(list(video_aucj_loss.values()))/np.sum(list(video_num_frames.values()))})