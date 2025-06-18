import argparse
import gc
import glob
import os
import sys
import time
import random
import copy
import pickle
import pdb

import torch
torch.use_deterministic_algorithms(True, warn_only=True)

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
import cv2
from PIL import Image
from torchvision import transforms, utils

from collections import OrderedDict, defaultdict

from tqdm import tqdm
from os.path import join

import wandb

from loss import *
from ViNet_A_model import *
from utils_sal import *
from ViNet_A_visual_dataloader import *
from ViNet_A_audio_visual_dataloader import *


def list_of_strings(string):
	return string.split(',')

parser = argparse.ArgumentParser()


parser.add_argument('--no_workers',default=4, type=int)



parser.add_argument('--decoder_groups', default=32, type=int)



parser.add_argument('--dataset',default="mvva", type=str)

parser.add_argument('--split',default='no_split', type=str)

# added

parser.add_argument('--use_skip', default=1, type=int)

parser.add_argument('--neck_name', default='neck', type=str)

parser.add_argument('--videos_root_path', default='', type=str)

parser.add_argument('--videos_frames_root_path', default='', type=str)



parser.add_argument('--len_snippet',default=32, type=int)


parser.add_argument('--fixation_data_path', default='', type=str)

parser.add_argument('--gt_sal_maps_path', default='', type=str)

parser.add_argument('--fold_lists_path', default='', type=str)




#checkpoint_path
parser.add_argument('--checkpoint_path',default="None", type=str)



#use_channel_shuffle
parser.add_argument('--use_channel_shuffle',default=True, type=bool)


#model_tag
parser.add_argument('--model_tag',default='', type=str)

#model_name

parser.add_argument('--model_name',default='', type=str)

#save_inferences
parser.add_argument('--use_action_classification',default=0, type=int)

#save_inferences
parser.add_argument('--save_inferences',default=0, type=int)

#save_path

parser.add_argument('--save_path',default= None, type=str)

#metrics_save_path

parser.add_argument('--metrics_save_path',default= '~/ICASSP_Saliency/metrics', type=str)

#compute_metrics

parser.add_argument('--compute_metrics',default= 1, type=int)

parser.add_argument('--video_names_list',default='', type=list_of_strings)


args = parser.parse_args()
print(args)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


model_tag = args.model_tag + 'ViNet_A'
	

model = ViNet_A(args)




if args.dataset == "DHF1K":

	test_dataset = DHF1K_Dataset(args,mode='test')



elif args.dataset == "UCF":
	print("Using UCF dataset")

	test_dataset = UCF_Dataset(args,mode='test')

elif args.dataset == "Hollywood2":
	print("Using Hollywood2 dataset")

	test_dataset = Hollywood2_Dataset(args,mode='test')


elif args.dataset == "mvva":
	print("Using MVVA dataset")

	test_dataset = MVVA_Dataset(args,mode = "test")
  


elif args.dataset == "SumMe":
	print("Using SumMe dataset")

	test_dataset = Other_AudioVisual_Dataset(args,dataset='SumMe', mode="test")

elif args.dataset == "Coutrot_db2":
	print("Using Coutrot_db2 dataset")

	test_dataset = Other_AudioVisual_Dataset(args,dataset='Coutrot_db2', mode="test")

elif args.dataset == "Coutrot_db1":
	print("Using Coutrot_db1 dataset")

	test_dataset = Other_AudioVisual_Dataset(args,dataset='Coutrot_db1', mode="test")

elif args.dataset == "ETMD_av":
	print("Using ETMD dataset")

	test_dataset = Other_AudioVisual_Dataset(args,dataset='ETMD_av', mode="test")

elif args.dataset == "DIEM":
	print("Using DIEM dataset")

	test_dataset = Other_AudioVisual_Dataset(args,dataset='DIEM', mode="test")




print("Loading the dataset...")

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loading the model...


if args.checkpoint_path!="None":
	print("Loading pretrained {} model weights if any: {}".format(model_tag,args.checkpoint_path))
	model.load_state_dict(torch.load(os.path.expanduser(args.checkpoint_path), map_location=device),strict=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
	print("Let's use", torch.cuda.device_count(), "GPUs!")
	teacher_model = nn.DataParallel(model)
model.to(device)

model_name = args.checkpoint_path.split('/')[-1].split('.')[0]

print(device)


def test(model, loader, device, args):
	model.eval()


	video_kldiv_loss = defaultdict(int)
	video_cc_loss = defaultdict(int)
	video_sim_loss = defaultdict(int)
	video_nss_loss = defaultdict(int)
	video_aucj_loss = defaultdict(int)
	video_num_frames = defaultdict(int)

	total_time = 0
	num_frames = 0
	
	kldiv_nan_counts,cc_nan_counts,sim_nan_counts,nss_nan_counts,aucj_nan_counts = 0,0,0,0,0

	nan_counts = {"KLDiv": 0, "CC": 0, "SIM": 0, "NSS": 0, "AUCj": 0}

	for idx, sample in tqdm(enumerate(loader)):

		# getting the input
		num_frames += 1

		img_clips = sample[0]
		gt_sal = sample[1]
		binary_img = sample[2]
		video_name = sample[3][0]
		mid_frame = sample[4]


		img_clips = img_clips.to(device)
		img_clips = img_clips.permute((0,2,1,3,4))


		pred_sal = model(img_clips)
	

		# post processing of output
		
		gt_sal = gt_sal.squeeze(0).numpy()

		pred_sal = pred_sal.cpu().squeeze(0).numpy()
		pred_sal = cv2.resize(pred_sal, (gt_sal.shape[1], gt_sal.shape[0]))
		pred_sal = blur(pred_sal).unsqueeze(0).cuda()


		
		if args.save_inferences and args.save_path is not None:
			os.makedirs(join(args.save_path, 'inferences',args.dataset,'ViNet_A',video_name),exist_ok=True)
			img_save(pred_sal, join(args.save_path,'inferences',args.dataset,'ViNet_A', video_name, 'img_%05d.png'%(mid_frame+1)), normalize=True)
			

		gt_sal = torch.FloatTensor(gt_sal).unsqueeze(0).cuda()

		assert pred_sal.size() == gt_sal.size()

		# metrics computation

		if args.compute_metrics:
			cc_loss = cc(pred_sal, gt_sal)

		
			kldiv_loss = kldiv(pred_sal, gt_sal)

			sim_loss = similarity(pred_sal, gt_sal)

			aucj_loss = auc_judd(pred_sal.cpu(), binary_img.cpu())

			if isinstance(aucj_loss, float):
				aucj_loss = torch.FloatTensor([aucj_loss])


			nss_loss = torch.FloatTensor([0.0]).cuda()
			for i in range(len(binary_img)):

				nss_loss += nss(pred_sal[i,:,:].unsqueeze(0).detach().to(device), binary_img[i].unsqueeze(0).to(device))
					
			nss_loss = nss_loss/len(binary_img)


			if np.isnan(cc_loss.item()):
				cc_loss = torch.FloatTensor([0.0]).cuda()
				cc_nan_counts += 1
			if np.isnan(kldiv_loss.item()):
				kldiv_loss = torch.FloatTensor([0.0]).cuda()
				kldiv_nan_counts += 1
			if np.isnan(sim_loss.item()):
				sim_loss = torch.FloatTensor([0.0]).cuda()
				sim_nan_counts += 1
			if np.isnan(nss_loss.item()):
				nss_loss = torch.FloatTensor([0.0]).cuda()
				nss_nan_counts += 1
			if np.isnan(aucj_loss.item()):
				aucj_loss = torch.FloatTensor([0.0]).cuda()
				aucj_nan_counts += 1

			

			video_kldiv_loss[video_name] += kldiv_loss.item()
			video_cc_loss[video_name] += cc_loss.item()
			video_sim_loss[video_name] += sim_loss.item()
			video_nss_loss[video_name] += nss_loss.item()
			video_aucj_loss[video_name] += aucj_loss.item()
			video_num_frames[video_name] += 1

	nan_counts = {"KLDiv": kldiv_nan_counts, "CC": cc_nan_counts, "SIM": sim_nan_counts, "NSS": nss_nan_counts, "AUCj": aucj_nan_counts}

	print("Total number of NaNs: ", nan_counts)

	return video_kldiv_loss, video_cc_loss, video_sim_loss, video_nss_loss, video_aucj_loss, video_num_frames




with torch.no_grad():
	video_kldiv_loss, video_cc_loss, video_sim_loss, video_nss_loss, video_aucj_loss, video_num_frames = test(model,test_loader, device, args)


# getting per video metrics
video_metrics_dict = defaultdict(dict)

for video_name in video_kldiv_loss.keys():

	video_metrics_dict[video_name]['kldiv_loss'] = video_kldiv_loss[video_name]/video_num_frames[video_name]
	video_metrics_dict[video_name]['cc_loss'] = video_cc_loss[video_name]/video_num_frames[video_name]
	video_metrics_dict[video_name]['sim_loss'] = video_sim_loss[video_name]/video_num_frames[video_name]
	video_metrics_dict[video_name]['nss_loss'] = video_nss_loss[video_name]/video_num_frames[video_name]
	video_metrics_dict[video_name]['aucj_loss'] = video_aucj_loss[video_name]/video_num_frames[video_name]


# getting full test/val set metrics
video_metrics_dict['full_metrics']['test_avg_loss']=np.sum(list(video_kldiv_loss.values()))/np.sum(list(video_num_frames.values()))
video_metrics_dict['full_metrics']['test_cc_loss']=np.sum(list(video_cc_loss.values()))/np.sum(list(video_num_frames.values()))
video_metrics_dict['full_metrics']['test_sim_loss']=np.sum(list(video_sim_loss.values()))/np.sum(list(video_num_frames.values()))
video_metrics_dict['full_metrics']['test_nss_loss']=np.sum(list(video_nss_loss.values()))/np.sum(list(video_num_frames.values()))
video_metrics_dict['full_metrics']['test_aucj_loss']=np.sum(list(video_aucj_loss.values()))/np.sum(list(video_num_frames.values()))


# saving the video_metrics_dict
print("model nmame is ", model_name)
print("metrics save path is ", join(args.metrics_save_path,args.dataset))

os.makedirs(join(os.path.expanduser(args.metrics_save_path),args.dataset),exist_ok=True)
#


r = json.dumps(video_metrics_dict,indent=4)

with open(join(os.path.expanduser(args.metrics_save_path),args.dataset, f"{model_name}_{str(args.split)}_video_metrics_dict.json"), 'w') as f:
	f.write(r)
	