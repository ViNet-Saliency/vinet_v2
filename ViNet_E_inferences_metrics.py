
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

from loss import *
import cv2

import pdb

import wandb

from tqdm import tqdm

from os.path import join
import random
from collections import defaultdict 
import copy

sys.path.append('./ViNet_S')
sys.path.append('./ViNet_A')
from ViNet_S import ViNet_S_model, ViNet_S_dataloader, utils
from ViNet_A import ViNet_A_model, ViNet_A_visual_dataloader, ViNet_A_audio_visual_dataloader, utils_sal

from ViNet_S_model import *
from utils import *
from ViNet_S_dataloader import OtherAudioVisualDataset as vinet_s_av, UCFDataset as vinet_s_ucf, HollywoodDataset as vinet_s_holly, MVVADataset as vinet_s_mvva, DHF1KDataset as vinet_s_dhf1k

from ViNet_A_model import *
from utils_sal import *
from ViNet_A_visual_dataloader import DHF1K_Dataset as vinet_a_dhf1k, Hollywood2_Dataset as vinet_a_holly, UCF_Dataset as vinet_a_ucf
from ViNet_A_audio_visual_dataloader import MVVA_Dataset as vinet_a_mvva, Other_AudioVisual_Dataset as vinet_a_av




parser = argparse.ArgumentParser()

parser.add_argument('--batch_size',default=1, type=int)

parser.add_argument('--no_workers',default=4, type=int)


parser.add_argument('--clip_size',default=32, type=int)


parser.add_argument('--decoder_groups', default=32, type=int)


parser.add_argument('--decoder_upsample',default=1, type=int)



parser.add_argument('--grouped_conv',default=True, type=bool)
parser.add_argument('--root_grouping', default=True, type=bool)

parser.add_argument('--frames_path', default='images', type=str)
parser.add_argument('--dataset',default="DHF1K", type=str)

parser.add_argument('--alternate',default=1, type=int)

parser.add_argument('--split',default='no_split', type=str)


# added

parser.add_argument('--use_skip', default=1, type=int)

parser.add_argument('--neck_name', default='neck', type=str)

parser.add_argument('--videos_root_path', default='neck2', type=str)

parser.add_argument('--videos_frames_root_path', default='', type=str)


parser.add_argument('--len_snippet',default=64, type=int)


parser.add_argument('--fixation_data_path', default='', type=str)

parser.add_argument('--gt_sal_maps_path', default='', type=str)

parser.add_argument('--fold_lists_path', default='', type=str)

parser.add_argument('--model_save_root_path',default="/home/girmaji08/EEAA/SaliencyModel/saved_models/", type=str)


parser.add_argument('--load_weight',default="None", type=str)


#use_channel_shuffle
parser.add_argument('--use_channel_shuffle',default=True, type=bool)


#compute_metrics
parser.add_argument('--compute_metrics',default=True, type=bool)

#checkpoint path
parser.add_argument('--checkpoint_path_1',default='', type=str)
parser.add_argument('--checkpoint_path_2',default='', type=str)


#save_inferences
parser.add_argument('--save_inferences',default=False,type=bool)

#save_path
parser.add_argument('--save_path',default=None, type=str)


#metrics_save_path
parser.add_argument('--metrics_save_path',default=None, type=str)


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

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model_name = f"ViNet_E_{args.dataset}_{args.split}"
model_vinet_a = ViNet_A(args)
model_vinet_s = VideoSaliencyModel()


if args.dataset == "DHF1K":
	test_vinet_s_dataset = vinet_s_dhf1k(args.clip_size, mode="test", alternate=args.alternate, frames_path=args.frames_path,args = args)
	test_vinet_a_dataset = vinet_a_dhf1k(args,mode='test')

elif args.dataset == "Hollywood2":
	test_vinet_s_dataset = vinet_s_holly(args.clip_size, mode="test", frames_path=args.frames_path,args = args)
	test_vinet_a_dataset = vinet_a_holly(args,mode='test')

elif args.dataset == "UCF":
	test_vinet_s_dataset = vinet_s_ucf(args.clip_size, mode="test", frames_path=args.frames_path,args = args)
	test_vinet_a_dataset = vinet_a_ucf(args,mode='test')

elif args.dataset == "DIEM":
	test_vinet_s_dataset = vinet_s_av(args.clip_size, mode="test", dataset_name=args.dataset, split=args.split ,args = args)
	test_vinet_a_dataset = vinet_a_av(args,dataset='DIEM', mode="test")

elif args.dataset == "Coutrot_db1":
	test_vinet_s_dataset = vinet_s_av(args.clip_size, mode="test", dataset_name=args.dataset, split=args.split ,args = args)

	test_vinet_a_dataset = vinet_a_av(args,dataset='Coutrot_db1', mode="test")

elif args.dataset == "Coutrot_db2":
	test_vinet_s_dataset = vinet_s_av(args.clip_size, mode="test", dataset_name=args.dataset, split=args.split ,args = args)
	test_vinet_a_dataset = vinet_a_av(args,dataset='Coutrot_db2', mode="test")

elif args.dataset == "ETMD_av":
	test_vinet_s_dataset = vinet_s_av(args.clip_size, mode="test", dataset_name=args.dataset, split=args.split ,args = args)
	test_vinet_a_dataset = vinet_a_av(args,dataset='ETMD_av', mode="test")

elif args.dataset == 'mvva':
	test_vinet_s_dataset = vinet_s_mvva(args.clip_size, mode="test", dataset_name=args.dataset, split=args.split ,args = args)
 
	test_vinet_a_dataset = vinet_a_mvva(args,mode = "test")   



print("Loading the dataset...")

test_vinet_s_loader = torch.utils.data.DataLoader(test_vinet_s_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)

test_vinet_a_loader = torch.utils.data.DataLoader(test_vinet_a_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)


if args.checkpoint_path_1 != '':
	print("Loading pretrained vinet_a model weights if any: ",args.checkpoint_path_1)
	model_vinet_a.load_state_dict(torch.load(args.checkpoint_path_1))
	
if args.checkpoint_path_2 != '':
	print("Loading pretrained ViNet (36MB) (vinet_s) weights if any: ",args.checkpoint_path_2)
	model_vinet_s.load_state_dict(torch.load(args.checkpoint_path_2))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
	print("Let's use", torch.cuda.device_count(), "GPUs!")
	model_vinet_a = nn.DataParallel(model_vinet_a)
	model_vinet_s = nn.DataParallel(model_vinet_s)
model_vinet_a.to(device)
model_vinet_s.to(device)

print(device)


def test(model_vinet_a, model_vinet_s, loader_1, loader_2, device, args):
	model_vinet_a.eval()
	model_vinet_s.eval()


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

		img_clips_vinet_a, img_clips_vinet_s = sample[0][0], sample[1][0]
		gt_sal_vinet_a, gt_sal_vinet_s = sample[0][1], sample[1][1]
		binary_img_vinet_a, binary_img_vinet_s = sample[0][2], sample[1][2]
		video_name_vinet_a, video_name_vinet_s = sample[0][3][0], sample[1][3][0]
		mid_frame_vinet_a, end_frame_vinet_s = sample[0][4], sample[1][4]

		assert binary_img_vinet_a.shape == binary_img_vinet_s.shape
		assert video_name_vinet_a == video_name_vinet_s
		assert mid_frame_vinet_a == end_frame_vinet_s

		img_clips_vinet_a = img_clips_vinet_a.to(device)
		img_clips_vinet_s = img_clips_vinet_s.to(device)
		img_clips_vinet_a = img_clips_vinet_a.permute((0,2,1,3,4))
		img_clips_vinet_s = img_clips_vinet_s.permute((0,2,1,3,4))

		# doing inference
		
		pred_sal_vinet_a = model_vinet_a(img_clips_vinet_a)
		pred_sal_vinet_s = model_vinet_s(img_clips_vinet_s)


		# averaging the saliency maps
		pred_sal_vinet_s = F.interpolate(pred_sal_vinet_s.unsqueeze(0),size=(256,456),mode = 'bicubic',align_corners = False).squeeze(0)
		stacked_sal_maps = torch.stack((pred_sal_vinet_a, pred_sal_vinet_s), dim=0)
		pred_sal = torch.mean(stacked_sal_maps, dim=0)

		# post processing of output
		gt_sal = gt_sal_vinet_a.squeeze(0).numpy()

		pred_sal = pred_sal.cpu().squeeze(0).numpy()
		pred_sal = cv2.resize(pred_sal, (gt_sal.shape[1], gt_sal.shape[0]))
		pred_sal = blur(pred_sal).unsqueeze(0).cuda()


		
		if args.save_inferences and args.save_path is not None:
			os.makedirs(join(args.save_path, 'inferences',args.dataset,'ViNet_E',video_name_vinet_a),exist_ok=True)
			img_save(pred_sal, join(args.save_path,'inferences',args.dataset,'ViNet_E', video_name_vinet_a, 'img_%05d.png'%(end_frame_vinet_s+1)), normalize=True)
			
			
		gt_sal = torch.FloatTensor(gt_sal).unsqueeze(0).cuda()

		assert pred_sal.size() == gt_sal.size()

		# metrics computation

		if args.compute_metrics:
			cc_loss = cc(pred_sal, gt_sal)

		
			kldiv_loss = kldiv(pred_sal, gt_sal)

			sim_loss = similarity(pred_sal, gt_sal)

			aucj_loss = auc_judd(pred_sal.cpu(), binary_img_vinet_a.cpu())

			if isinstance(aucj_loss, float):
				aucj_loss = torch.FloatTensor([aucj_loss])


			nss_loss = torch.FloatTensor([0.0]).cuda()
			for i in range(len(binary_img_vinet_a)):

				nss_loss += nss(pred_sal[i,:,:].unsqueeze(0).detach().to(device), binary_img_vinet_a[i].unsqueeze(0).to(device))
					
			nss_loss = nss_loss/len(binary_img_vinet_a)


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

			video_kldiv_loss[video_name_vinet_a] += kldiv_loss.item()
			video_cc_loss[video_name_vinet_a] += cc_loss.item()
			video_sim_loss[video_name_vinet_a] += sim_loss.item()
			video_nss_loss[video_name_vinet_a] += nss_loss.item()
			video_aucj_loss[video_name_vinet_a] += aucj_loss.item()
			video_num_frames[video_name_vinet_a] += 1



	print("Total number of NaNs: ", nan_counts)
	return video_kldiv_loss, video_cc_loss, video_sim_loss, video_nss_loss, video_aucj_loss, video_num_frames




with torch.no_grad():
	video_kldiv_loss, video_cc_loss, video_sim_loss, video_nss_loss, video_aucj_loss, video_num_frames = test(model_vinet_a, model_vinet_s, test_vinet_a_loader, test_vinet_s_loader, device, args)




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
	
	