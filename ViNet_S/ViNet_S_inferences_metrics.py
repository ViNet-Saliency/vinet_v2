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
from collections import OrderedDict
import wandb

from ViNet_S_model import *
from utils import *
from ViNet_S_dataloader import *

from tqdm import tqdm

from os.path import join
import random
from collections import defaultdict 
import copy

def list_of_strings(string):
	return string.split(',')

parser = argparse.ArgumentParser()


parser.add_argument('--no_workers',default=4, type=int)


parser.add_argument('--frames_path', default='images', type=str)
parser.add_argument('--decoder_groups', default=32, type=int)
parser.add_argument('--decoder_upsample',default=1, type=int)
parser.add_argument('--num_hier',default=3, type=int)
parser.add_argument('--clip_size',default=32, type=int)
parser.add_argument('--batch_size',default=1, type=int)
parser.add_argument('--pin_memory',default=False, type=bool)


parser.add_argument('--grouped_conv',default=True, type=bool)
parser.add_argument('--root_grouping', default=True, type=bool)
parser.add_argument('--depth_grouping', default=False, type=bool)
parser.add_argument('--efficientnet', default=False, type=bool)
parser.add_argument('--use_trilinear_upsampling', default=False, type=bool)
parser.add_argument('--alternate',default=1, type=int)

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

# added
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



model_name = args.checkpoint_path.split('/')[-1].split('.')[0]

model = VideoSaliencyModel(
	use_upsample=bool(args.decoder_upsample),
    num_hier=args.num_hier,
    num_clips=args.clip_size,
    grouped_conv=args.grouped_conv,
    root_grouping=args.root_grouping,
    depth=args.depth_grouping,
    efficientnet=args.efficientnet,
    BiCubic = False,#not args.use_trilinear_upsampling,
)



if args.dataset == "DHF1K":
    test_dataset = DHF1KDataset(args.clip_size, mode="test", alternate=args.alternate, frames_path=args.frames_path,args = args)
   

elif args.dataset == "Hollywood2":
    test_dataset = HollywoodDataset(args.clip_size, mode="test", frames_path=args.frames_path,args = args)
    
elif args.dataset == "UCF":
    test_dataset = UCFDataset(args.clip_size, mode="test", frames_path=args.frames_path,args = args)
    

elif args.dataset == "DIEM":
    test_dataset = OtherAudioVisualDataset(args.clip_size, mode="test", dataset_name=args.dataset, split=args.split ,args = args)
    

elif args.dataset == "AVAD":
    test_dataset = OtherAudioVisualDataset(args.clip_size, mode="test", dataset_name=args.dataset, split=args.split ,args = args)
    
elif args.dataset == "ETMD_av":
    test_dataset = OtherAudioVisualDataset(args.clip_size, mode="test", dataset_name=args.dataset, split=args.split ,args = args)
    
elif args.dataset == "Coutrot_db1":
    test_dataset = OtherAudioVisualDataset(args.clip_size, mode="test", dataset_name=args.dataset, split=args.split ,args = args)
    
elif args.dataset == "Coutrot_db2":
    test_dataset = OtherAudioVisualDataset(args.clip_size, mode="test", dataset_name=args.dataset, split=args.split ,args = args)
    
elif args.dataset == "mvva":
    test_dataset = MVVADataset(args.clip_size, mode="test", dataset_name=args.dataset, split=args.split ,args = args)
    


print("Loading the dataset...")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.no_workers, pin_memory=args.pin_memory)


if args.checkpoint_path!='':
	print("Loading pretrained ViNet (36MB) weights if any: ",args.checkpoint_path)
	model.load_state_dict(torch.load(args.checkpoint_path))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
	print("Let's use", torch.cuda.device_count(), "GPUs!")
	model = nn.DataParallel(model)
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
	kldiv_nan_counts,cc_nan_counts,sim_nan_counts,nss_nan_counts,aucj_nan_counts = 0,0,0,0,0

	num_frames = 0

	for idx, sample in tqdm(enumerate(loader)):

		# getting the input
		num_frames += 1

		img_clips = sample[0]
		gt_sal = sample[1]
		binary_img = sample[2]
		video_name = sample[3][0]
		end_idx = sample[4]

		img_clips = img_clips.to(device)
		img_clips = img_clips.permute((0,2,1,3,4))


		pred_sal = model(img_clips)

		# post processing of output
		
		gt_sal = gt_sal.squeeze(0).numpy()

		pred_sal = pred_sal.cpu().squeeze(0).numpy()
		pred_sal = cv2.resize(pred_sal, (gt_sal.shape[1], gt_sal.shape[0]))
		pred_sal = blur(pred_sal).unsqueeze(0).cuda()

		if args.save_inferences and args.save_path is not None:
			os.makedirs(join(args.save_path, 'inferences',args.dataset,'ViNet_S',video_name),exist_ok=True)
			img_save(pred_sal, join(args.save_path,'inferences',args.dataset,'ViNet_S', video_name, 'img_%05d.png'%(end_idx+1)), normalize=True)
			
			


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

			nan_counts = {"KLDiv": kldiv_nan_counts, "CC": cc_nan_counts, "SIM": sim_nan_counts, "NSS": nss_nan_counts, "AUCj": aucj_nan_counts}

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

