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

from ViNet_A_model import *
from utils_sal import *

import pdb
from collections import OrderedDict
import wandb


from ViNet_A_visual_dataloader import *
from ViNet_A_audio_visual_dataloader import *
# from EEAA_audio_visual_dataloader import *

from tqdm import tqdm

from os.path import join
import random


parser = argparse.ArgumentParser()

parser.add_argument('--no_epochs',default=120, type=int)
parser.add_argument('--lr',default=1e-4, type=float)

parser.add_argument('--kldiv',default=1, type=int)
parser.add_argument('--cc',default=1, type=int)
parser.add_argument('--nss',default=0, type=int)
parser.add_argument('--sim',default=0, type=int)

parser.add_argument('--optim',default="Adam", type=str)
parser.add_argument('--lr_sched',default=0, type=int)


parser.add_argument('--step_size',default=10, type=int)

parser.add_argument('--kldiv_coeff',default=1.0, type=float)
parser.add_argument('--cc_coeff',default=-0.5, type=float)
parser.add_argument('--sim_coeff',default=0.0, type=float)
parser.add_argument('--nss_coeff',default=-0.1, type=float)


parser.add_argument('--batch_size',default=8, type=int)
parser.add_argument('--log_interval',default=5, type=int)
parser.add_argument('--no_workers',default=4, type=int)



parser.add_argument('--decoder_groups', default=32, type=int)


parser.add_argument('--decoder_upsample',default=1, type=int)



parser.add_argument('--dataset',default="mvva", type=str)

parser.add_argument('--split',default='no_split', type=str)

# added

parser.add_argument('--use_skip', default=1, type=int)

parser.add_argument('--neck_name', default='neck2', type=str)

parser.add_argument('--videos_root_path', default='', type=str)

parser.add_argument('--videos_frames_root_path', default='', type=str)



parser.add_argument('--len_snippet',default=32, type=int)


parser.add_argument('--fixation_data_path', default='', type=str)

parser.add_argument('--gt_sal_maps_path', default='', type=str)

parser.add_argument('--fold_lists_path', default='', type=str)

parser.add_argument('--model_save_path',default="/home/sid/fantastic_gains/saved_models/", type=str)


#checkpoint_path
parser.add_argument('--checkpoint_path',default="None", type=str)



#use_channel_shuffle
parser.add_argument('--use_channel_shuffle',default=True, type=bool)


#model_tag
parser.add_argument('--model_tag',default='', type=str)


#use_action_classification
parser.add_argument('--use_action_classification',default=0, type=int)

#seed
parser.add_argument('--seed',default=0, type=int)

#scheduler
parser.add_argument('--scheduler',default=0, type=int)



args = parser.parse_args()


if args.dataset == 'mvva':
    seed = 1100

if args.dataset == 'Hollywood2':
    seed = 867
    print("Seed used is : ",seed)

if args.dataset == 'DHF1K':
    seed = 867
    print("Seed used is : ",seed)

if args.dataset == 'Coutrot_db2':
    seed = 867#np.random.randint(0,10000)#867
    print("Seed used is : ",seed)

if args.dataset == 'Coutrot_db1':
    seed = 867#np.random.randint(0,10000)#867
    print("Seed used is : ",seed)

if args.dataset == 'DIEM':
    seed = 6161#np.random.randint(0,10000)#867
    print("Seed used is : ",seed)

if args.dataset == 'SumMe':
    seed = 867#np.random.randint(0,10000)#867
    print("Seed used is : ",seed)

if args.dataset == 'ETMD_av':
    seed = 867#np.random.randint(0,10000)#867
    print("Seed used is : ",seed)


if args.dataset == 'UCF':
    seed = 867#np.random.randint(0,10000)#867
    print("Seed used is : ",seed)


np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



model_tag = f"ViNet_A_{args.dataset}_{args.split}"

model_save_path = join(args.model_save_path,model_tag)

print(args.model_save_path)
print("model save path is : ", model_save_path)
 
# create the directory to save the models
os.makedirs(args.model_save_path,exist_ok=True)


wandb.login()

# wandb.init()
wandb.init(
    project='fantastic_gains',
    name= '%s' % (model_tag),
    config={
        'batch_size': args.batch_size,
        'dataset': args.dataset,
        'seed': seed,
        'decoder_groups': args.decoder_groups,
        'no_epochs': args.no_epochs,
        'lr': args.lr,
        'lr_sched': args.lr_sched,
        'scheduler': args.scheduler,
        'num_GPUs': torch.cuda.device_count(),
        'optim': args.optim,
        'neck_name': args.neck_name,
        'split': args.split,
        'nss_coeff': args.nss_coeff,
        'cc_coeff': args.cc_coeff,
        'sim_coeff': args.sim_coeff,
        'kldiv_coeff': args.kldiv_coeff,
        'kldiv': args.kldiv,
        'nss': args.nss,
        'cc': args.cc,
        'sim': args.sim
    }
)



model = ViNet_A(args)





if args.dataset == "DHF1K":
    train_dataset = DHF1K_Dataset(args,mode='train')
    val_dataset = DHF1K_Dataset(args,mode='val')

elif args.dataset == "UCF":
    print("Using UCF dataset")
    train_dataset = UCF_Dataset(args, mode="train")
    val_dataset = UCF_Dataset(args, mode="val")

elif args.dataset == "Hollywood2":
    print("Using Hollywood2 dataset")
    train_dataset = Hollywood2_Dataset(args, mode="train")
    val_dataset = Hollywood2_Dataset(args, mode="val")
   

elif args.dataset == "mvva":
    print("Using MVVA dataset")
    train_dataset = MVVA_Dataset(args, mode="train")
    val_dataset = MVVA_Dataset(args, mode="val")
  
elif args.dataset == "SumMe":
    print("Using SumMe dataset")
    train_dataset = Other_AudioVisual_Dataset(args,dataset='SumMe', mode="train")
    val_dataset = Other_AudioVisual_Dataset(args,dataset='SumMe', mode="val")

elif args.dataset == "Coutrot_db2":
    print("Using Coutrot_db2 dataset")
    train_dataset = Other_AudioVisual_Dataset(args,dataset='Coutrot_db2', mode="train")
    val_dataset = Other_AudioVisual_Dataset(args,dataset='Coutrot_db2', mode="val")

elif args.dataset == "Coutrot_db1":
    print("Using Coutrot_db1 dataset")
    train_dataset = Other_AudioVisual_Dataset(args,dataset='Coutrot_db1', mode="train")
    val_dataset = Other_AudioVisual_Dataset(args,dataset='Coutrot_db1', mode="val")

elif args.dataset == "ETMD_av":
    print("Using ETMD dataset")
    train_dataset = Other_AudioVisual_Dataset(args,dataset='ETMD_av', mode="train")
    val_dataset = Other_AudioVisual_Dataset(args,dataset='ETMD_av', mode="val")

elif args.dataset == "DIEM":
    print("Using DIEM dataset")
    train_dataset = Other_AudioVisual_Dataset(args,dataset='DIEM', mode="train")
    val_dataset = Other_AudioVisual_Dataset(args,dataset='DIEM', mode="val")




print("Loading the dataset...")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.no_workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)


if  args.checkpoint_path == "None":
    if args.use_action_classification:

        weight_dict = OrderedDict(torch.load(os.path.expanduser('~/ICASSP_Saliency/pretrained_models/SLOWFAST_R50_K400.pth.tar')))
        print("---------------------Loading PreTrained Kinetics400 Action Classification Weights...........")
    else:
        weight_dict = OrderedDict(torch.load(os.path.expanduser('~/ICASSP_Saliency/pretrained_models/AVA_SLOWFAST_R50_ACAR_HR2O.pth.tar')))
        print("---------------------Loading PreTrained AVA Action Detection(STAL) Weights...........")

    model_dict = model.backbone.state_dict()
    # pdb.set_trace()
    if not args.use_action_classification:
        print("USING STAL BACKBONE WEIGHTS")
        new_state_dict = {}
        for name, param in weight_dict.items():
            name = '.'.join(name.split('.')[3:])
            new_state_dict[name] = param
        weight_dict = new_state_dict

    print ('loaded SlowFast weight file')
    model.backbone.load_state_dict(weight_dict, strict=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.checkpoint_path!="None":
    print("Loading pretrained {} model weights if any: {}".format(args.model_tag,args.checkpoint_path))
    model.load_state_dict(torch.load(os.path.expanduser(args.checkpoint_path), map_location=device),strict=True)

# print(args)
print(model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)

params = list(filter(lambda p: p.requires_grad, model.parameters())) 
if args.optim == 'Adam':
    optimizer = torch.optim.Adam(params, lr=args.lr)
else:
    optimizer = torch.optim.SGD(params, lr=args.lr)


print("Using the device: ",device)

def train(model, optimizer, loader, epoch, device, args):
    model.train()
    tic = time.time()
    
    total_loss = AverageMeter()
    total_cc_loss = AverageMeter()
    total_kldiv_loss = AverageMeter()
    total_sim_loss = AverageMeter()
    total_nss_loss = AverageMeter()
    cur_loss = AverageMeter()

    count = 0

    for idx, sample in tqdm(enumerate(loader)):
        img_clips = sample[0]
        gt_sal = sample[1]
        binary_img = sample[2]

        img_clips = img_clips.to(device)
        img_clips = img_clips.permute((0,2,1,3,4))
        gt_sal = gt_sal.to(device)
        
        optimizer.zero_grad()

        pred_sal = model(img_clips)

        # print(pred_sal.size(),gt_sal.size())

        assert pred_sal.size() == gt_sal.size()

        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f'NaN or Inf detected in training gradients of {name} at epoch {epoch}')
                    continue  # Skip the rest of the loop to avoid errors

        if args.cc:
            cc_loss = cc(pred_sal, gt_sal)
            if torch.isnan(cc_loss).any() or torch.isinf(cc_loss).any():
                print("cc_loss is nan or inf")
                cc_loss = torch.FloatTensor([0.0]).to(device)
                count += 1
                optimizer.zero_grad()
                del img_clips, gt_sal, pred_sal, cc_loss, kldiv_loss, #nss_loss
                gc.collect()
                torch.cuda.empty_cache()
                continue
        if args.kldiv:
            kldiv_loss = kldiv(pred_sal, gt_sal)
            if torch.isnan(kldiv_loss).any() or torch.isinf(kldiv_loss).any():
                print("kldiv_loss is nan or inf")
                kldiv_loss = torch.FloatTensor([0.0]).to(device)
                count += 1
                optimizer.zero_grad()
                del img_clips, gt_sal, pred_sal, cc_loss, kldiv_loss, #nss_loss
                gc.collect()
                torch.cuda.empty_cache()
                continue
        if args.sim:
            sim_loss = similarity(pred_sal, gt_sal)

        if args.nss:
            if idx == 0:
                print("loss function includes NSS")
                print("length of binary_img", len(binary_img))
            nss_loss = torch.FloatTensor([0.0]).cuda()
            for i in range(len(binary_img)):
                nss_loss += nss(pred_sal[i,:,:].unsqueeze(0).detach().to(device), binary_img[i].unsqueeze(0).to(device))

            nss_loss = nss_loss/len(binary_img)

        loss_list = []
        for n,(i,j) in enumerate(zip([args.cc, args.kldiv, args.sim, args.nss],[args.cc_coeff, args.kldiv_coeff, args.sim_coeff, args.nss_coeff])):
            
            if i and n==0: loss_list.append(j*cc_loss)
            if i and n==1: loss_list.append(j*kldiv_loss)
            if i and n==2: loss_list.append(j*sim_loss)
            if i and n==3: loss_list.append(j*nss_loss)

        loss = sum(loss_list)

        loss.backward()

        optimizer.step()

        total_loss.update(loss.item())
        if args.cc:
            total_cc_loss.update(cc_loss.item())
        if args.kldiv:
            total_kldiv_loss.update(kldiv_loss.item())
        if args.sim:
            total_sim_loss.update(sim_loss.item())
        if args.nss:
            total_nss_loss.update(nss_loss.item())

        cur_loss.update(loss.item())

        if idx%args.log_interval==(args.log_interval-1):
            print('[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes'.format(epoch, idx, cur_loss.avg, (time.time()-tic)/60))
            cur_loss.reset()
            sys.stdout.flush()


    losses_dict = {}

    losses_dict['train_avg_loss'] = total_loss.avg
    if args.cc:
        losses_dict['train_cc_loss'] = total_cc_loss.avg
    if args.kldiv:
        losses_dict['train_kldiv_loss'] = total_kldiv_loss.avg
    if args.sim:
        losses_dict['train_sim_loss'] = total_sim_loss.avg
    if args.nss:
        losses_dict['train_nss_loss'] = total_nss_loss.avg


    return total_loss.avg,losses_dict

def validate(model, loader, epoch, device, args):
    model.eval()
    tic = time.time()
    total_loss = AverageMeter()
    total_cc_loss = AverageMeter()
    total_kldiv_loss = AverageMeter()
    total_sim_loss = AverageMeter()
    total_nss_loss = AverageMeter()
    tic = time.time()
    count = 0
    # with wandb.init(config=sweep_configuration):
    for idx, sample in tqdm(enumerate(loader)):

        img_clips = sample[0]
        gt_sal = sample[1]
        binary_img = sample[2]

        img_clips = img_clips.to(device)
        img_clips = img_clips.permute((0,2,1,3,4))

        pred_sal = model(img_clips)
        
        gt_sal = gt_sal.squeeze(0).numpy()

        pred_sal = pred_sal.cpu().squeeze(0).numpy()


        pred_sal = cv2.resize(pred_sal, (gt_sal.shape[1], gt_sal.shape[0]))
        pred_sal = blur(pred_sal).unsqueeze(0).cuda()


        gt_sal = torch.FloatTensor(gt_sal).unsqueeze(0).cuda()


        assert pred_sal.size() == gt_sal.size()


        if args.cc:
            cc_loss = cc(pred_sal, gt_sal)

            if torch.isnan(cc_loss).any() or torch.isinf(cc_loss).any():
                cc_loss = torch.FloatTensor([0.0]).cuda()
                count += 1


        if args.kldiv:
            kldiv_loss = kldiv(pred_sal, gt_sal)

            if torch.isnan(kldiv_loss).any() or torch.isinf(kldiv_loss).any():
                kldiv_loss = torch.FloatTensor([0.0]).cuda()
                count += 1


        if args.sim:
            sim_loss = similarity(pred_sal, gt_sal)

        if args.nss:
            if idx == 0:
                print("loss function includes NSS")
                print("length of binary_img", len(binary_img))
            nss_loss = torch.FloatTensor([0.0]).cuda()
            for i in range(len(binary_img)):
                nss_loss += nss(pred_sal[i,:,:].unsqueeze(0).detach().to(device), binary_img[i].unsqueeze(0).to(device))
                
            nss_loss = nss_loss/len(binary_img)
        
        loss_list = []
        for n,(i,j) in enumerate(zip([args.cc, args.kldiv, args.sim, args.nss],[args.cc_coeff, args.kldiv_coeff, args.sim_coeff, args.nss_coeff])):
            
            if i and n==0: loss_list.append(j*cc_loss)
            if i and n==1: loss_list.append(j*kldiv_loss)
            if i and n==2: loss_list.append(j*sim_loss)
            if i and n==3: loss_list.append(j*nss_loss)

        loss = sum(loss_list)


        total_loss.update(loss.item())
        if args.cc:
            total_cc_loss.update(cc_loss.item())
        if args.kldiv:
            total_kldiv_loss.update(kldiv_loss.item())
        if args.sim:
            total_sim_loss.update(sim_loss.item())
        if args.nss:
            total_nss_loss.update(nss_loss.item())



    losses_dict = {}

    losses_dict['val_avg_loss'] = total_loss.avg
    if args.cc:
        losses_dict['val_cc_loss'] = total_cc_loss.avg
    if args.kldiv:
        losses_dict['val_kldiv_loss'] = total_kldiv_loss.avg
    if args.sim:
        losses_dict['val_sim_loss'] = total_sim_loss.avg
    if args.nss:
        losses_dict['val_nss_loss'] = total_nss_loss.avg

    print('[{:2d}, val] avg_loss : {:.5f} cc_loss : {:.5f}  time : {:3f}'.format(epoch, total_loss.avg, total_cc_loss.avg, (time.time()-tic)/60))
    sys.stdout.flush()

    print("Nans in validation losses count is ",count)

    return total_loss.avg,losses_dict



best_model = None



for epoch in range(0, args.no_epochs):



    loss,train_losses_dict = train(model, optimizer, train_loader, epoch, device, args)


    with torch.no_grad():
       
        val_loss,val_losses_dict = validate(model, val_loader, epoch, device, args)
        
        losses_dict = {**train_losses_dict,**val_losses_dict}

        wandb.log(losses_dict,step=epoch)

        if epoch == 0 :
            val_loss = np.inf
            best_loss = val_loss
        if val_loss <= best_loss:
            best_loss = val_loss
            best_model = model
            print('[{:2d},  save, {}]'.format(epoch, model_save_path))
            if torch.cuda.device_count() > 1:    
                torch.save(model.module.state_dict(), model_save_path)
            else:
                torch.save(model.state_dict(), model_save_path)


