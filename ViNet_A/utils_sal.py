import torch
import torch.nn as nn
from loss import *
import cv2
from torchvision import transforms, utils
from PIL import Image
import random
from os.path import join
import scipy.io as sio

def check_gpu_memory():
    return torch.cuda.memory_allocated(), torch.cuda.memory_cached()

def garbage_collect_if_needed(threshold_percent=60):
    allocated, cached = check_gpu_memory()
    utilization_percent = (allocated / cached) * 100
    if utilization_percent > threshold_percent:
        gc.collect()
        torch.cuda.empty_cache()

def get_loss(pred_map, gt, args):
    loss = torch.FloatTensor([0.0]).cuda()
    if args.kldiv:
        loss += args.kldiv_coeff * kldiv(pred_map, gt)
    if args.cc:
        loss += args.cc_coeff * cc(pred_map, gt)
    if args.l1:
        loss += args.l1_coeff * criterion(pred_map, gt)
    if args.sim:
        loss += args.sim_coeff * similarity(pred_map, gt)
    if args.nss:
        loss += args.nss_coeff * nss(pred_map, gt)
    return loss

def get_frame_indices(mid_frame,len_snippet,num_frames):
     
    frame_indices = np.arange(mid_frame-(len_snippet // 2),mid_frame + (len_snippet // 2))

    temp = []
    for i in frame_indices:
        if i < 0:
            temp.append(0)
        else:
            temp.append(i)
    frame_indices = temp
    temp = []
    for i in frame_indices:
        if i >= num_frames:
            temp.append(num_frames-1)
        else:
            temp.append(i)

    frame_indices = temp

    return frame_indices


def loss_func(pred_map, gt, args):
    loss = torch.FloatTensor([0.0]).cuda()
    criterion = nn.L1Loss()
    assert pred_map.size() == gt.size()

    if len(pred_map.size()) == 4:
        ''' Clips: BxClXHxW '''
        assert pred_map.size(0)==args.batch_size
        pred_map = pred_map.permute((1,0,2,3))
        gt = gt.permute((1,0,2,3))

        for i in range(pred_map.size(0)):
            loss += get_loss(pred_map[i], gt[i], args)

        loss /= pred_map.size(0)
        return loss
    
    return get_loss(pred_map, gt, args)

class AverageMeter(object):

    '''Computers and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count

def blur(img):
    k_size = 11
    bl = cv2.GaussianBlur(img,(k_size,k_size),0)
    return torch.FloatTensor(bl)

def img_save(tensor, fp, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0, format=None):
    grid = utils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, value_range=range, scale_each=scale_each)

    ndarr = torch.round(grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)).to('cpu', torch.uint8).numpy()
    ndarr = ndarr[:,:,0]
    im = Image.fromarray(ndarr)
    exten = fp.split('.')[-1]
    if exten=="png":
        im.save(fp, format=format)
    else:
        im.save(fp, format=format, quality=100) #for jpg


def num_params(model):
    return sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())

def get_fixation(path_indata, dname, _id,dataset_name):
	info = sio.loadmat(join(path_indata,dname, 'fixMap_{}.mat'.format(_id)))
	return info['eyeMap']


  
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
    # else:

    #     video_name1,video_name2 = video_name[0],video_name[1]
    #     mid_frame1,mid_frame2 = mid_frame[0],mid_frame[1]
    #     num_rois1,num_rois2 = num_rois[0],num_rois[1]

    #     mask = np.ones((16,29),dtype=np.uint8)

    #     h,w = 16,29

    #     for d in annotation_data:
    #         if d['video'] == video_name1 and d['mid_frame'] == mid_frame1:
    #             target = d['labels']


    #     # print("Masking the input")
    #     use_orig = random.sample([True,False],1)
    #     if use_orig:
    #         if num_rois1 >= 2:
    #             if len(target) >= num_rois1:
    #                 if num_rois not in [2,3]:
    #                     proposals_dropped = random.sample(target,int(round(max(len(target),num_rois1)/3)))#random.sample(target,2)##int(np.floor(len(target)/3))#random.sample(target,2)#int(np.floor(len(target)/3)))#max(len(target),num_rois) - 3)
    #                 else:
    #                     proposals_dropped = random.sample(target,1)
    #                 for proposal in proposals_dropped:
    #                     target.remove(proposal)
    #                     bboxes = proposal['bounding_box']
    #                     # xmin,ymin,xmax,ymax = int(bboxes[0]*w),int(bboxes[1]*h),int(bboxes[2]*w),int(bboxes[3]*h)
    #                     xmin,ymin,xmax,ymax = int(max(bboxes[0]-114/456,0)*w),int(bboxes[1]*h),int(min(bboxes[2] - 114/456,1)*w),int(bboxes[3]*h)
    #                     mask[ymin:ymax,xmin:xmax] = 0

    #     for d in annotation_data:
    #         if d['video'] == video_name2 and d['mid_frame'] == mid_frame2:
    #             target = d['labels']


    #     # print("Masking the input")
    #     use_orig = random.sample([True,False],1)
    #     if use_orig:
    #         if num_rois2 >= 2:
    #             if len(target) >= num_rois2:
    #                 if num_rois not in [2,3]:
    #                     proposals_dropped = random.sample(target,int(round(max(len(target),num_rois2)/3)))#random.sample(target,2)##int(np.floor(len(target)/3))#random.sample(target,2)#int(np.floor(len(target)/3)))#max(len(target),num_rois) - 3)
    #                 else:
    #                     proposals_dropped = random.sample(target,1)
    #                 for proposal in proposals_dropped:
    #                     target.remove(proposal)
    #                     bboxes = proposal['bounding_box']
    #                     # xmin,ymin,xmax,ymax = int(bboxes[0]*w),int(bboxes[1]*h),int(bboxes[2]*w),int(bboxes[3]*h)
    #                     xmin,ymin,xmax,ymax = int(max(bboxes[0]-114/456,0)*w),int(bboxes[1]*h),int(min(bboxes[2] - 114/456,1)*w),int(bboxes[3]*h)
    #                     mask[ymin:ymax,xmin:xmax] = 0
    #                 # np.save('mask.npy',mask)        
    

				# for d in self.annotation_data:
				# 		if d['video'] == video_name and d['mid_frame'] == mid_frame:
				# 			target = d['labels']

				# mask = np.ones(clip[0].size(),dtype=np.uint8)

				# h,w = clip[0].shape[-2:]
				# # print("Masking the input")
				# use_orig = random.sample([True,False],1)
				# if use_orig:
				# 	if num_rois >= 3:
				# 		if len(target) >= num_rois:
				# 			if num_rois not in [3]:
				# 				proposals_dropped = random.sample(target,2)#random.sample(target,int(round(max(len(target),num_rois)/3)))#int(np.floor(len(target)/3))#random.sample(target,2)#int(np.floor(len(target)/3)))#max(len(target),num_rois) - 3)
				# 			else:
				# 				proposals_dropped = random.sample(target,1)
				# 			for proposal in proposals_dropped:
				# 				target.remove(proposal)
				# 				bboxes = proposal['bounding_box']
				# 				xmin,ymin,xmax,ymax = int(bboxes[0]*w),int(bboxes[1]*h),int(bboxes[2]*w),int(bboxes[3]*h)
				# 				mask[:,:,ymin:ymax,xmin:xmax] = 0
				# 			np.save('mask.npy',mask)
				# clip = [i*mask for i in clip]


                                    # if ymax-ymin < 3 or xmax-xmin < 3:
                    #     xmin = max(0,xmin-2)
                    #     ymin = max(0,ymin-2)
                    #     xmax = min(w,xmax+2)
                    #     ymax = min(h,ymax+2) 