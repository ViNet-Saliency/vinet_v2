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

def get_input_mask(clip,annotation_data,mid_frame,video_name,num_rois):
    for d in annotation_data:
        if d['video'] == video_name and d['mid_frame'] == mid_frame:
            target = d['labels']

    mask = np.ones(clip[0].size(),dtype=np.uint8)

    h,w = clip[0].shape[-2:]
    # print("Masking the input")
    use_orig = random.sample([True,False],1)
    if use_orig:
        if num_rois >= 3:
            if len(target) >= num_rois:
                if num_rois not in [3]:
                    proposals_dropped = random.sample(target,2)#random.sample(target,int(round(max(len(target),num_rois)/3)))#int(np.floor(len(target)/3))#random.sample(target,2)#int(np.floor(len(target)/3)))#max(len(target),num_rois) - 3)
                else:
                    proposals_dropped = random.sample(target,1)
                for proposal in proposals_dropped:
                    target.remove(proposal)
                    bboxes = proposal['bounding_box']
                    xmin,ymin,xmax,ymax = int(bboxes[0]*w),int(bboxes[1]*h),int(bboxes[2]*w),int(bboxes[3]*h)
                    mask[:,:,ymin:ymax,xmin:xmax] = 0
                np.save('mask.npy',mask)
    clip = [i*mask for i in clip]

    return clip,mask


class DropBlock(torch.nn.Module):
    def __init__(self, block_size=3, keep_prob=0.9):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.gamma = None
        self.kernel_size = (block_size, block_size)
        self.stride = (1, 1)
        self.padding = (block_size//2, block_size//2)
    
    def calculate_gamma(self, x):
        return (1 - self.keep_prob) * x.shape[-1]**2/\
                (self.block_size**2 * (x.shape[-1] - self.block_size + 1)**2) 
    
    def forward(self, x):
        if self.gamma is None:
            self.gamma = self.calculate_gamma(x)
        p = torch.ones_like(x) * self.gamma
        mask = 1 - torch.nn.functional.max_pool2d(torch.bernoulli(p),
                                                  self.kernel_size,
                                                  self.stride,
                                                  self.padding)
        return mask * x * (mask.numel()/mask.sum())
    



def get_feature_mask(annotation_data,mid_frame,video_name,num_rois):

    # if not isinstance(video_name,tuple):
    for d in annotation_data:
        if d['video'] == video_name and d['mid_frame'] == mid_frame:
            target = d['labels']

    dp = DropBlock()

    mask0 = np.ones((1,16,29),dtype=np.uint8)

    h,w = 16,29

    # print("Masking the input")
    use_orig = np.random.choice([True,False],1,p=[0.4,0.6])[0]
    if use_orig:
        if num_rois >= 2:
            if len(target) >= num_rois:
                if num_rois not in [2,3]:
                    proposals_dropped = random.sample(target,int(round(max(len(target),num_rois)/3)))#random.sample(target,2)##int(np.floor(len(target)/3))#random.sample(target,2)#int(np.floor(len(target)/3)))#max(len(target),num_rois) - 3)
                else:
                    proposals_dropped = random.sample(target,1)
                for proposal in proposals_dropped:
                    target.remove(proposal)
                    bboxes = proposal['bounding_box']
                    xmin,ymin,xmax,ymax = int(bboxes[0]*w),int(bboxes[1]*h),int(bboxes[2]*w),int(bboxes[3]*h)
                    # xmin,ymin,xmax,ymax = int(max(bboxes[0]-114/456)*w),int(bboxes[1]*h),int(bboxes[2]*w),int(bboxes[3]*h)
                    
                    y = np.random.choice([0,1],size = (1,ymax-ymin,xmax-xmin),p=[0.6,0.4])
                    mask0[:,ymin:ymax,xmin:xmax] = y
                    # mask0[:,ymin:ymax,xmin:xmax] = dp(torch.Tensor(mask0[:,ymin:ymax,xmin:xmax]))

                np.save('mask.npy',mask0)

    mask0 = mask0[0]


    mask1 = np.ones((1,16,29),dtype=np.uint8)
    h,w = 16,29

    # print("Masking the input")
    use_orig = np.random.choice([True,False],1,p=[0.4,0.6])[0]
    if use_orig:
        if num_rois >= 2:
            if len(target) >= num_rois:
                if num_rois not in [2,3]:
                    proposals_dropped = random.sample(target,int(round(max(len(target),num_rois)/3)))#random.sample(target,2)##int(np.floor(len(target)/3))#random.sample(target,2)#int(np.floor(len(target)/3)))#max(len(target),num_rois) - 3)
                else:
                    proposals_dropped = random.sample(target,1)
                for proposal in proposals_dropped:
                    target.remove(proposal)
                    bboxes = proposal['bounding_box']
                    xmin,ymin,xmax,ymax = int(bboxes[0]*w),int(bboxes[1]*h),int(bboxes[2]*w),int(bboxes[3]*h)
  
                    y = np.random.choice([0,1],size = (1,ymax-ymin,xmax-xmin),p=[0.6,0.4])
                    mask1[:,ymin:ymax,xmin:xmax] = y
                    # mask1[:,ymin:ymax,xmin:xmax] = dp(torch.Tensor(mask1[:,ymin:ymax,xmin:xmax]))
                # np.save('mask.npy',mask)
    mask1 = mask1[0]

    mask2 = np.ones((1,32,57),dtype=np.uint8)

    h,w = 32,57

    # print("Masking the input")
    use_orig = np.random.choice([True,False],1,p=[0.4,0.6])[0]
    if use_orig:
        if num_rois >= 2:
            if len(target) >= num_rois:
                if num_rois not in [2,3]:
                    proposals_dropped = random.sample(target,int(round(max(len(target),num_rois)/3)))#random.sample(target,2)##int(np.floor(len(target)/3))#random.sample(target,2)#int(np.floor(len(target)/3)))#max(len(target),num_rois) - 3)
                else:
                    proposals_dropped = random.sample(target,1)
                for proposal in proposals_dropped:
                    target.remove(proposal)
                    bboxes = proposal['bounding_box']
                    xmin,ymin,xmax,ymax = int(bboxes[0]*w),int(bboxes[1]*h),int(bboxes[2]*w),int(bboxes[3]*h)
         
                    y = np.random.choice([0,1],size = (1,ymax-ymin,xmax-xmin),p=[0.6,0.4])
                    mask2[:,ymin:ymax,xmin:xmax] = y
                    # mask2[:,ymin:ymax,xmin:xmax] = dp(torch.Tensor(mask2[:,ymin:ymax,xmin:xmax]))

    mask2 = mask2[0]


    mask3 = np.ones((1,64,114),dtype=np.uint8)

    h,w = 64,114

    # print("Masking the input")
    use_orig = np.random.choice([True,False],1,p=[0.4,0.6])[0]
    if use_orig:
        if num_rois >= 2:
            if len(target) >= num_rois:
                if num_rois not in [2,3]:
                    proposals_dropped = random.sample(target,int(round(max(len(target),num_rois)/3)))#random.sample(target,2)##int(np.floor(len(target)/3))#random.sample(target,2)#int(np.floor(len(target)/3)))#max(len(target),num_rois) - 3)
                else:
                    proposals_dropped = random.sample(target,1)
                for proposal in proposals_dropped:
                    target.remove(proposal)
                    bboxes = proposal['bounding_box']
                    xmin,ymin,xmax,ymax = int(bboxes[0]*w),int(bboxes[1]*h),int(bboxes[2]*w),int(bboxes[3]*h)
                    # xmin,ymin,xmax,ymax = int(max(bboxes[0]-114/456)*w),int(bboxes[1]*h),int(bboxes[2]*w),int(bboxes[3]*h)

                    y = np.random.choice([0,1],size = (1,ymax-ymin,xmax-xmin),p=[0.6,0.4])
                    mask3[:,ymin:ymax,xmin:xmax] = y
                    # mask3[:,ymin:ymax,xmin:xmax] = dp(torch.Tensor(mask3[:,ymin:ymax,xmin:xmax]))

    mask3 = mask3[0]



    mask4 = np.ones((1,64,114),dtype=np.uint8)

    h,w = 64,114

    # print("Masking the input")
    use_orig = np.random.choice([True,False],1,p=[0.4,0.6])[0]
    if use_orig:
        if num_rois >= 2:
            if len(target) >= num_rois:
                if num_rois not in [2,3]:
                    proposals_dropped = random.sample(target,int(round(max(len(target),num_rois)/3)))#random.sample(target,2)##int(np.floor(len(target)/3))#random.sample(target,2)#int(np.floor(len(target)/3)))#max(len(target),num_rois) - 3)
                else:
                    proposals_dropped = random.sample(target,1)
                for proposal in proposals_dropped:
                    target.remove(proposal)
                    bboxes = proposal['bounding_box']
                    xmin,ymin,xmax,ymax = int(bboxes[0]*w),int(bboxes[1]*h),int(bboxes[2]*w),int(bboxes[3]*h)
                    # xmin,ymin,xmax,ymax = int(max(bboxes[0]-114/456)*w),int(bboxes[1]*h),int(bboxes[2]*w),int(bboxes[3]*h)

                    y = np.random.choice([0,1],size = (1,ymax-ymin,xmax-xmin),p=[0.6,0.4])
                    mask4[:,ymin:ymax,xmin:xmax] = y

                    # mask4[:,ymin:ymax,xmin:xmax] = dp(torch.Tensor(mask4[:,ymin:ymax,xmin:xmax]))

    mask4 = mask4[0]


    return (mask0,mask1,mask2,mask3,mask4)
   
def get_aug_info(init_size, params):
	size = init_size
	bbox = [0.0, 0.0, 1.0, 1.0]
	flip = False
	
	for t in params:
		if t is None:
			continue
			
		if t['transform'] == 'RandomHorizontalFlip':
			if t['flip']:
				flip = not flip
			continue
		
		if t['transform'] == 'Scale':
			if isinstance(t['size'], int):
				w, h = size
				if (w <= h and w == t['size']) or (h <= w and h == t['size']):
					continue
				if w < h:
					ow = t['size']
					oh = int(t['size'] * h / w)
					size = [ow, oh]
				else:
					oh = t['size']
					ow = int(t['size'] * w / h)
					size = [ow, oh]
			else:
				size = t['size']
			continue
			
		if t['transform'] == 'CenterCrop':
			w, h = size
			size = t['size']
			
			x1 = int(round((w - size[0]) / 2.))
			y1 = int(round((h - size[1]) / 2.))
			x2 = x1 + size[0]
			y2 = y1 + size[1]
			
		elif t['transform'] == 'CornerCrop':
			w, h = size
			size = [t['size']] * 2

			if t['crop_position'] == 'c':
				th, tw = (t['size'], t['size'])
				x1 = int(round((w - tw) / 2.))
				y1 = int(round((h - th) / 2.))
				x2 = x1 + tw
				y2 = y1 + th
			elif t['crop_position'] == 'tl':
				x1 = 0
				y1 = 0
				x2 = t['size']
				y2 = t['size']
			elif t['crop_position'] == 'tr':
				x1 = w - size
				y1 = 0
				x2 = w
				y2 = t['size']
			elif t['crop_position'] == 'bl':
				x1 = 0
				y1 = h - t['size']
				x2 = t['size']
				y2 = h
			elif t['crop_position'] == 'br':
				x1 = w - t['size']
				y1 = h - t['size']
				x2 = w
				y2 = h
			
		elif t['transform'] == 'ScaleJitteringRandomCrop':
			min_length = min(size[0], size[1])
			jitter_rate = float(t['scale']) / min_length
			
			w = int(jitter_rate * size[0])
			h = int(jitter_rate * size[1])
			size = [t['size']] * 2
			
			x1 = t['pos_x'] * (w - t['size'])
			y1 = t['pos_y'] * (h - t['size'])
			x2 = x1 + t['size']
			y2 = y1 + t['size']
			
		dl = float(x1) / w * (bbox[2] - bbox[0])
		dt = float(y1) / h * (bbox[3] - bbox[1])
		dr = float(x2) / w * (bbox[2] - bbox[0])
		db = float(y2) / h * (bbox[3] - bbox[1])
		
		if flip:
			bbox = [bbox[2] - dr, bbox[1] + dt, bbox[2] - dl, bbox[1] + db]
		else:
			bbox = [bbox[0] + dl, bbox[1] + dt, bbox[0] + dr, bbox[1] + db]

	return {'init_size': init_size, 'crop_box': bbox, 'flip': flip}
  
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
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