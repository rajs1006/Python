from __future__ import division
from __future__ import print_function

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import make_grid
import pandas as pd
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

import os
import shutil
#from os import join
from time import time
import numpy as np  
import cv2 

import pandas as pd
from time import time

import random

# %run Utils.ipynb
# %run Visualize.ipynb
# %run Model.ipynb

import torch.nn as nn
import torch.nn.functional as F

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.cm import jet

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
#%matplotlib inline

from scipy.optimize import minimize
from numpy import linalg as lin
from shapely.geometry import Polygon as P
from shapely.geometry import box

from sklearn.metrics import confusion_matrix, average_precision_score, f1_score, precision_score, recall_score, precision_recall_curve
from scipy.spatial.transform import Rotation

path = "/content/1000/captures_buffalo_1000_sift"

image_folder = os.path.join(path, 'OutputKP')
output_folder = os.path.join(path, 'output')

os.makedirs(output_folder, exist_ok=True)
os.makedirs(image_folder, exist_ok=True)



class Dataset(Dataset):

    def __init__(self, root_dir = 'captures', folder="Train/" , kp_file = 'image_%05d_img', transform=None, length=5):
        self.root_dir = os.path.join(path, root_dir, folder)
        
        self.key_pts_file = os.path.join(self.root_dir, kp_file)
        self.transform = transform
        
        files = os.listdir(self.root_dir)
        self.dataLen = int(len(files)/length)

    def __len__(self):
        return self.dataLen - 1

    def __getitem__(self, idx):
        # ignoring the first image as, they are not properly annotated
        idx = idx + 1
        
        image = imageio.imread(os.path.join(self.root_dir, "{}.png".format(self.key_pts_file %idx)))
        if(image.shape[2] == 4):
                image = image[:,:,0:3]
        
        initialKPs = np.array(pd.read_csv(os.path.join(self.root_dir, "{}-ORB.txt".format(self.key_pts_file %idx)), header=None))
        initialKPs = np.c_[ initialKPs, np.ones(initialKPs.shape[0]) ]
        
        KPs = np.array(pd.read_csv(os.path.join(self.root_dir, "{}-GT.txt".format(self.key_pts_file %idx)), header=None))
        KPs = np.c_[ KPs, np.ones(KPs.shape[0]) ]
        
        ## Area of image
        scaleArea = box(0,0,image.shape[0], image.shape[1]).area
        
        image = Image.fromarray(image)
        
        bb = np.array(pd.read_csv(os.path.join(self.root_dir, "{}-BOUND.txt".format(self.key_pts_file %idx)), header=None)).ravel()
        rot = np.array(pd.read_csv(os.path.join(self.root_dir, "{}-Rot.txt".format(self.key_pts_file %idx)), header=None, sep=' '))
        trans = np.array(pd.read_csv(os.path.join(self.root_dir, "{}-Trans.txt".format(self.key_pts_file %idx)), header=None, sep=' ')).reshape((1,3))
        
        item = {'image': image, 'original_image': np.asarray(image) ,'bb': bb, 'initial_keypoints' : initialKPs, 'keypoints': KPs, 'rot':rot, 'trans':trans,'scaleArea':scaleArea}
        if self.transform is not None:
            item = self.transform(item)
        return item
    
    
    
class DatasetReal(Dataset):

    def __init__(self, root_dir = 'captures', folder="RealTest/" , kp_file = 'image_%05d_img', transform=None, length=4):
        self.root_dir = os.path.join(path, root_dir, folder)
        
        self.key_pts_file = os.path.join(self.root_dir, kp_file)
        self.transform = transform
        
        files = os.listdir(self.root_dir)
        self.dataLen = int(len(files)/length)

    def __len__(self):
        return self.dataLen

    def __getitem__(self, idx):
        
        image = imageio.imread(os.path.join(self.root_dir, "{}.png".format(self.key_pts_file %idx)))
        if(image.shape[2] == 4):
                image = image[:,:,0:3]
        
        scaleArea = box(0,0,image.shape[0], image.shape[1]).area
        
        image = Image.fromarray(image)
        
        bb = np.array(pd.read_csv(os.path.join(self.root_dir, "{}-BOUND.txt".format(self.key_pts_file %idx)), header=None, sep=' ')).reshape((4))
        #bb = np.array([ [b[0], b[1]], [b[0], b[3]], [b[2], b[1]], [b[2], b[3]] ])
        
        camera = np.array(pd.read_csv(os.path.join(self.root_dir, "{}-Camera.txt".format(self.key_pts_file %idx)), header=None, sep=' ')).reshape((3,3))
        rot = np.array(pd.read_csv(os.path.join(self.root_dir, "{}-Rot.txt".format(self.key_pts_file %idx)), header=None, sep=' ')).reshape((3,3))
        trans = np.array(pd.read_csv(os.path.join(self.root_dir, "{}-Trans.txt".format(self.key_pts_file %idx)), header=None, sep=' ')).reshape((1,3))
        
        item = {'image': image, 'original_image': np.asarray(image), 'bb': bb, 'camera':camera, 'rot':rot, 'trans':trans, 'scaleArea':scaleArea}
        if self.transform is not None:
            item = self.transform(item)
        return item

    
#Transformations

def generate_heatmap(heatmap, pt, sigma):
    heatmap[int(pt[1])][int(pt[0])] = 1
    heatmap = cv2.GaussianBlur(heatmap, sigma, 0)
    am = np.amax(heatmap)
    heatmap /= am
    return heatmap

def render_onehot_heatmap(coord, input_shape,output_shape):
        #print(coord.shape)
        num_kps = 18
        batch_size = 1

        x = np.reshape(coord[:,0] / input_shape[1] * output_shape[1],[-1])
        y = np.reshape(coord[:,1] / input_shape[0] * output_shape[0],[-1])
        x_floor = np.floor(x)
        y_floor = np.floor(y)

        x_floor = np.clip(x_floor, 0, output_shape[1] - 1)  # fix out-of-bounds x
        y_floor = np.clip(y_floor, 0, output_shape[0] - 1)  # fix out-of-bounds y
        #print("floor ", x_floor, y_floor)
        indices_batch = np.expand_dims(\
                np.reshape(\
                np.transpose(\
                np.tile(\
                np.expand_dims(np.arange(batch_size),0)\
                ,[num_kps,1])\
                ,[1,0])\
                ,[-1]).astype(float),1)
        #print("indices_batch" , indices_batch.shape)
        indices_batch = np.concatenate([indices_batch, indices_batch, indices_batch, indices_batch], axis=0)
        indices_joint = np.expand_dims(np.tile(np.arange(num_kps),[batch_size]),1).astype(float)
        indices_joint = np.concatenate([indices_joint, indices_joint, indices_joint, indices_joint], axis=0)
        #print("indices_joint" , indices_joint.shape)
        indices_lt = np.concatenate([np.expand_dims(y_floor-1,1), np.expand_dims(x_floor-1,1)], axis=1)
        indices_lb = np.concatenate([np.expand_dims(y_floor,1), np.expand_dims(x_floor-1,1)], axis=1)
        indices_rt = np.concatenate([np.expand_dims(y_floor-1,1), np.expand_dims(x_floor,1)], axis=1)
        indices_rb = np.concatenate([np.expand_dims(y_floor,1), np.expand_dims(x_floor,1)], axis=1)

        indices = np.concatenate([indices_lt, indices_lb, indices_rt, indices_rb], axis=0)
        #print("indices" , indices.shape, np.where(indices==64))
        indices = np.concatenate([indices_batch, indices, indices_joint], axis=1).astype(int)

        prob_lt = (1 - (x - x_floor)) * (1 - (y - y_floor))
        prob_lb = (1 - (x - x_floor)) * (y - y_floor)
        prob_rt = (x - x_floor) * (1 - (y - y_floor))
        prob_rb = (x - x_floor) * (y - y_floor)
        probs = np.concatenate([prob_lt, prob_lb, prob_rt, prob_rb], axis=0)

        heatmap = scatter_nd_numpy(indices, probs, (batch_size, *output_shape, num_kps))
        normalizer = np.reshape(np.sum(heatmap,axis=(1,2)),[batch_size,1,1,num_kps])
        normalizer = np.where(np.equal(normalizer,0),np.ones_like(normalizer),normalizer)
        heatmap = heatmap / normalizer
        
        return np.squeeze(heatmap) 
    
def scatter_nd_numpy(indices, updates, shape):
    target = np.zeros(shape, dtype=updates.dtype)
    indices = tuple(indices.reshape(-1, indices.shape[-1]).T)
    updates = updates.ravel()
    np.add.at(target, indices, updates)
    return target

def render_gaussian_heatmap(coord, output_shape, input_shape, sigma):
        
        x = [i for i in range(output_shape[1])]
        y = [i for i in range(output_shape[0])]
        xx,yy = np.meshgrid(y,x, indexing='ij')
        xx = np.reshape(xx, (*output_shape,1))
        yy = np.reshape(yy, (*output_shape,1))
        
        
        x = np.reshape(coord[:,0],[1,1,coord.shape[0]]) / input_shape[1] * output_shape[1]
        y = np.reshape(coord[:,1],[1,1,coord.shape[0]]) / input_shape[0] * output_shape[0]
        
        heatmap = np.exp(-(((xx-x)/np.float(sigma))**2)/np.float(2) -(((yy-y)/np.float(sigma))**2)/np.float(2))
        #print("heatmap.shape  ", heatmap.shape)
        return heatmap * 255.


def heatmaps_to_locs(heatmaps, outSize = (64, 64)):
    heatmaps = heatmaps.cpu().numpy()
    conf = np.max(heatmaps, axis=(-2,-1))
    locs = np.argmax(heatmaps.reshape((*heatmaps.shape[:2], -1)), axis=-1)
    locs = np.stack(np.unravel_index(locs, outSize)[::-1], axis=-1) # reverse x,y
    return torch.from_numpy(np.concatenate([locs, conf[..., None]], axis=-1).astype('float64'))


class CropAndPad:

    def __init__(self, out_size=(256,256), train=True, real=True):
        self.out_size = out_size[::-1]
        self.train = train
        self.real = real

    def __call__(self, sample):
        image, bb = sample['image'], sample['bb']
       # img_size = image.size
        if self.train:
            min_x,max_y,max_x,min_y = bb[0], bb[1], bb[2], bb[3]
        else:
            if self.real:
                ## This is for Homebrew dataset
                ##min_x,max_y,max_x,min_y = bb[0]-25, bb[1]-25, bb[0] + bb[2] + 25 , bb[1] + bb[3] + 25
                ## For others, probabaly
                min_x,max_y,max_x,min_y = bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]
            else:
                min_x,max_y,max_x,min_y = bb[0], bb[1], bb[2], bb[3]

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        width, height = max_x-min_x, max_y-min_y
        
        scaleFactor = max([width, height])
        min_x = int(center_x) - int(scaleFactor)//2 
        min_y = int(center_y) - int(scaleFactor)//2
        max_x = int(center_x) + int(scaleFactor)//2 
        max_y = int(center_y) + int(scaleFactor)//2 
        ## This is for Homebrew dataset
        ## Image crop works in a way (0, 0, 10, 10) but here the 
        ## image coordinates are revresed on Y-axis nd so the crop.
        sample['image'] = image.crop(box=(min_x,min_y,max_x,max_y))
        sample['orig_image'] = image
        sample['center'] = np.array([center_x, center_y], dtype=np.float32)
        sample['width'] = width
        sample['height'] = height
        ## This sclae is used for OKS calculation
        sample['scaleArea'] = np.sqrt(np.divide(box(min_x,min_y,max_x,max_y).area, sample['scaleArea']))
        #print(sample['scaleArea'])
        
        w, h= self.out_size
        ## Crop and scale
        sample['crop'] = np.array([min_x, min_y], dtype=np.float32)
        sample['scale'] = np.array([w/scaleFactor, h/scaleFactor] , dtype=np.float32)
        
        if width != self.out_size[0]:
            sample['image'] = sample['image'].resize((w, h))
        if 'mask' in sample:
            sample['mask'] = sample['mask'].crop(box=(min_x,min_y,max_x,max_y)).resize((w, h))
        if 'keypoints' in sample:
            keypoints = sample['keypoints']
            for i in range(keypoints.shape[0]):
                if keypoints[i,0] < min_x or keypoints[i,0] > max_x or keypoints[i,1] < min_y or keypoints[i,1] > max_y:
                    keypoints[i,:] = [0,0,0]
                else:
                    keypoints[i,:2] = (keypoints[i,:2]-sample['crop'] )*sample['scale']
            sample['keypoints'] = keypoints
                
        if 'initial_keypoints' in sample:
            initial_keypoints = sample['initial_keypoints']
            for i in range(initial_keypoints.shape[0]):
                if initial_keypoints[i,0] < min_x or initial_keypoints[i,0] > max_x \
                                or initial_keypoints[i,1] < min_y or initial_keypoints[i,1] > max_y:
                    initial_keypoints[i,:] = [0,0,0]
                else:
                    initial_keypoints[i,:2] = (initial_keypoints[i,:2]-sample['crop'] )*sample['scale']
        
            sample['initial_keypoints'] = initial_keypoints
        sample.pop('bb')
        return sample

# Convert keypoint locations to heatmaps
class LocsToHeatmaps:

    def __init__(self, img_size=(256,256), out_size=(64,64), sigma=1, algo : str=None):
        self.img_size = img_size
        self.out_size = out_size
        self.x_scale = 1.0 * out_size[0]/img_size[0]
        self.y_scale = 1.0 * img_size[0]/img_size[0]
        self.sigma=sigma
        x = np.arange(0, out_size[1], dtype=np.float)
        y = np.arange(0, out_size[0], dtype=np.float)
        self.yg, self.xg = np.meshgrid(y,x, indexing='ij')
        self.algo = algo
        
        return

    def __call__(self, sample):
        sigma = 7
        gaussian_hm = np.zeros((self.out_size[0], self.out_size[1], sample['keypoints'].shape[0]))
        if self.algo == 'PoseFix':
            gaussian_hm = render_onehot_heatmap(sample['keypoints'], self.img_size, self.out_size)
            #print(gaussian_hm.shape)
            #gaussian_hm = render_gaussian_heatmap(sample['keypoints'], self.img_size, self.img_size, sigma)
            #print(gaussian_hm.shape)
        else:
            for i,keypoint in enumerate(sample['keypoints']):
                if keypoint[2] != 0:
                    gaussian_hm[:,:,i] = generate_heatmap(gaussian_hm[:,:,i], tuple(keypoint.astype(np.int) * self.x_scale), (sigma, sigma))
        sample['keypoint_locs'] = sample['keypoints'][:,:2]
        sample['visible_keypoints'] = sample['keypoints'][:,2]
        sample['keypoint_heatmaps'] = gaussian_hm
        
        gaussian_hm_init = np.zeros((self.img_size[0], self.img_size[1], sample['initial_keypoints'].shape[0]))
        #print(" gaussian_hm_init   : ", gaussian_hm_init.shape)
        if self.algo == 'PoseFix':
            gaussian_hm_init = render_gaussian_heatmap(sample['keypoints'], self.img_size, self.img_size, sigma)
        else:
            for i,initial_keypoints in enumerate(sample['initial_keypoints']):
                if initial_keypoints[2] != 0:
                    gaussian_hm_init[:,:,i] = generate_heatmap(gaussian_hm_init[:,:,i], tuple(initial_keypoints.astype(np.int) * self.x_scale ), \
                                                               (sigma, sigma))
        sample['initial_keypoints_locs'] = sample['initial_keypoints'][:,:2]
        sample['visible_initial_keypoints'] = sample['initial_keypoints'][:,2]
        sample['initial_keypoints_heatmaps'] = gaussian_hm_init
        
        return sample

# Convert numpy arrays to Tensor objects
# Permute the image dimensions
class ToTensor:

    def __init__(self, downsample_mask=False):
        self.tt = transforms.ToTensor()
        self.downsample_mask=downsample_mask

    def __call__(self, sample):
        sample['image'] = self.tt(sample['image'])
        if 'orig_image' in sample:
            sample['orig_image'] = self.tt(sample['orig_image'])
        if 'mask' in sample:
            if self.downsample_mask:
                sample['mask'] = self.tt(sample['mask'].resize((64,64), Image.ANTIALIAS))
            else:
                sample['mask'] = self.tt(sample['mask'])
        if 'in_mask' in sample:
            sample['in_mask'] = self.tt(sample['in_mask'])
            # sample['in_mask'] = sample['in_mask'].unsqueeze(0)
        if 'keypoint_heatmaps' in sample:
            sample['keypoint_heatmaps'] =\
                torch.from_numpy(sample['keypoint_heatmaps'].astype(np.float32).transpose(2,0,1))
            sample['keypoint_locs'] =\
                torch.from_numpy(sample['keypoint_locs'].astype(np.float32))
            sample['visible_keypoints'] =\
                torch.from_numpy(sample['visible_keypoints'].astype(np.float32))
            
        if 'initial_keypoints_heatmaps' in sample:
            sample['initial_keypoints_heatmaps'] =\
                torch.from_numpy(sample['initial_keypoints_heatmaps'].astype(np.float32).transpose(2,0,1))
            sample['initial_keypoints_locs'] =\
                torch.from_numpy(sample['initial_keypoints_locs'].astype(np.float32))
            sample['visible_initial_keypoints'] =\
                torch.from_numpy(sample['visible_initial_keypoints'].astype(np.float32))
            
        return sample

class Normalize:

    def __call__(self, sample):
        sample['image'] = 2*(sample['image']-0.5)
        if 'in_mask' in sample:
            sample['in_mask'] = 2*(sample['in_mask']-0.5)
        return sample
    
    
'''
Code is from https://github.com/bearpaw/pytorch-pose
Hourglass network inserted in the pre-activated Resnet
Use lr=0.01 for current version
(c) YANG, Wei
'''
__all__ = ['HourglassNet', 'hg']

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        #print("Bottleneck   ", out.shape)
        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes*block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        #print("Hourglass   ", out.shape)
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, block, num_stacks=2, num_blocks=4, num_classes=16, ch_input = 21):
        super(HourglassNet, self).__init__()

        self.inplanes = 256
        self.num_feats = 256
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(ch_input, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        #self.layer4 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats*block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            if i < num_stacks-1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu,
            )

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_
        return out


def hg(num_stacks=1, num_blocks=1, num_classes=10, ch_input=21):
    model = HourglassNet(Bottleneck, num_stacks=num_stacks, num_blocks=num_blocks, num_classes=num_classes, ch_input = ch_input)
    return model


class Trainer(object):

    def __init__(self, root_dir = 'captures', num_classes = 18, batch_size = 1, length=6, algo = None, loadModel = None):
        #self.device = torch.device('cpu')
        self.device = torch.cuda.current_device()
        torch.cuda.manual_seed(123)  # Set seed

        if algo == 'PoseFix':
          ch_input=21
        else:
          ch_input=3
        self.num_classes = num_classes
        self.model = hg(num_stacks=1, num_blocks=1, num_classes=self.num_classes, ch_input=ch_input) 

        self.model.to(self.device)
        
        # define loss function and optimizer
        self.algo = algo;
        self.ver = 'v1'
        #print("self.algo  : ", self.algo)
        if self.algo == 'PoseFix':
            train_transform_list = [CropAndPad(out_size=(256, 256)),LocsToHeatmaps(out_size=(64,64), algo = algo),ToTensor(),Normalize()]
            
            self.softmax = torch.nn.Softmax(dim=1)
            self.cross_entropy = torch.nn.CrossEntropyLoss().to(self.device)
            self.heatmap_loss = torch.nn.L1Loss().to(self.device)

            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                                 lr = 2.5e-4)
        elif self.algo == 'SemanticKDD':
            train_transform_list = [CropAndPad(out_size=(256, 256)),LocsToHeatmaps(out_size=(64, 64), algo = algo),ToTensor(),Normalize()]
            
            self.heatmap_loss = torch.nn.MSELoss().to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                                 lr = 2.5e-4)
        else:
            train_transform_list = [CropAndPad(out_size=(256, 256)),LocsToHeatmaps(out_size=(64, 64), algo = algo),ToTensor(),Normalize()]
            
            self.heatmap_loss = torch.nn.MSELoss().to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                                 lr = 2.5e-4)
        
        self.train_ds = Dataset(root_dir=root_dir, transform=transforms.Compose(train_transform_list), length=length)
        self.train_data_loader = DataLoader(self.train_ds, batch_size=batch_size,
                                            num_workers=8,
                                            pin_memory=True,
                                            shuffle=True)
        self.loadModel = loadModel
        if self.loadModel is not None:
          print('Loadig exsiting checkpoint')
          self.checkpoint = torch.load(self.loadModel)

          self.startEpoch = self.checkpoint['epoch']
          self.model.load_state_dict(self.checkpoint['model'])
          self.optimizer.load_state_dict(self.checkpoint['optimizer'])
          self.loss = self.checkpoint['loss']
          
          self.startEpoch  = self.startEpoch  + 1


        self.summary_iters = []
        self.losses = []
        self.pcks = []

    def train(self, epochs = 400):
        
        self.total_step_count = 0
        start_time = time()

        startEpoch = 1
        if self.loadModel is not None:
          startEpoch = self.startEpoch
        
        for epoch in range(startEpoch, epochs + 1):

            running_loss = 0.0
            pBar = tqdm(enumerate(self.train_data_loader))
            for step, batch in pBar:
              
                self.model.train()
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}

                self.optimizer.zero_grad()
                # print(batch['image'].shape, batch['keypoint_heatmaps'].shape, batch['initial_keypoints_heatmaps'].shape, \
                #       torch.cat([batch['image'], batch['initial_keypoints_heatmaps']], 1 ).shape)
                
                if self.algo == 'PoseFix':
                    pred_heatmap_list = self.model(torch.cat([batch['image'], batch['initial_keypoints_heatmaps']], 1 ).to(self.device))
                    kp_transpose = batch['keypoint_heatmaps'].transpose(1,2).transpose(2,3)
                    #print(kp_transpose[:,:,:, -1].shape, pred_heatmap_list[-1].long().shape)
                    cros_entropy_loss = self.cross_entropy(pred_heatmap_list[-1], kp_transpose[:,:,:, -1].long())
                    l1_loss = self.heatmap_loss(batch['keypoint_heatmaps'], pred_heatmap_list[-1])

                    self.loss = cros_entropy_loss + l1_loss    
                else:
                    pred_heatmap_list = self.model(batch['image'].to(self.device))
                    self.loss = self.heatmap_loss(batch['keypoint_heatmaps'], pred_heatmap_list[-1])
                                                       
                self.loss.backward()
                self.optimizer.step()                                          
                
                self.total_step_count += 1
                
                # print loss statistics
                running_loss += self.loss.item()

                pBar.set_description("Processing epoch : %s step %s loss : %s" % (epoch, step+1, running_loss))

            if epoch % 50 == 0:
              checkpoint_file = '{}/model_checkpoint{}_{}.pt' if self.algo is None else '{}/{}-model_checkpoint{}_{}.pt'.format(output_folder, self.algo, self.ver, epoch)
              checkpoint = {'epoch' : epoch  , 'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'loss': self.loss}

              #print('saving model to ', checkpoint_file)
              torch.save(checkpoint, checkpoint_file)
              #saveFileToDrive(output_folder, '{}-model_checkpoint{}_{}.pt'.format(self.algo, self.ver, epoch))

        checkpoint_file = '{}/model_checkpoint{}.pt' if self.algo is None else '{}/{}-model_checkpoint{}.pt'.format(output_folder, self.algo, self.ver)
        checkpoint = {'epoch' : epochs , 'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'loss': self.loss}
        print('saving model to ', checkpoint_file)
        torch.save(checkpoint, checkpoint_file)
        
        
            
def main():
    
    #algo = 'PoseFix'
    algo='SemanticKDD'
    
    print('Starting training...')
    trainer = Trainer(algo = algo, batch_size = 1, num_classes=51, length=10)
    trainer.train(epochs = 500)

    
main()