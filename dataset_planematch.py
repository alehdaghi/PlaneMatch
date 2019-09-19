import os
import torch

from skimage import io, transform
import numpy as np
import torchvision
import torch.nn as nn
import math
from cropextract import *
from torch.utils.data import Dataset, DataLoader
import scipy.misc as smi
import imageio

import scipy.io as sio
import scipy.misc as smi
import pickle
from matplotlib import pyplot as plt

class PlanarPatchDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        f = open(os.path.join(root_dir, 'train.pkl'), 'rb')
        self.landmarks_frame = pickle.load(f)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)
        #return 256

    def __getitem__(self, idx):

        a = str(self.landmarks_frame[idx][1][0])
        p = str(self.landmarks_frame[idx][1][1])
        n = str(self.landmarks_frame[idx][1][2])

        rgb_anchor = os.path.join(self.root_dir, 'images/'+a+'.rgb.npy')
        normal_anchor = os.path.join(self.root_dir, 'images/'+a+'.n.npy')
        mask_anchor = os.path.join(self.root_dir, 'planes/' + self.landmarks_frame[idx][0][0]+'.npy')

        rgb_positive = os.path.join(self.root_dir, 'images/'+ p + '.rgb.npy')
        normal_positive = os.path.join(self.root_dir, 'images/'+ p + '.n.npy')
        mask_positive = os.path.join(self.root_dir, 'planes/' + self.landmarks_frame[idx][0][1]+'.npy')

        rgb_negative = os.path.join(self.root_dir, 'images/'+n + '.rgb.npy')
        normal_negative = os.path.join(self.root_dir, 'images/'+n + '.n.npy')
        mask_negative = os.path.join(self.root_dir, 'planes/' + self.landmarks_frame[idx][0][2]+'.npy')




        size = (240, 320)
        # rgb_anchor_image        = cv2.resize(imageio.imread(rgb_anchor), dsize=size, interpolation=cv2.INTER_LINEAR)
        # normal_anchor_image     = cv2.resize(imageio.imread(normal_anchor), dsize=size, interpolation=cv2.INTER_LINEAR)
        # mask_anchor_image       = cv2.resize(imageio.imread(mask_anchor), dsize=size, interpolation=cv2.INTER_LINEAR)
        # rgb_negative_image      = cv2.resize(imageio.imread(rgb_negative), dsize=size, interpolation=cv2.INTER_LINEAR)
        # normal_negative_image   = cv2.resize(imageio.imread(normal_negative), dsize=size, interpolation=cv2.INTER_LINEAR)
        # mask_negative_image     = cv2.resize(imageio.imread(mask_negative), dsize=size, interpolation=cv2.INTER_LINEAR)
        # rgb_positive_image      = cv2.resize(imageio.imread(rgb_positive), dsize=size, interpolation=cv2.INTER_LINEAR)
        # normal_positive_image   = cv2.resize(imageio.imread(normal_positive), dsize=size, interpolation=cv2.INTER_LINEAR)
        # mask_positive_image     = cv2.resize(imageio.imread(mask_positive), dsize=size, interpolation=cv2.INTER_LINEAR)

        rgb_anchor_image        = np.load(rgb_anchor)
        normal_anchor_image     = np.load(normal_anchor)
        mask_anchor_image       = np.load(mask_anchor)
        rgb_negative_image      = np.load(rgb_negative)
        normal_negative_image   = np.load(normal_negative)
        mask_negative_image     = np.load(mask_negative)
        rgb_positive_image      = np.load(rgb_positive)
        normal_positive_image   = np.load(normal_positive)
        mask_positive_image     = np.load(mask_positive)

        # m = 0.1 * np.stack((mask_anchor_image.astype(np.uint8), np.zeros(size), np.zeros(size)) , axis=-1)
        # plt.imshow( normal_anchor_image + m.astype(np.int))
        # plt.show()
        #
        # m = 0.1 * np.stack((mask_positive_image.astype(np.uint8), np.zeros(size), np.zeros(size)), axis=-1)
        # plt.imshow(normal_positive_image + m.astype(np.int))
        # plt.show()
        #
        # m = 0.1 * np.stack((mask_negative_image.astype(np.uint8), np.zeros(size), np.zeros(size)), axis=-1)
        # plt.imshow(normal_negative_image + m.astype(np.int))
        # plt.show()

        #plt.imshow(np.ma.array(mask_anchor_image, mask=~mask_anchor_image))


        sample = {
            'rgb_anchor_image': rgb_anchor_image,
            'normal_anchor_image': normal_anchor_image,
            'mask_anchor_image': mask_anchor_image,
            'rgb_negative_image': rgb_negative_image,
            'normal_negative_image': normal_negative_image,
            'mask_negative_image': mask_negative_image,
            'rgb_positive_image': rgb_positive_image,
            'normal_positive_image': normal_positive_image,
            'mask_positive_image': mask_positive_image
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        rgb_anchor_image,\
        normal_anchor_image,\
        mask_anchor_image,\
        rgb_negative_image,\
        normal_negative_image,\
        mask_negative_image,\
        rgb_positive_image,\
        normal_positive_image,\
        mask_positive_image = \
            sample['rgb_anchor_image'],\
            sample['normal_anchor_image'],\
            sample['mask_anchor_image'],\
            sample['rgb_negative_image'],\
            sample['normal_negative_image'],\
            sample['mask_negative_image'],\
            sample['rgb_positive_image'],\
            sample['normal_positive_image'],\
            sample['mask_positive_image']



        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        rgb_anchor_image = rgb_anchor_image.astype(np.float).transpose((2, 0, 1)) - 128
        rgb_negative_image = rgb_negative_image.astype(np.float).transpose((2, 0, 1)) - 128
        rgb_positive_image = rgb_positive_image.astype(np.float).transpose((2, 0, 1)) - 128

        normal_anchor_image = normal_anchor_image.astype(np.float).transpose((2, 0, 1)) - 128
        normal_negative_image = normal_negative_image.astype(np.float).transpose((2, 0, 1)) - 128
        normal_positive_image = normal_positive_image.astype(np.float).transpose((2, 0, 1)) - 128

        mask_anchor_image = mask_anchor_image.astype(np.float) - 128
        mask_negative_image = mask_negative_image.astype(np.float) - 128
        mask_positive_image = mask_positive_image.astype(np.float) - 128

        return {
            'rgb_anchor_image': torch.from_numpy(rgb_anchor_image),
            'normal_anchor_image': torch.from_numpy(normal_anchor_image),
            'mask_anchor_image': torch.from_numpy(mask_anchor_image),
            'rgb_negative_image': torch.from_numpy(rgb_negative_image),
            'normal_negative_image': torch.from_numpy(normal_negative_image),
            'mask_negative_image': torch.from_numpy(mask_negative_image),
            'rgb_positive_image': torch.from_numpy(rgb_positive_image),
            'normal_positive_image': torch.from_numpy(normal_positive_image),
            'mask_positive_image': torch.from_numpy(mask_positive_image)
        }

