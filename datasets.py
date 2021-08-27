"""
 @Time    : 2021/8/27 16:00
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : CVPR2021_PDNet
 @File    : datasets.py
 @Function:
 
"""
import os
import os.path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import numpy as np
import torch.utils.data as data
from skimage import io
import torch.nn.functional as F
import random


def make_dataset(root):
    image_path = os.path.join(root, 'image')
    depth_path = os.path.join(root, 'depth_normalized')
    mask_path = os.path.join(root, 'mask_single')
    img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('.jpg')]

    return [(os.path.join(image_path, img_name + '.jpg'), os.path.join(depth_path, img_name + '.png'), os.path.join(mask_path, img_name + '.png')) for img_name in img_list]


class ImageFolder(data.Dataset):
    # image and mask should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root, joint_transform=None, transform=None, depth_transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.depth_transform = depth_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, depth_path, gt_path = self.imgs[index]

        img = Image.open(img_path).convert('RGB')
        depth_uint16 = io.imread(depth_path)
        depth = (depth_uint16 / 65535).astype(np.float32)
        depth = Image.fromarray(depth)
        target = Image.open(gt_path).convert('L')

        if self.joint_transform is not None:
            img, depth, target = self.joint_transform(img, depth, target)
        if self.transform is not None:
            img = self.transform(img)
        if self.depth_transform is not None:
            depth = self.depth_transform(depth)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, depth, target

    def collate(self, batch):
        size = [320, 352, 384, 416][random.randint(0, 3)]

        image, depth, mask = [list(item) for item in zip(*batch)]

        image = torch.stack(image, dim=0)
        image = F.interpolate(image, size=(size, size), mode="bilinear", align_corners=False)
        depth = torch.stack(depth, dim=0)
        depth = F.interpolate(depth, size=(size, size), mode="bilinear", align_corners=False)
        mask = torch.stack(mask, dim=0)
        mask = F.interpolate(mask, size=(size, size), mode="nearest")

        return image, depth, mask

    def __len__(self):
        return len(self.imgs)
