"""
 @Time    : 2021/8/27 15:52
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : CVPR2021_PDNet
 @File    : joint_transforms.py
 @Function:
 
"""
import random

from PIL import Image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, depth, mask):
        assert img.size == mask.size
        assert img.size == depth.size
        for t in self.transforms:
            img, depth, mask = t(img, depth, mask)
        return img, depth, mask


class RandomHorizontallyFlip(object):
    def __call__(self, img, depth, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), depth.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, depth, mask


class Resize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)  PIL: (w, h)

    def __call__(self, img, depth, mask):
        assert img.size == mask.size
        assert img.size == depth.size

        return img.resize(self.size, Image.BILINEAR), depth.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)
