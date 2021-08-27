"""
 @Time    : 2021/8/27 16:40
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : CVPR2021_PDNet
 @File    : config.py
 @Function:
 
"""
import os

backbone_path = './backbone/resnet/resnet50-19c8e357.pth'

datasets_root = './data/RGBD-Mirror'

training_root = os.path.join(datasets_root, 'train')
testing_root = os.path.join(datasets_root, 'test')
