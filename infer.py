"""
 @Time    : 2021/8/27 15:02
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : CVPR2021_PDNet
 @File    : infer.py
 @Function:
 
"""
import time
import datetime
import os

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict
from numpy import mean
import numpy as np
from skimage import io

from config import *
from misc import *
from pdnet import PDNet

torch.manual_seed(2021)
device_ids = [0]
torch.cuda.set_device(device_ids[0])

results_path = './results'
check_mkdir(results_path)
ckpt_path = './ckpt'
exp_name = 'PDNet'
args = {
    'scale': 416,
    'save_results': True,
}

print(torch.__version__)

img_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
depth_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
])

to_pil = transforms.ToPILImage()

to_test = OrderedDict([
    ('RGBD-Mirror', testing_root),
])

results = OrderedDict()


def main():
    net = PDNet(backbone_path).cuda(device_ids[0])

    print('Load {}.pth for testing'.format(exp_name))
    net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name + '.pth')))
    print('Load {}.pth succeed!'.format(exp_name))

    net.eval()
    with torch.no_grad():
        start = time.time()
        for name, root in to_test.items():
            time_list = []
            image_path = os.path.join(root, 'image')
            depth_path = os.path.join(root, 'depth_normalized')

            if args['save_results']:
                check_mkdir(os.path.join(results_path, exp_name))

            img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('jpg')]
            for idx, img_name in enumerate(img_list):
                img = Image.open(os.path.join(image_path, img_name + '.jpg')).convert('RGB')
                depth = io.imread(os.path.join(depth_path, img_name + '.png'))
                depth = (depth / 65535).astype(np.float32)
                depth = np.expand_dims(depth, 2)
                depth = transforms.ToPILImage()(depth)

                w, h = img.size
                img_var = Variable(img_transform(img).unsqueeze(0)).cuda(device_ids[0])
                depth_var = Variable(depth_transform(depth).unsqueeze(0)).cuda(device_ids[0])

                start_each = time.time()
                prediction1 = net(img_var, depth_var)
                time_each = time.time() - start_each
                time_list.append(time_each)

                prediction1 = np.array(transforms.Resize((h, w))(to_pil(prediction1.data.squeeze(0).cpu())))

                if args['save_results']:
                    Image.fromarray(prediction1).convert('L').save(os.path.join(results_path, exp_name, img_name + '.png'))

            print(('{}'.format(exp_name)))
            print("{}'s average Time Is : {:.1f} ms".format(name, mean(time_list) * 1000))
            print("{}'s average Time Is : {:.1f} fps".format(name, 1 / mean(time_list)))

    end = time.time()
    print("Total Testing Time: {}".format(str(datetime.timedelta(seconds=int(end - start)))))


if __name__ == '__main__':
    main()
