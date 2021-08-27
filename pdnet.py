"""
 @Time    : 2021/8/27 14:35
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : CVPR2021_PDNet
 @File    : pdnet.py
 @Function:
 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import backbone.resnet.resnet as resnet


class PM(nn.Module):
    """ positioning module """

    def __init__(self, in_dim_x, in_dim_y):
        super(PM, self).__init__()
        self.in_dim_x = in_dim_x
        self.in_dim_y = in_dim_y
        self.in_dim_xy = in_dim_x + in_dim_y
        self.in_dim_2xy = (in_dim_x + in_dim_y) * 2

        # discontinuity
        self.fusion1 = nn.Sequential(nn.Conv2d(self.in_dim_xy, self.in_dim_xy, 3, 1, 1),
                                     nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())

        self.local_main = nn.Sequential(nn.Conv2d(self.in_dim_xy, self.in_dim_xy, 3, 1, 1, 1),
                                        nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())
        self.context_main = nn.Sequential(nn.Conv2d(self.in_dim_xy, self.in_dim_xy, 5, 1, 4, 2),
                                          nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())
        self.global_main = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                         nn.Conv2d(self.in_dim_xy, self.in_dim_xy, 1, 1, 0),
                                         nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())
        self.bn_main1 = nn.BatchNorm2d(self.in_dim_xy)
        self.relu_main1 = nn.ReLU()
        self.bn_main2 = nn.BatchNorm2d(self.in_dim_xy)
        self.relu_main2 = nn.ReLU()

        self.local_rgb = nn.Sequential(nn.Conv2d(self.in_dim_x, self.in_dim_x, 3, 1, 1, 1),
                                       nn.BatchNorm2d(self.in_dim_x), nn.ReLU())
        self.context_rgb = nn.Sequential(nn.Conv2d(self.in_dim_x, self.in_dim_x, 5, 1, 4, 2),
                                         nn.BatchNorm2d(self.in_dim_x), nn.ReLU())
        self.global_rgb = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                        nn.Conv2d(self.in_dim_x, self.in_dim_x, 1, 1, 0),
                                        nn.BatchNorm2d(self.in_dim_x), nn.ReLU())
        self.bn_rgb1 = nn.BatchNorm2d(self.in_dim_x)
        self.relu_rgb1 = nn.ReLU()
        self.bn_rgb2 = nn.BatchNorm2d(self.in_dim_x)
        self.relu_rgb2 = nn.ReLU()

        self.local_depth = nn.Sequential(nn.Conv2d(self.in_dim_y, self.in_dim_y, 3, 1, 1, 1),
                                         nn.BatchNorm2d(self.in_dim_y), nn.ReLU())
        self.context_depth = nn.Sequential(nn.Conv2d(self.in_dim_y, self.in_dim_y, 5, 1, 4, 2),
                                           nn.BatchNorm2d(self.in_dim_y), nn.ReLU())
        self.global_depth = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                          nn.Conv2d(self.in_dim_y, self.in_dim_y, 1, 1, 0),
                                          nn.BatchNorm2d(self.in_dim_y), nn.ReLU())
        self.bn_depth1 = nn.BatchNorm2d(self.in_dim_y)
        self.relu_depth1 = nn.ReLU()
        self.bn_depth2 = nn.BatchNorm2d(self.in_dim_y)
        self.relu_depth2 = nn.ReLU()

        self.fusion2 = nn.Sequential(nn.Conv2d(self.in_dim_2xy, self.in_dim_xy, 3, 1, 1),
                                     nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())

        # similarity
        self.fusion3 = nn.Sequential(nn.Conv2d(self.in_dim_xy, self.in_dim_xy, 3, 1, 1),
                                     nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())

        self.value = nn.Conv2d(self.in_dim_xy, self.in_dim_xy, 1, 1, 0)

        self.gap_rgb = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp_rgb = nn.Sequential(nn.Conv2d(self.in_dim_x, self.in_dim_x // 8, 1, 1, 0),
                                     nn.BatchNorm2d(self.in_dim_x // 8), nn.ReLU(),
                                     nn.Conv2d(self.in_dim_x // 8, 1, 1, 1, 0))

        self.gap_depth = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp_depth = nn.Sequential(nn.Conv2d(self.in_dim_y, self.in_dim_y // 8, 1, 1, 0),
                                       nn.BatchNorm2d(self.in_dim_y // 8), nn.ReLU(),
                                       nn.Conv2d(self.in_dim_y // 8, 1, 1, 1, 0))

        self.softmax_weight = nn.Softmax(dim=1)

        self.query_rgb = nn.Conv2d(self.in_dim_x, self.in_dim_x // 8, 1, 1, 0)
        self.key_rgb = nn.Conv2d(self.in_dim_x, self.in_dim_x // 8, 1, 1, 0)

        self.query_depth = nn.Conv2d(self.in_dim_y, self.in_dim_y // 8, 1, 1, 0)
        self.key_depth = nn.Conv2d(self.in_dim_y, self.in_dim_y // 8, 1, 1, 0)

        self.softmax_dependency = nn.Softmax(dim=-1)

        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x, y):
        """
            inputs :
                x : input rgb feature maps (B X C1 X H X W)
                y : input depth feature maps (B X C2 X H X W)
            returns :
                out : enhanced feature maps (B X (C1+C2) X H X W)
        """
        # discontinuity
        fusion1 = self.fusion1(torch.cat((x, y), 1))

        local_main = self.local_main(fusion1)
        context_main = self.context_main(fusion1)
        global_main = self.global_main(fusion1).expand_as(fusion1)
        contrast_main1 = self.relu_main1(self.bn_main1(local_main - context_main))
        contrast_main2 = self.relu_main2(self.bn_main2(local_main - global_main))
        contrast_main = contrast_main1 + contrast_main2

        local_rgb = self.local_rgb(x)
        context_rgb = self.context_rgb(x)
        global_rgb = self.global_rgb(x).expand_as(x)
        contrast_rgb1 = self.relu_rgb1(self.bn_rgb1(local_rgb - context_rgb))
        contrast_rgb2 = self.relu_rgb2(self.bn_rgb2(local_rgb - global_rgb))
        contrast_rgb = contrast_rgb1 + contrast_rgb2

        local_depth = self.local_depth(y)
        context_depth = self.context_depth(y)
        global_depth = self.global_depth(y).expand_as(y)
        contrast_depth1 = self.relu_depth1(self.bn_depth1(local_depth - context_depth))
        contrast_depth2 = self.relu_depth2(self.bn_depth2(local_depth - global_depth))
        contrast_depth = contrast_depth1 + contrast_depth2

        concatenation = torch.cat((contrast_main, contrast_rgb, contrast_depth), 1)
        fusion2 = self.fusion2(concatenation)

        # similarity
        fusion3 = self.fusion3(torch.cat((x, y), 1))
        B, C, H, W = fusion3.size()
        value = self.value(fusion3).view(B, -1, H * W)

        weight_rgb = self.mlp_rgb(self.gap_rgb(x))
        weight_depth = self.mlp_depth(self.gap_depth(y))
        softmax_weight = self.softmax_weight(torch.cat((weight_rgb, weight_depth), 1))
        weight_rgb_normalized, weight_depth_normalized = softmax_weight.split(1, dim=1)

        query_rgb = self.query_rgb(x).view(B, -1, H * W).permute(0, 2, 1)
        key_rgb = self.key_rgb(x).view(B, -1, H * W)
        energy_rgb = torch.bmm(query_rgb, key_rgb)
        energy_rgb = energy_rgb * weight_rgb_normalized.squeeze(1).expand_as(energy_rgb)

        query_depth = self.query_depth(y).view(B, -1, H * W).permute(0, 2, 1)
        key_depth = self.key_depth(y).view(B, -1, H * W)
        energy_depth = torch.bmm(query_depth, key_depth)
        energy_depth = energy_depth * weight_depth_normalized.squeeze(1).expand_as(energy_depth)

        energy = energy_rgb + energy_depth
        attention_element = self.softmax_dependency(energy)

        fusion4 = torch.bmm(value, attention_element.permute(0, 2, 1)).view(B, C, H, W)

        fusion4 = self.gamma * fusion4 + fusion3

        # final output features
        fusion = fusion2 + fusion4

        return fusion, weight_rgb_normalized, weight_depth_normalized, self.gamma


class DM(nn.Module):
    """ delineating module """

    def __init__(self, in_dim_x, in_dim_y, in_dim_z):
        super(DM, self).__init__()
        self.in_dim_x = in_dim_x
        self.in_dim_y = in_dim_y
        self.in_dim_xy = in_dim_x + in_dim_y
        self.in_dim_2xy = (in_dim_x + in_dim_y) * 2
        self.in_dim_z = in_dim_z

        self.up_main = nn.Sequential(nn.Conv2d(self.in_dim_z, self.in_dim_xy, 3, 1, 1),
                                     nn.BatchNorm2d(self.in_dim_xy), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))
        self.up_rgb = nn.Sequential(nn.Conv2d(self.in_dim_z, self.in_dim_x, 3, 1, 1),
                                    nn.BatchNorm2d(self.in_dim_x), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))
        self.up_depth = nn.Sequential(nn.Conv2d(self.in_dim_z, self.in_dim_y, 3, 1, 1),
                                      nn.BatchNorm2d(self.in_dim_y), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))

        self.fusion1 = nn.Sequential(nn.Conv2d(self.in_dim_xy, self.in_dim_xy, 3, 1, 1),
                                     nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())

        self.local_main = nn.Sequential(nn.Conv2d(self.in_dim_xy, self.in_dim_xy, 3, 1, 1, 1),
                                        nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())
        self.context_main = nn.Sequential(nn.Conv2d(self.in_dim_xy, self.in_dim_xy, 5, 1, 4, 2),
                                          nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())
        self.bn_main = nn.BatchNorm2d(self.in_dim_xy)
        self.relu_main = nn.ReLU()

        self.local_rgb = nn.Sequential(nn.Conv2d(self.in_dim_x, self.in_dim_x, 3, 1, 1, 1),
                                       nn.BatchNorm2d(self.in_dim_x), nn.ReLU())
        self.context_rgb = nn.Sequential(nn.Conv2d(self.in_dim_x, self.in_dim_x, 5, 1, 4, 2),
                                         nn.BatchNorm2d(self.in_dim_x), nn.ReLU())
        self.bn_rgb = nn.BatchNorm2d(self.in_dim_x)
        self.relu_rgb = nn.ReLU()

        self.local_depth = nn.Sequential(nn.Conv2d(self.in_dim_y, self.in_dim_y, 3, 1, 1, 1),
                                         nn.BatchNorm2d(self.in_dim_y), nn.ReLU())
        self.context_depth = nn.Sequential(nn.Conv2d(self.in_dim_y, self.in_dim_y, 5, 1, 4, 2),
                                           nn.BatchNorm2d(self.in_dim_y), nn.ReLU())
        self.bn_depth = nn.BatchNorm2d(self.in_dim_y)
        self.relu_depth = nn.ReLU()

        self.fusion2 = nn.Sequential(nn.Conv2d(self.in_dim_2xy, self.in_dim_xy, 3, 1, 1),
                                     nn.BatchNorm2d(self.in_dim_xy), nn.ReLU())

    def forward(self, x, y, z):
        """
            inputs :
                x : input rgb feature maps (B X C1 X H1 X W1)
                y : input depth feature maps (B X C2 X H1 X W1)
                z : input higher-level feature maps (B X C3 X H2 X W2)
            returns :
                out : enhanced feature maps (B X (C1+C2) X H1 X W1)
        """
        up_main = self.up_main(z)
        fusion1 = self.fusion1(torch.cat((x, y), 1))
        feature_main = fusion1 + up_main
        local_main = self.local_main(feature_main)
        context_main = self.context_main(feature_main)
        contrast_main = self.relu_main(self.bn_main(local_main - context_main))

        up_rgb = self.up_rgb(z)
        feature_rgb = x + up_rgb
        local_rgb = self.local_rgb(feature_rgb)
        context_rgb = self.context_rgb(feature_rgb)
        contrast_rgb = self.relu_rgb(self.bn_rgb(local_rgb - context_rgb))

        up_depth = self.up_depth(z)
        feature_depth = y + up_depth
        local_depth = self.local_depth(feature_depth)
        context_depth = self.context_depth(feature_depth)
        contrast_depth = self.relu_depth(self.bn_depth(local_depth - context_depth))

        concatenation = torch.cat((contrast_main, contrast_rgb, contrast_depth), 1)
        fusion2 = self.fusion2(concatenation)

        return fusion2


###################################################################
# ########################## NETWORK ##############################
###################################################################
class PDNet(nn.Module):
    def __init__(self, backbone_path=None):
        super(PDNet, self).__init__()
        # params

        # backbone
        resnet50 = resnet.resnet50(backbone_path)
        self.layer0 = nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu)
        self.layer1 = nn.Sequential(resnet50.maxpool, resnet50.layer1)
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4

        # depth feature extraction
        self.depth_conv0 = nn.Sequential(nn.Conv2d(1, 8, 3, 1, 1), nn.BatchNorm2d(8), nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.depth_conv1 = nn.Sequential(nn.Conv2d(8, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.depth_conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.depth_conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.depth_conv4 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        # channel reduction
        self.cr4 = nn.Sequential(nn.Conv2d(2048, 512, 1, 1, 0), nn.BatchNorm2d(512), nn.ReLU())
        self.cr3 = nn.Sequential(nn.Conv2d(1024, 256, 1, 1, 0), nn.BatchNorm2d(256), nn.ReLU())
        self.cr2 = nn.Sequential(nn.Conv2d(512, 128, 1, 1, 0), nn.BatchNorm2d(128), nn.ReLU())
        self.cr1 = nn.Sequential(nn.Conv2d(256, 64, 1, 1, 0), nn.BatchNorm2d(64), nn.ReLU())

        # positioning
        self.pm = PM(512, 128)

        # delineating
        self.dm3 = DM(256, 64, 640)
        self.dm2 = DM(128, 32, 320)
        self.dm1 = DM(64, 16, 160)

        # predict
        self.predict4 = nn.Conv2d(640, 1, 3, 1, 1)
        self.predict3 = nn.Conv2d(320, 1, 3, 1, 1)
        self.predict2 = nn.Conv2d(160, 1, 3, 1, 1)
        self.predict1 = nn.Conv2d(80, 1, 3, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x, y):
        # x: [batch_size, channel=3, h, w]
        # y: [batch_size, channel=1, h, w]

        layer0 = self.layer0(x)  # [-1, 64, h/2, w/2]
        layer1 = self.layer1(layer0)  # [-1, 256, h/4, w/4]
        layer2 = self.layer2(layer1)  # [-1, 512, h/8, w/8]
        layer3 = self.layer3(layer2)  # [-1, 1024, h/16, w/16]
        layer4 = self.layer4(layer3)  # [-1, 2048, h/32, w/32]

        depth_conv0 = self.depth_conv0(y)
        depth_conv1 = self.depth_conv1(depth_conv0)
        depth_conv2 = self.depth_conv2(depth_conv1)
        depth_conv3 = self.depth_conv3(depth_conv2)
        depth_conv4 = self.depth_conv4(depth_conv3)

        # channel reduction
        cr4 = self.cr4(layer4)
        cr3 = self.cr3(layer3)
        cr2 = self.cr2(layer2)
        cr1 = self.cr1(layer1)

        # positioning
        pm, weight_rgb_normalized, weight_depth_normalized, gamma = self.pm(cr4, depth_conv4)

        # delineating
        dm3 = self.dm3(cr3, depth_conv3, pm)
        dm2 = self.dm2(cr2, depth_conv2, dm3)
        dm1 = self.dm1(cr1, depth_conv1, dm2)

        # predict
        predict4 = self.predict4(pm)
        predict3 = self.predict3(dm3)
        predict2 = self.predict2(dm2)
        predict1 = self.predict1(dm1)

        # rescale
        predict4 = F.interpolate(predict4, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict3 = F.interpolate(predict3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.interpolate(predict2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict1 = F.interpolate(predict1, size=x.size()[2:], mode='bilinear', align_corners=True)

        if self.training:
            return predict4, predict3, predict2, predict1

        return torch.sigmoid(predict4), torch.sigmoid(predict3), torch.sigmoid(predict2), torch.sigmoid(predict1)
