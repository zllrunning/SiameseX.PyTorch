# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F
from models import builder


class SiamRPN(nn.Module):

    def __init__(self, tracker_name):
        super(SiamRPN, self).__init__()
        self.tracker_name = tracker_name
        self.model = getattr(builder, tracker_name)()

    def forward(self, detection):
        detection_feature = self.model.features(detection)

        conv_score = self.model.conv_cls2(detection_feature)
        conv_regression = self.model.conv_r2(detection_feature)

        pred_score = self.model.xcorr(self.cls1_kernel, conv_score, 10)
        pred_regression = self.model.regress_adjust(self.xcorr(self.r1_kernel, conv_regression, 20))

        return pred_regression, pred_score

    def temple(self, z):
        z_f = self.model.features(z)
        self.r1_kernel = self.model.conv_r1(z_f)
        self.cls1_kernel = self.model.conv_cls1(z_f)


class SiamRPNVGG(SiamRPN):
    def __init__(self, tracker_name):
        super(SiamRPNVGG, self).__init__(tracker_name)
        self.cfg = {'lr': 0.45, 'window_influence': 0.44, 'penalty_k': 0.04, 'instance_size': 255, 'adaptive': False} # 0.355


class SiamRPNPP(nn.Module):
    def __init__(self, tracker_name):
        super(SiamRPNPP, self).__init__()
        self.tracker_name = tracker_name
        self.model = getattr(builder, tracker_name)()

    def temple(self, z):
        zf = self.model.features(z)
        zf = self.model.neck(zf)
        self.zf = zf

    def forward(self, x):
        xf = self.model.features(x)
        xf = self.model.neck(xf)
        print(xf[0].size())
        cls, loc = self.model.head(self.zf, xf)
        return loc, cls


class SiamRPNPPRes50(SiamRPNPP):
    def __init__(self, tracker_name='SiamRPNPP'):
        super(SiamRPNPPRes50, self).__init__(tracker_name)
        self.cfg = {'lr': 0.45, 'window_influence': 0.44, 'penalty_k': 0.04, 'instance_size': 255, 'adaptive': False} # 0.355












# class SiamRPN(nn.Module):
#     def __init__(self, size=2, feature_out=512, anchor=5):
#         configs = [3, 96, 256, 384, 384, 256]
#         configs = list(map(lambda x: 3 if x==3 else x*size, configs))
#         feat_in = configs[-1]
#         super(SiamRPN, self).__init__()
#         self.featureExtract = nn.Sequential(
#             nn.Conv2d(configs[0], configs[1], kernel_size=11, stride=2),
#             nn.BatchNorm2d(configs[1]),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(configs[1], configs[2], kernel_size=5),
#             nn.BatchNorm2d(configs[2]),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(configs[2], configs[3], kernel_size=3),
#             nn.BatchNorm2d(configs[3]),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(configs[3], configs[4], kernel_size=3),
#             nn.BatchNorm2d(configs[4]),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(configs[4], configs[5], kernel_size=3),
#             nn.BatchNorm2d(configs[5]),
#         )
#
#         self.anchor = anchor
#         self.feature_out = feature_out
#
#         self.conv_r1 = nn.Conv2d(feat_in, feature_out*4*anchor, 3)
#         self.conv_r2 = nn.Conv2d(feat_in, feature_out, 3)
#         self.conv_cls1 = nn.Conv2d(feat_in, feature_out*2*anchor, 3)
#         self.conv_cls2 = nn.Conv2d(feat_in, feature_out, 3)
#         self.regress_adjust = nn.Conv2d(4*anchor, 4*anchor, 1)
#
#         self.r1_kernel = []
#         self.cls1_kernel = []
#
#         self.cfg = {}
#
#     def forward(self, x):
#         x_f = self.featureExtract(x)
#         return self.regress_adjust(F.conv2d(self.conv_r2(x_f), self.r1_kernel)), \
#                F.conv2d(self.conv_cls2(x_f), self.cls1_kernel)
#
#     def temple(self, z):
#         z_f = self.featureExtract(z)
#         r1_kernel_raw = self.conv_r1(z_f)
#         cls1_kernel_raw = self.conv_cls1(z_f)
#         kernel_size = r1_kernel_raw.data.size()[-1]
#         self.r1_kernel = r1_kernel_raw.view(self.anchor*4, self.feature_out, kernel_size, kernel_size)
#         self.cls1_kernel = cls1_kernel_raw.view(self.anchor*2, self.feature_out, kernel_size, kernel_size)
#
#
# class SiamRPNBIG(SiamRPN):
#     def __init__(self):
#         super(SiamRPNBIG, self).__init__(size=2)
#         self.cfg = {'lr':0.295, 'window_influence': 0.42, 'penalty_k': 0.055, 'instance_size': 271, 'adaptive': True} # 0.383
#
#
# class SiamRPNvot(SiamRPN):
#     def __init__(self):
#         super(SiamRPNvot, self).__init__(size=1, feature_out=256)
#         self.cfg = {'lr':0.45, 'window_influence': 0.44, 'penalty_k': 0.04, 'instance_size': 271, 'adaptive': False} # 0.355
#
#
# class SiamRPNotb(SiamRPN):
#     def __init__(self):
#         super(SiamRPNotb, self).__init__(size=1, feature_out=256)
#         self.cfg = {'lr': 0.30, 'window_influence': 0.40, 'penalty_k': 0.22, 'instance_size': 271, 'adaptive': False} # 0.655
