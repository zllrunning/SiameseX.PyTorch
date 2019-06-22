import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from .modules import Bottleneck_CI, Bottleneck_BIG_CI, ResNet, Inception, InceptionM, ResNeXt, ResNetPP, BasicBlock, Bottleneck


eps = 1e-5


class AlexNet(nn.Module):
    """
    AlexNet backbone
    """
    def __init__(self):
        super(AlexNet, self).__init__()
        self.feature_channel = 256
        self.feature = nn.Sequential(
            # conv1
            nn.Conv2d(3, 96, 11, 2),
            nn.BatchNorm2d(96, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv2
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv3
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(384, 256, 3, 1, groups=2))

    def forward(self, x):
        x = self.feature(x)
        return x


class Vgg(nn.Module):
    """
    Vgg backbone
    """
    def __init__(self):
        super(Vgg, self).__init__()
        self.feature_channel = 256
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x = self.features(x)
        return x


class ResNet22(nn.Module):
    """
    FAT: fix all at first (for siamrpn)
    """
    def __init__(self):
        super(ResNet22, self).__init__()
        self.features = ResNet(Bottleneck_CI, [3, 4], [True, False], [False, True])
        # self.feature_size = 512
        self.feature_channel = 512

    def forward(self, x):
        x = self.features(x)
        return x


class Incep22(nn.Module):
    def __init__(self):
        super(Incep22, self).__init__()
        self.features = Inception(InceptionM, [3, 4])
        # self.feature_size = 640
        self.feature_channel = 640

    def forward(self, x):
        x = self.features(x)
        return x


class ResNeXt22(nn.Module):
    def __init__(self):
        super(ResNeXt22, self).__init__()
        self.features = ResNeXt(num_blocks=[3, 4], cardinality=32, bottleneck_width=4)
        # self.feature_size = 512
        self.feature_channel = 512

    def forward(self, x):
        x = self.features(x)
        return x


class ResNet22W(nn.Module):
    """
    ResNet22W: double 3*3 layer (only) channels in residual blob
    """
    def __init__(self):
        super(ResNet22W, self).__init__()
        self.features = ResNet(Bottleneck_BIG_CI, [3, 4], [True, False], [False, True], firstchannels=64, channels=[64, 128])
        # self.feature_size = 512
        self.feature_channel = 512

    def forward(self, x):
        x = self.features(x)

        return x


# for SiamRPN++
def resnet18(**kwargs):
    """Constructs a ResNet-18 model.

    """
    model = ResNetPP(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


# for SiamRPN++
def resnet34(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = ResNetPP(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


# for SiamRPN++
def resnet50(**kwargs):
    """Constructs a ResNet-50 model.

    """
    model = ResNetPP(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model









