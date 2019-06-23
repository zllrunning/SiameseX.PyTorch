import math
import torch
from torchvision import models
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn as nn
from torch.autograd import Variable
from .heads import Corr_Up, MultiRPN, DepthwiseRPN
from .backbones import AlexNet, Vgg, ResNet22, Incep22, ResNeXt22, ResNet22W, resnet50, resnet34, resnet18
from neck import AdjustLayer, AdjustAllLayer
from .utils import load_pretrain


__all__ = ['SiamFC_', 'SiamFC', 'SiamVGG', 'SiamFCRes22', 'SiamFCIncep22', 'SiamFCNext22', 'SiamFCRes22W',
           'SiamRPN', 'SiamRPNVGG', 'SiamRPNRes22', 'SiamRPNIncep22', 'SiamRPNResNeXt22', 'SiamRPNPP']


class SiamFC_(nn.Module):
    def __init__(self):
        super(SiamFC_, self).__init__()
        self.features = None
        # self.head = None

    def head(self, z, x):
        n, c, h, w = x.size()
        x = x.view(1, n * c, h, w)
        out = F.conv2d(x, z, groups=n)
        out = out.view(n, 1, out.size(-2), out.size(-1))
        return out

    def feature_extractor(self, x):
        return self.features(x)

    def connector(self, template_feature, search_feature):
        pred_score = self.head(template_feature, search_feature)
        return pred_score

    def branch(self, allin):
        allout = self.feature_extractor(allin)
        return allout

    def forward(self, template, search):
        zf = self.feature_extractor(template)
        xf = self.feature_extractor(search)
        score = self.connector(zf, xf)
        return score


class SiamFC(SiamFC_):

    def __init__(self):
        super(SiamFC, self).__init__()
        self.features = AlexNet()
        self._initialize_weights()

    def forward(self, z, x):
        zf = self.features(z)
        xf = self.features(x)
        score = self.head(zf, xf)
        return score

    def head(self, z, x):
        # fast cross correlation
        n, c, h, w = x.size()
        x = x.view(1, n * c, h, w)
        out = F.conv2d(x, z, groups=n)
        out = out.view(n, 1, out.size(-2), out.size(-1))

        # adjust the scale of responses
        out = 0.001 * out + 0.0

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, mode='fan_out',
                                     nonlinearity='relu')
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SiamVGG(nn.Module):

    def __init__(self):
        super(SiamVGG, self).__init__()
        self.features = Vgg()
        self.bn_adjust = nn.BatchNorm2d(1)
        self._initialize_weights()

        # init weight with pretrained model
        mod = models.vgg16(pretrained=True)
        for i in xrange(len(self.features.state_dict().items()) - 2):
            self.features.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]

    def forward(self, z, x):
        zf = self.features(z)
        xf = self.features(x)
        score = self.head(zf, xf)

        return score

    def head(self, z, x):
        out = []
        for i in range(x.size(0)):
            out.append(F.conv2d(x[i, :, :, :].unsqueeze(0), z[i, :, :, :].unsqueeze(0)))

        return self.bn_adjust(torch.cat(out, dim=0))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SiamFCRes22(SiamFC_):
    def __init__(self, **kwargs):
        super(SiamFCRes22, self).__init__(**kwargs)
        self.features = ResNet22()
        self.head = Corr_Up()
        self.criterion = nn.BCEWithLogitsLoss()

    def _cls_loss(self, pred, label, select):
        if len(select.size()) == 0: return 0
        pred = torch.index_select(pred, 0, select)
        label = torch.index_select(label, 0, select)
        return self.criterion(pred, label)  # the same as tf version

    def _weighted_BCE(self, pred, label):
        label[label == -1] = 0  # be careful, because when loading data, the label is -1 or 1.
                                # it is suitable for SoftMarginLoss, but not suitable for BCEWithLogitsLoss.
        pred = pred.view(-1)
        label = label.view(-1)
        pos = Variable(label.data.eq(1).nonzero().squeeze()).cuda()
        neg = Variable(label.data.eq(0).nonzero().squeeze()).cuda()

        loss_pos = self._cls_loss(pred, label, pos)
        loss_neg = self._cls_loss(pred, label, neg)
        return loss_pos * 0.5 + loss_neg * 0.5

    def train_loss(self, pred, label):
        return torch.mean(self._weighted_BCE(pred, label))


class SiamFCIncep22(SiamFCRes22):
    def __init__(self, **kwargs):
        super(SiamFCIncep22, self).__init__(**kwargs)
        self.features = Incep22()
        # self.head = Corr_Up()


class SiamFCNext22(SiamFCRes22):
    def __init__(self, **kwargs):
        super(SiamFCNext22, self).__init__(**kwargs)
        self.features = ResNeXt22()
        # self.head = Corr_Up()


class SiamFCRes22W(SiamFCRes22):
    def __init__(self, **kwargs):
        super(SiamFCRes22W, self).__init__(**kwargs)
        self.features = ResNet22W()
        # self.head = Corr_Up()


class SiamRPN(nn.Module):
    def __init__(self):
        super(SiamRPN, self).__init__()
        self.width = int(256)
        self.height = int(256)
        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0
        self.features = AlexNet()

        self.regress_adjust = nn.Conv2d(4 * 5, 4 * 5, 1)
        self.mid()
        self._initialize_weights()

    def mid(self):
        self.conv_cls1 = nn.Conv2d(self.features.feature_channel, self.features.feature_channel * 2 * 5, kernel_size=3,
                                   stride=1, padding=0)
        self.conv_r1 = nn.Conv2d(self.features.feature_channel, self.features.feature_channel * 4 * 5, kernel_size=3,
                                 stride=1, padding=0)
        self.conv_cls2 = nn.Conv2d(self.features.feature_channel, self.features.feature_channel, kernel_size=3,
                                   stride=1, padding=0)
        self.conv_r2 = nn.Conv2d(self.features.feature_channel, self.features.feature_channel, kernel_size=3, stride=1,
                                 padding=0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def xcorr(self, z, x, channels):
        out = []
        kernel_size = z.data.size()[-1]
        for i in range(x.size(0)):
            out.append(F.conv2d(x[i, :, :, :].unsqueeze(0),
                                z[i, :, :, :].unsqueeze(0).view(channels, self.features.feature_channel, kernel_size, kernel_size)))

        return torch.cat(out, dim=0)

    def forward(self, template, detection):
        template_feature = self.features(template)
        detection_feature = self.features(detection)

        kernel_score = self.conv_cls1(template_feature)
        kernel_regression = self.conv_r1(template_feature)
        conv_score = self.conv_cls2(detection_feature)
        conv_regression = self.conv_r2(detection_feature)

        pred_score = self.xcorr(kernel_score, conv_score, 10)
        pred_regression = self.regress_adjust(self.xcorr(kernel_regression, conv_regression, 20))

        return pred_score, pred_regression


class SiamRPNVGG(SiamRPN):
    def __init__(self):
        super(SiamRPNVGG, self).__init__()
        self.features = Vgg()
        self.mid()
        self._initialize_weights()

        # init weight with pretrained model
        mod = models.vgg16(pretrained=True)
        for i in xrange(len(self.features.state_dict().items()) - 2):
            self.features.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]


class SiamRPNRes22(SiamRPN):
    def __init__(self):
        super(SiamRPNRes22, self).__init__()
        self.features = ResNet22()
        self.mid()
        self._initialize_weights()


class SiamRPNIncep22(SiamRPN):
    def __init__(self):
        super(SiamRPNIncep22, self).__init__()
        self.features = Incep22()
        self.mid()
        self._initialize_weights()


class SiamRPNResNeXt22(SiamRPN):
    def __init__(self):
        super(SiamRPNResNeXt22, self).__init__()
        self.features = ResNeXt22()
        self.mid()
        self._initialize_weights()


class SiamRPNPP(nn.Module):
    def __init__(self):
        super(SiamRPNPP, self).__init__()
        # self.width = int(256)
        # self.height = int(256)
        # self.header = torch.IntTensor([0, 0, 0, 0])
        # self.seen = 0
        self.features = resnet50(**{'used_layers': [2, 3, 4]})

        self.neck = AdjustAllLayer(**{'in_channels': [512, 1024, 2048], 'out_channels': [256, 256, 256]})

        self.head = MultiRPN(**{'anchor_num': 5, 'in_channels': [256, 256, 256], 'weighted': True})

    def template(self, z):
        zf = self.features(z)
        zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.features(x)
        xf = self.neck(xf)
        cls, loc = self.head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc,
               }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, template, detection):
        zf = self.features(template)
        xf = self.features(detection)

        zf = self.neck(zf)
        xf = self.neck(xf)

        cls, loc = self.head(zf, xf)

        return cls, loc




















