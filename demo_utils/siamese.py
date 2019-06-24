import numpy as np
import scipy.io
import sys
import six
import os.path
from PIL import Image, ImageStat
import collections
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

from demo_utils.crops import extract_crops_z, extract_crops_x, pad_frame, gen_xz
import models.builder as builder
from models.builder import *
sys.path.append('../')

Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])


def Image_to_Tensor(img, mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]):
    zn = np.asarray(img, 'float')
    zr = zn.transpose(2, 0, 1)
    for c in range(0, 3):
        zr[c] = ((zr[c]/255)-mean[c])/std[c]
    zt = torch.from_numpy(zr).float()
    return zt


class SiameseNet(nn.Module):

    def __init__(self, tracker_name):
        super(SiameseNet, self).__init__()
        self.tracker_name = tracker_name
        self.model = getattr(builder, tracker_name)()

    def forward(self, z, x):
        z = self.model.features(z)
        x = self.model.features(x)
        out = self.mdoel.head(z, x)
        return out

    def branch(self, allin):
        allout = self.model.features(allin)
        return allout

    def get_template_z(self, pos_x, pos_y, z_sz, image,
                       design):
        if isinstance(image, six.string_types):
            image = Image.open(image).convert('RGB')
        avg_chan = ImageStat.Stat(image).mean
        frame_padded_z, npad_z = pad_frame(image, image.size, pos_x, pos_y, z_sz, avg_chan)
        z_crops = extract_crops_z(frame_padded_z, npad_z, pos_x, pos_y, z_sz, design.exemplar_sz)
        template_z = self.branch(Variable(z_crops).cuda())
        return image, template_z

    def get_template_z_new(self, pos_x, pos_y, z_sz, image,
                           design):
        if isinstance(image, six.string_types):
            image = Image.open(image).convert('RGB')
        # avg_chan = ImageStat.Stat(image).mean
        # frame_padded_z, npad_z = pad_frame(image, image.size, pos_x, pos_y, z_sz, avg_chan)
        # z_crops = extract_crops_z(frame_padded_z, npad_z, pos_x, pos_y, z_sz, design.exemplar_sz)
        z = gen_xz(image, Rectangle(pos_x, pos_y, z_sz, z_sz), to='z')
        # cv2.imshow('z', np.array(z))
        tz = Image_to_Tensor(z).unsqueeze(0)
        template_z = self.branch(Variable(tz).cuda())

        # print template_z same
        return image, template_z

    def get_scores(self, pos_x, pos_y, scaled_search_area, template_z, filename,
                   design, final_score_sz):
        image = Image.open(filename).convert('RGB')
        avg_chan = ImageStat.Stat(image).mean
        frame_padded_x, npad_x = pad_frame(image, image.size, pos_x, pos_y, scaled_search_area[2], avg_chan)
        x_crops = extract_crops_x(frame_padded_x, npad_x, pos_x, pos_y, scaled_search_area[0], scaled_search_area[1],
                                  scaled_search_area[2], design.search_sz)

        template_x = self.branch(Variable(x_crops).cuda())

        template_z = template_z.repeat(template_x.size(0), 1, 1, 1)

        scores = self.model.head(template_z, template_x)

        # scores = self.bn_adjust(scores)
        # TODO: any elegant alternator?
        scores = scores.squeeze().permute(1, 2, 0).data.cpu().numpy()
        scores_up = cv2.resize(scores, (final_score_sz, final_score_sz), interpolation=cv2.INTER_CUBIC)
        scores_up = scores_up.transpose((2, 0, 1))
        return image, scores_up

    def get_scores_new(self, pos_x, pos_y, scaled_search_area, template_z, filename,
                       design, final_score_sz):
        image = Image.open(filename).convert('RGB')
        # avg_chan = ImageStat.Stat(image).mean
        # frame_padded_x, npad_x = pad_frame(image, image.size, pos_x, pos_y, scaled_search_area[2], avg_chan)
        # x_crops = extract_crops_x(frame_padded_x, npad_x, pos_x, pos_y, scaled_search_area[0], scaled_search_area[1], scaled_search_area[2], design.search_sz)
        txs = []
        for scale in scaled_search_area:
            x = gen_xz(image, Rectangle(pos_x, pos_y, scale, scale), to='x')

            tx = Image_to_Tensor(x).unsqueeze(0)
            txs.append(tx.squeeze(0))
        x_crops = torch.stack(txs)
        # print Rectangle(pos_x, pos_y,scale,scale)
        template_x = self.branch(Variable(x_crops).cuda())

        template_z = template_z.repeat(template_x.size(0), 1, 1, 1)

        scores = self.model.head(template_z, template_x)

        # scores = self.bn_adjust(scores)
        # TODO: any elegant alternator?
        scores = scores.squeeze().permute(1, 2, 0).data.cpu().numpy()
        scores_up = cv2.resize(scores, (final_score_sz, final_score_sz), interpolation=cv2.INTER_CUBIC)
        scores_up = scores_up.transpose((2, 0, 1))
        return image, scores_up


def load_siamfc_from_matconvnet(net_path, model):
    params_names_list, params_values_list = load_matconvnet(net_path)

    params_values_list = [torch.from_numpy(p) for p in params_values_list]
    for l, p in enumerate(params_values_list):
        param_name = params_names_list[l]
        if 'conv' in param_name and param_name[-1] == 'f':
            p = p.permute(3, 2, 0, 1)
        p = torch.squeeze(p)
        params_values_list[l] = p

    net = nn.Sequential(
        model.conv1,
        model.conv2,
        model.conv3,
        model.conv4,
        model.conv5
    )

    for l, layer in enumerate(net):
        layer[0].weight.data[:] = params_values_list[params_names_list.index('br_conv%df' % (l + 1))]
        layer[0].bias.data[:] = params_values_list[params_names_list.index('br_conv%db' % (l + 1))]

        if l < len(net) - 1:
            layer[1].weight.data[:] = params_values_list[params_names_list.index('br_bn%dm' % (l + 1))]
            layer[1].bias.data[:] = params_values_list[params_names_list.index('br_bn%db' % (l + 1))]

            bn_moments = params_values_list[params_names_list.index('br_bn%dx' % (l + 1))]
            layer[1].running_mean[:] = bn_moments[:,0]
            layer[1].running_var[:] = bn_moments[:,1] ** 2
        else:
            model.bn_adjust.weight.data[:] = params_values_list[params_names_list.index('fin_adjust_bnm')]
            model.bn_adjust.bias.data[:] = params_values_list[params_names_list.index('fin_adjust_bnb')]

            bn_moments = params_values_list[params_names_list.index('fin_adjust_bnx')]
            model.bn_adjust.running_mean[:] = bn_moments[0]
            model.bn_adjust.running_var[:] = bn_moments[1] ** 2

    return model

def load_matconvnet(net_path):
    mat = scipy.io.loadmat(net_path)
    net_dot_mat = mat.get('net')
    params = net_dot_mat['params']
    params = params[0][0]
    params_names = params['name'][0]
    params_names_list = [params_names[p][0] for p in range(params_names.size)]
    params_values = params['value'][0]
    params_values_list = [params_values[p] for p in range(params_values.size)]

    return params_names_list, params_values_list