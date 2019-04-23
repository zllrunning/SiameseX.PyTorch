import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import os
import csv
import numpy as np
from PIL import Image
import time
import torch
import h5py
import collections
from demo_utils.siamese import SiameseNet
from demo_utils.parse_arguments import parse_arguments


Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])


def load_net(fname, net):
    pretrained_dict = torch.load(fname)['state_dict']
    pretrained_dict = {'model.'+key: value for key, value in pretrained_dict.items()}
    net.load_state_dict(pretrained_dict)



def convert_bbox_format(bbox, to='center-based'):
    x, y, target_width, target_height = bbox.x, bbox.y, bbox.width, bbox.height
    if to == 'top-left-based':
        x -= get_center(target_width)
        y -= get_center(target_height)
    elif to == 'center-based':
        y += get_center(target_height)
        x += get_center(target_width)
    else:
        raise ValueError("Bbox format: {} was not recognized".format(to))
    return Rectangle(x*1.0, y*1.0, target_width*1.0, target_height*1.0)


def get_center(x):
    return (x - 1.) / 2.


class SiamVGGTracker(object):

    def __init__(self, tracker_name, imagefile, region):
        # param
        self.hp, self.evaluation, self.run, self.env, self.design = parse_arguments()
        
        self.final_score_sz = 273 
        
        # init network
        self.siam = SiameseNet(tracker_name)
        if tracker_name == 'SiamFC':
            pretrained = None
        elif tracker_name == 'SiamVGG':
            pretrained = None
        elif tracker_name == 'SiamFCRes22':
            pretrained = 'cp/siamrescheckpoint.pth.tar'
        elif tracker_name == 'SiamFCIncep22':
            pretrained = None
        elif tracker_name == 'SiamFCNext22':
            pretrained = './cp/siamresnextcheckpoint.pth.tar'

        load_net(pretrained, self.siam)
        self.siam.cuda()
        
        # init bbox
        bbox = convert_bbox_format(region, 'center-based')
        
        self.pos_x, self.pos_y, self.target_w, self.target_h = bbox.x, bbox.y, bbox.width, bbox.height
        
        # init scale factor, penalty
        self.scale_factors = self.hp.scale_step**np.linspace(-np.ceil(self.hp.scale_num/2), np.ceil(self.hp.scale_num/2), self.hp.scale_num)
        hann_1d = np.expand_dims(np.hanning(self.final_score_sz), axis=0)
        self.penalty = np.transpose(hann_1d) * hann_1d
        self.penalty = self.penalty / np.sum(self.penalty)
        
        context = self.design.context*(self.target_w+self.target_h)
        self.z_sz = np.sqrt(np.prod((self.target_w+context)*(self.target_h+context)))
        self.x_sz = float(self.design.search_sz) / self.design.exemplar_sz * self.z_sz
        
        image_, self.templates_z_ = self.siam.get_template_z_new(self.pos_x, self.pos_y, self.z_sz, imagefile, self.design)

    def track(self, imagefile):
        scaled_search_area = self.x_sz * self.scale_factors
        scaled_target_w = self.target_w * self.scale_factors
        scaled_target_h = self.target_h * self.scale_factors
        image_, scores_ = self.siam.get_scores_new(self.pos_x, self.pos_y, scaled_search_area, self.templates_z_, imagefile, self.design, self.final_score_sz)
        
        scores_ = np.squeeze(scores_)
        scores_[0, :, :] = self.hp.scale_penalty*scores_[0, :, :]
        scores_[2, :, :] = self.hp.scale_penalty*scores_[2, :, :]
        new_scale_id = np.argmax(np.amax(scores_, axis=(1, 2)))
        
        self.x_sz = (1-self.hp.scale_lr)*self.x_sz + self.hp.scale_lr*scaled_search_area[new_scale_id]         
        self.target_w = (1-self.hp.scale_lr)*self.target_w + self.hp.scale_lr*scaled_target_w[new_scale_id]
        self.target_h = (1-self.hp.scale_lr)*self.target_h + self.hp.scale_lr*scaled_target_h[new_scale_id]
        score_ = scores_[new_scale_id, :, :]
        score_ = score_ - np.min(score_)
        score_ = score_/np.sum(score_)
        score_ = (1-self.hp.window_influence)*score_ + self.hp.window_influence*self.penalty
        self.pos_x, self.pos_y = _update_target_position(self.pos_x, self.pos_y, score_, self.final_score_sz, self.design.tot_stride, self.design.search_sz, self.hp.response_up, self.x_sz)
        
        bbox = Rectangle(self.pos_x, self.pos_y, self.target_w, self.target_h)
        bbox = convert_bbox_format(bbox, 'top-left-based')
        
        return bbox, np.max(score_)
    



def _update_target_position(pos_x, pos_y, score, final_score_sz, tot_stride, search_sz, response_up, x_sz):
    # find location of score maximizer
    #score = score[32:241,32:241]
    #final_score_sz = final_score_sz - 64 
    
    p = np.asarray(np.unravel_index(np.argmax(score), np.shape(score)))
    # displacement from the center in search area final representation ...
    center = float(final_score_sz - 1) / 2
    disp_in_area = p - center
    # displacement from the center in instance crop
    disp_in_xcrop = disp_in_area * float(tot_stride) / response_up
    # displacement from the center in instance crop (in frame coordinates)
    disp_in_frame = disp_in_xcrop *  x_sz / search_sz
    # *position* within frame in frame coordinates
    pos_y, pos_x = pos_y + disp_in_frame[0], pos_x + disp_in_frame[1]
    return pos_x, pos_y


