import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image import *
import torchvision.transforms.functional as F


class listDataset(Dataset):
    def __init__(self, ilsvrc, youtube, data_type='RPN', shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4, coco=0):
        
        self.coco = coco
        self.data_type = data_type
        self.nSamples = None
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

        ilsvrc_sample = []
        youtube_sample = []

        if self.train:
            if ilsvrc is not None:
                ilsvrc_sample = [ilsvrc[i] for i in sorted(random.sample(xrange(len(ilsvrc)), 1000))]
            if youtube is not None:
                youtube_sample = [youtube[i] for i in sorted(random.sample(xrange(len(youtube)), 5000))]
            all_sample = ilsvrc_sample + youtube_sample
            
        self.nSamples = len(all_sample)
        self.lines = []

        for i in xrange(self.nSamples):
            sequence = all_sample[i]
            ran_id = random.randint(0, len(sequence)-1)
                    
            while len(sequence[ran_id]) < 2:
                        
                        sequence = all_sample[random.randint(0, self.nSamples-1)]
                        
                        ran_id = random.randint(0, len(sequence)-1)
                        
            track_obj = sequence[ran_id]
                    
            ran_f1 = random.randint(0, len(track_obj)-1)
                    
            ran_f2 = random.randint(0, len(track_obj)-1)
            self.lines.append([track_obj[ran_f1], track_obj[ran_f2]])
        random.shuffle(self.lines)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        
        pair_infos = self.lines[index]

        if self.data_type == 'NORPN':
        
            z, x, template, gt_box = load_data(pair_infos, self.coco)

            if self.transform is not None:
                z = self.transform(z)
                x = self.transform(x)

            template = torch.from_numpy(template)
            gt_box = torch.from_numpy(gt_box)

            return z, x, template, gt_box

        elif self.data_type == 'RPN':

            z, x, gt_box, regression_target, label = load_data_rpn(pair_infos, self.coco)

            if self.transform is not None:
                z = self.transform(z)
                x = self.transform(x)

            regression_target = torch.from_numpy(regression_target)
            label = torch.from_numpy(label)

            return z, x, regression_target, label

