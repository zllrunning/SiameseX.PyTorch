import h5py
import torch
import shutil
import collections
from PIL import ImageStat
from PIL import Image
import numpy as np
import cv2
import math
import os


def is_valid_number(x):
    return not(math.isnan(x) or math.isinf(x) or x > 1e4)

def get_center(x):
    return (x - 1.) / 2


def convert_array_to_rec(array):
    return Rectangle(array[0],array[1],array[2],array[3])


Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])


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


def get_zbox(bbox, p_rate=0.25):
    x, y, target_width, target_height = bbox.x, bbox.y, bbox.width, bbox.height
    p = 2 * p_rate * (target_width+target_height)
    target_sz = np.sqrt(np.prod((target_width+p) * (target_height+p)))
    return Rectangle(x, y, target_sz, target_sz)


def get_xbox(zbox, dx=0, dy=0, padding_rate=1):
    x, y, target_width, target_height = zbox.x+dx*0.5*zbox.width, zbox.y+dy*0.5*zbox.height, zbox.width, zbox.height
    return Rectangle(x, y, target_width*256.0/128*padding_rate, target_height*256.0/128*padding_rate)


def gen_xz(img, inbox, to='x', pdrt=1):
    box = Rectangle(inbox.x, inbox.y, inbox.width*pdrt, inbox.height*pdrt)
    x_sz = (255, 255)
    z_sz = (127, 127)
    bg = Image.new('RGB', (int(box.width), int(box.height)), tuple(map(int, ImageStat.Stat(img).mean)))
    bg.paste(img, (-int(box.x-0.5*box.width), -int(box.y - 0.5*box.height)))
    if to == 'x':
        temp = bg.resize(x_sz)
    elif to == 'z':
        temp = bg.resize(z_sz)
    else:
        raise ValueError("Bbox format: {} was not recognized".format(to))
    return temp


def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():        
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)


def save_checkpoint(state, is_best, task_id, filename='checkpoint.pth.tar'):
    if not os.path.exists('./cp'):
        os.makedirs('./cp')
    torch.save(state, 'cp/'+task_id+filename)
    if is_best:
        shutil.copyfile('cp/'+task_id+filename, 'cp/'+task_id+'model_best.pth.tar')


def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
        Mx = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
        my = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
        My = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea/uarea


def sigmoid(x):
    return 1 / (1 + math.exp(-x))