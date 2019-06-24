import sys
import os
import time
import json
import random
import math
import numpy as np
import argparse
import cv2
import h5py

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
from models.loss import *
from image import load_data, generate_anchor, load_data_rpn
from models.builder import *
import dataset
from mmcv import Config
from utils import save_checkpoint, is_valid_number, bbox_iou
from models.utils import load_pretrain
from models.lr_scheduler import *


parser = argparse.ArgumentParser(description='PyTorch SiameseX')

parser.add_argument('--config', metavar='model', default='configs/SiamRPN.py', type=str,
                    help='which model to use.')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str,
                    help='path to the pretrained model')

parser.add_argument('--gpu', metavar='GPU', default='0', type=str,
                    help='GPU id to use.')


def main():
    
    global args, best_prec1, weight, segmodel
    
    best_prec1 = 0
    prec1 = 0
    
    coco = 0
    
    temp_args = parser.parse_args()

    args = Config.fromfile(temp_args.config)
    args.pre = temp_args.pre
    args.gpu = temp_args.gpu

    with open('./data/ilsvrc_vid_new.txt', 'r') as outfile:
        args.ilsvrc = json.load(outfile)
    with open('./data/vot2018_new.txt', 'r') as outfile:
        args.vot2018 = json.load(outfile)
    if os.path.isfile('youtube_final_new.txt'):
        with open('youtube_final_new.txt', 'r') as outfile:
            args.youtube = json.load(outfile)
    else:
        args.youtube = None
    # with open('vot2018.txt', 'r') as outfile:
    #     args.vot2018 = json.load(outfile)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)

    if args.model == 'SiamVGG':
        model = SiamVGG()
    elif args.model == 'SiamFC':
        model = SiamFC()
    elif args.model == 'SiamFCRes22':
        model = SiamFCRes22()
        load_pretrain(model, 'models/pretrained/CIResNet22_PRETRAIN.model')
    elif args.model == 'SiamFCIncep22':
        model = SiamFCIncep22()
        load_pretrain(model, 'models/pretrained/CIRIncep22_PRETRAIN.model')
    elif args.model == 'SiamFCNext22':
        model = SiamFCNext22()
        load_pretrain(model, 'models/pretrained/CIRNeXt22_PRETRAIN.model')
    elif args.model == 'SiamRPN':
        model = SiamRPN()
    elif args.model == 'SiamRPNVGG':
        model = SiamRPNVGG()
    elif args.model == 'SiamRPNRes22':
        model = SiamRPNRes22()
        load_pretrain(model, 'models/pretrained/CIResNet22_PRETRAIN.model')
    elif args.model == 'SiamRPNIncep22':
        model = SiamRPNIncep22()
        load_pretrain(model, 'models/pretrained/CIRIncep22_PRETRAIN.model')
    elif args.model == 'SiamRPNResNeXt22':
        model = SiamRPNResNeXt22()
        load_pretrain(model, 'models/pretrained/CIRNeXt22_PRETRAIN.model')
    elif args.model == 'SiamRPNPP':
        model = SiamRPNPP()
        load_pretrain(model, 'models/pretrained/ResNet50_PRETRAIN.model')

    model = model.cuda()
    model = model.eval()

    criterion = nn.SoftMarginLoss(size_average=False).cuda()  # for SiamFC and SiamVGG
    if 'SiamRPNPP' in args.model:
        optimizer, lr_scheduler = build_opt_lr(model, args.start_epoch)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.decay)

    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            
            args.start_epoch = 0
            best_prec1 = checkpoint['best_prec1']
            best_prec1 = 0
            
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    # prec1 = 0

    if not os.path.isdir('./cp/temp'):
        os.makedirs('./cp/temp')
    
    for epoch in range(args.start_epoch, args.epochs):
        if 'SiamRPNPP' in args.model:
            if args.backbone_train_epoch == epoch:
                print('start training backbone.')
                optimizer, lr_scheduler = build_opt_lr(model, epoch)

            lr_scheduler.step(epoch)
            cur_lr = lr_scheduler.get_cur_lr()
        else:
            cur_lr = adjust_learning_rate(optimizer, epoch)

        print('current learning rate : {}'.format(cur_lr))

        if args.model in ['SiamFC', 'SiamVGG', 'SiamFCRes22', 'SiamFCIncep22', 'SiamFCNext22']:
            train(model, criterion, optimizer, epoch, coco)
        elif args.model in ['SiamRPN', 'SiamRPNVGG', 'SiamRPNRes22', 'SiamRPNIncep22', 'SiamRPNResNeXt22']:
            trainRPN(model, optimizer, epoch, coco)
        elif args.model in ['SiamRPNPP']:
            trainRPNPP(model, optimizer, epoch, coco)

        # is_best = False
        
        is_best = prec1 > best_prec1
        
        best_prec1 = max(prec1, best_prec1)

        if epoch % 100 == 0:
            torch.save(model.state_dict(), './cp/temp/{}_{}.pth'.format(args.model, epoch))
        
        # print(' * best MAE {mae:.3f} '
        #       .format(mae=best_prec1))
        # print(' * MAE {mae:.3f} '
        #       .format(mae=prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.model)

            
def train(model, criterion, optimizer, epoch, coco):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(args.ilsvrc, args.youtube, args.data_type,
                            shuffle=True,
                            transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                             std=[0.229, 0.224, 0.225])]),
                            train=True,
                       
                            batch_size=args.batch_size,
                            num_workers=args.workers, coco=coco),
        batch_size=args.batch_size)

    model.train()
    end = time.time()
    
    for i, (z, x, template, gt_box)in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        z = z.cuda()
        z = Variable(z)
        x = x.cuda()
        x = Variable(x)
        template = template.type(torch.FloatTensor).cuda()
        template = Variable(template)
        
        oup = model(z, x)

        if isinstance(model, SiamFC) or isinstance(model, SiamVGG):
            loss = criterion(oup, template)
        elif isinstance(model, SiamFCRes22):
            loss = model.train_loss(oup, template)

        losses.update(loss.item(), x.size(0))

        optimizer.zero_grad()
        loss.backward()

        if isinstance(model, SiamFCRes22):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)  # gradient clip

        if is_valid_number(loss.item()):
            optimizer.step()

        # optimizer.step()
        
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))


def trainRPN(model, optimizer, epoch, coco):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(args.ilsvrc, args.youtube, args.data_type,
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]),
                            train=True,

                            batch_size=args.batch_size,
                            num_workers=args.workers, coco=coco),
        batch_size=args.batch_size)

    model.train()
    end = time.time()

    for i, (z, x, regression_target, conf_target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        z = z.cuda()
        z = Variable(z)
        x = x.cuda()
        x = Variable(x)

        pred_score, pred_regression = model(z, x)

        pred_conf = pred_score.reshape(-1, 2, 5 * 17 * 17).permute(0, 2, 1)

        pred_offset = pred_regression.reshape(-1, 4, 5 * 17 * 17).permute(0, 2, 1)

        regression_target = regression_target.type(torch.FloatTensor).cuda()
        conf_target = conf_target.type(torch.LongTensor).cuda()

        cls_loss = rpn_cross_entropy(pred_conf, conf_target)
        reg_loss = rpn_smoothL1(pred_offset, regression_target, conf_target)

        loss = cls_loss + reg_loss

        losses.update(loss.item(), x.size(0))
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))


def trainRPNPP(model, optimizer, epoch, coco):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(args.ilsvrc, args.youtube, args.data_type,
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]),
                            train=True,

                            batch_size=args.batch_size,
                            num_workers=args.workers, coco=coco, rpnpp=True),
        batch_size=args.batch_size)

    model.train()
    end = time.time()

    for i, (z, x, regression_target, conf_target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        z = z.cuda()
        z = Variable(z)
        x = x.cuda()
        x = Variable(x)

        pred_score, pred_regression = model(z, x)
        # print(pred_score.size(), pred_regression.size())
        b, a2, h, w = pred_score.size()

        pred_conf = pred_score.reshape(-1, 2, 5 * h * w).permute(0, 2, 1)

        pred_offset = pred_regression.reshape(-1, 4, 5 * h * w).permute(0, 2, 1)

        regression_target = regression_target.type(torch.FloatTensor).cuda()
        conf_target = conf_target.type(torch.LongTensor).cuda()

        # print(pred_conf.size(), conf_target.size())

        cls_loss = rpn_cross_entropy(pred_conf, conf_target)
        reg_loss = rpn_smoothL1(pred_offset, regression_target, conf_target)

        loss = cls_loss + reg_loss

        losses.update(loss.item(), x.size(0))
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    args.lr = args.original_lr
    
    for i in range(len(args.steps)):
        
        scale = args.scales[i] if i < len(args.scales) else 1

        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
    return args.lr


LRs = {
    'log': LogScheduler,
    'step': StepScheduler,
    'multi-step': MultiStepScheduler,
    'linear': LinearStepScheduler,
    'cos': CosStepScheduler}


def _build_lr_scheduler(optimizer, lr_type, epochs=50, last_epoch=-1):
    return LRs[lr_type](optimizer, last_epoch=last_epoch,
                            epochs=epochs, new_allowed=True)


def _build_warm_up_scheduler(optimizer, epochs=50, last_epoch=-1):
    warmup_epoch = args.lr_warm_epoch
    sc1 = _build_lr_scheduler(optimizer, 'step', warmup_epoch, last_epoch)
    sc2 = _build_lr_scheduler(optimizer, 'log', epochs - warmup_epoch, last_epoch)
    return WarmUPScheduler(optimizer, sc1, sc2, epochs, last_epoch)


def build_lr_scheduler(optimizer, epochs=50, last_epoch=-1):
    if args.warmup:
        return _build_warm_up_scheduler(optimizer, epochs, last_epoch)
    else:
        return _build_lr_scheduler(optimizer, args.original_lr, epochs, last_epoch)


def build_opt_lr(model, current_epoch=0):
    if current_epoch >= 20:
        for layer in ['layer2', 'layer3', 'layer4']:
            for param in getattr(model.features, layer).parameters():
                param.requires_grad = True
            for m in getattr(model.features, layer).modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
    else:
        for param in model.features.parameters():
            param.requires_grad = False
        for m in model.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           model.features.parameters()),
                          'lr': 0.1 * args.original_lr}]

    trainable_params += [{'params': model.neck.parameters(),
                        'lr': args.original_lr}]

    trainable_params += [{'params': model.head.parameters(),
                          'lr': args.original_lr}]

    optimizer = torch.optim.SGD(trainable_params,
                                momentum=args.momentum,
                                weight_decay=args.decay)

    lr_scheduler = build_lr_scheduler(optimizer, epochs=args.epochs)
    lr_scheduler.step(args.start_epoch)
    return optimizer, lr_scheduler


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    


if __name__ == '__main__':
    main()        