import torch
import torch.nn
from IPython import embed
import torch.nn.functional as F
import random
import numpy as np


def rpn_cross_entropy(input, target):
    r"""
    :param input: (15x15x5,2)
    :param target: (15x15x5,)
    :return:
    """
    mask_ignore = target == -1
    mask_calcu = 1 - mask_ignore
    loss = F.cross_entropy(input=input[mask_calcu], target=target[mask_calcu], size_average=False)
    # loss = torch.div(torch.sum(loss), 64)
    return loss


def rpn_cross_entropy_balance(input, target, num_pos, num_neg):
    r"""
    :param input: (N,1125,2)
    :param target: (15x15x5,)
    :return:
    """
    cal_index_pos = np.array([], dtype=np.int64)
    cal_index_neg = np.array([], dtype=np.int64)
    for batch_id in range(target.shape[0]):
        if len(np.where(target[batch_id].cpu() == 1)[0]) != 0:
            pos_index = np.random.choice(np.where(target[batch_id].cpu() == 1)[0], num_pos)
            cal_index_pos = np.append(cal_index_pos, batch_id * target.shape[1] + pos_index)
        neg_index = np.random.choice(np.where(target[batch_id].cpu() == 0)[0], num_neg)

        cal_index_neg = np.append(cal_index_neg, batch_id * target.shape[1] + neg_index)

    pos_loss = F.cross_entropy(input=input.reshape(-1, 2)[cal_index_pos], target=target.view(-1)[cal_index_pos]) / \
               cal_index_pos.shape[0]
    neg_loss = F.cross_entropy(input=input.reshape(-1, 2)[cal_index_neg], target=target.view(-1)[cal_index_neg]) / \
               cal_index_neg.shape[0]
    loss = (pos_loss + neg_loss) / 2

    # loss = F.cross_entropy(input=input.reshape(-1, 2)[cal_index], target=target.flatten()[cal_index])
    return loss


def rpn_cross_entropy_balance_without_norm(input, target, num_pos=16, num_neg=16):
    cal_index = np.array([], dtype=np.int64)
    cal_index_pos = np.array([], dtype=np.int64)
    cal_index_neg = np.array([], dtype=np.int64)
    for batch_id in range(target.shape[0]):
        if len(np.where(target[batch_id].cpu() == 1)[0]) > 16:
            pos_index = np.random.choice(np.where(target[batch_id].cpu() == 1)[0], num_pos)
            cal_index = np.append(cal_index, batch_id * target.shape[1] + pos_index)

            neg_index = np.random.choice(np.where(target[batch_id].cpu() == 0)[0], num_neg)

            cal_index = np.append(cal_index, batch_id * target.shape[1] + neg_index)

        else:
            pos_index = np.where(target[batch_id].cpu() == 1)[0]
            cal_index = np.append(cal_index, batch_id * target.shape[1] + pos_index)

            neg_index = np.random.choice(np.where(target[batch_id].cpu() == 0)[0], 32 - len(pos_index))

            cal_index = np.append(cal_index, batch_id * target.shape[1] + neg_index)

    loss = F.cross_entropy(input=input.reshape(-1, 2)[cal_index], target=target.view(-1)[cal_index], size_average=False)

    # loss = F.cross_entropy(input=input.reshape(-1, 2)[cal_index], target=target.flatten()[cal_index])
    return loss


# def rpn_cross_entropy_balance(input, target, num_pos, num_neg):
#     r"""
#     :param input: (N,1125,2)
#     :param target: (15x15x5,)
#     :return:
#     """
#     pos_index = np.random.choice(np.where(target.cpu().flatten() == 1)[0], target.shape[0] * num_pos)
#     neg_index = np.random.choice(np.where(target.cpu().flatten() == 0)[0], target.shape[0] * num_neg)
#     cal_index = np.append(neg_index, pos_index)
#
#     loss = F.cross_entropy(input=input.reshape(-1, 2)[cal_index], target=target.flatten()[cal_index])
#     return loss


def rpn_smoothL1(input, target, label):
    r'''
    :param input: torch.Size([1, 1125, 4])
    :param target: torch.Size([1, 1125, 4])
            label: (torch.Size([1, 1125]) pos neg or ignore
    :return:
    '''
    pos_index = np.where(label.cpu() == 1)
    loss = F.smooth_l1_loss(input[pos_index], target[pos_index], size_average=False)
    # loss = torch.div(torch.sum(loss), 64)

    return loss