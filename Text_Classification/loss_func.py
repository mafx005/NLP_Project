# -*- coding: utf-8 -*-
# @Time : 2021/8/24 14:48
# @Author : M
# @FileName: loss_func.py
# @Dec :

import torch.nn.functional as F


def compute_kl_loss(p, q, pad_mask=None):
    """
    R-Drop的Loss
    """
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss
