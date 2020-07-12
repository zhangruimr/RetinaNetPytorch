import torch as t
from torch.nn import functional as F
import math
import numpy as np
import cv2
from datasets import *
def clip_box(reg_output, size):
    bottom = t.zeros(reg_output.shape)
    top = t.ones(reg_output.shape) * size
    if t.cuda.is_available():
        bottom = bottom.cuda(2)
        top = top.cuda(2)
    reg_output = t.where(reg_output < 0, bottom, reg_output)
    reg_output = t.where(reg_output > size, top, reg_output)
    return  reg_output
def decodeBox(anchors, cls_output):
    a_x1, a_y1, a_x2, a_y2 = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
    a_x, a_y, a_w, a_h = (a_x1 + a_x2) * 0.5, (a_y1 + a_y2) * 0.5, a_x2 - a_x1, a_y2 - a_y1

    #gt_x1, gt_y1, gt_x2, gt_y2 = cls_output[:, 0], cls_output[:, 1], cls_output[:, 2], cls_output[:, 3]
    gt_x, gt_y, gt_w, gt_h = cls_output[:, 0], cls_output[:, 1], cls_output[:, 2], cls_output[:, 3]
    #gt_x, gt_y, gt_w, gt_h = (gt_x1 + gt_x2) * 0.5, (gt_y1 +  gt_y2) * 0.5, gt_x2 - gt_x1, gt_y2 - gt_y1

    x = gt_x * a_w + a_x
    y = gt_y * a_h + a_y
    w = t.exp(gt_w) * a_w
    h = t.exp(gt_h) * a_h

    x1 = x - 0.5 * w
    y1 = y - 0.5 * h
    x2 = x + 0.5 * w
    y2 = y + 0.5 * h
    return t.stack((x1, y1, x2, y2), 1)
def cal_box(detection, image_shape, size=608):
    h, w, c = image_shape
    min_stride = min(size / h, size / w)
    _h, _w = math.ceil(h * min_stride), math.ceil(w * min_stride)
    #print(h, w)
    #print(_h, _w)
    if _h == size:
        dif = (size - _w) // 2
        #print("dif",dif)
        detection[:, 0] = detection[:, 0] - dif
        detection[:, 2] = detection[:, 2] - dif
        detection[:, 0:4] = detection[:, 0:4] / min_stride

    elif _w == size:
        dif = (size - _h) // 2
        #print("dif", dif)
        detection[:, 1] = detection[:, 1] - dif
        detection[:, 3] = detection[:, 3] - dif
        detection[:, 0:4] = detection[:, 0:4] / min_stride
    return detection


def iou(anchors, label):
    a_x1, a_y1, a_x2, a_y2 = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
    gt_x1, gt_y1, gt_x2, gt_y2 = label[0], label[1], label[2], label[3]

    insection = t.clamp(t.min(a_x2, gt_x2) - t.max(a_x1, gt_x1), min=0) * t.clamp(t.min(a_y2, gt_y2) - t.max(a_y1, gt_y1), min=0)

    union  = (a_x2 - a_x1) * (a_y2 - a_y1) + (gt_x2 - gt_x1) * (gt_y2 - gt_y1) - insection

    iou = insection / union

    return iou
def nms(cls, reg):

    val, id = t.max(cls, 1)
    filter_mask = val > 0.45
    if not t.sum(filter_mask) > 0:
        return None
    res = t.cat((reg, val.reshape((-1, 1)), id.reshape((-1, 1)).float()), 1)
    id = id[filter_mask]
    val = val[filter_mask]
    res = res[filter_mask]

    seq = t.argsort(val, descending=True)
    res = res[seq, :].float()

    objects = []
    while len(res) > 0:
        box = res[0]
        objects.append(box)
        ious = iou(res[:, 0:4], box[0:4])
        #print("ious:", ious)
        iou_mask = ious > 0.5
        cls_mask = res[:, -1] == box[-1]

        mask = iou_mask & cls_mask
        mask = ~mask

        res = res[mask]

    objects = t.stack(objects, 0)
    #print(objects)
    return objects



