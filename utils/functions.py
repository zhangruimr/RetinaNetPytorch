import torch as t
from torch.nn import functional as F
import math
import numpy as np
import cv2

def transform( image, size):
    image = t.from_numpy(image / 255).float().permute((2, 0, 1)).contiguous()
    image = resize(image, size)
    image = pad(image,  size)
    return image


def resize(image, size):
    c, h, w = image.shape
    min_stride = min(size / h, size / w)
    image = F.interpolate(image.unsqueeze(0), (math.ceil(h * min_stride), math.ceil(w * min_stride))).squeeze(0)
    return image


def pad(self, image, size):
    _, _h, _w = image.shape
    if _h == size:
        dif = (size - _w) // 2
        pad = (dif, size - _w - dif, 0, 0)

    elif _w == size:
        dif = (size - _h) // 2
        pad = (0, 0, dif, size - _h - dif)
    try:
        image = F.pad(image.unsqueeze(0), pad=pad).squeeze(0)
    except:
        image = None
    return image

def decodeBox(anchors, cls_output):
    a_x1, a_y1, a_x2, a_y2 = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
    a_x, a_y, a_w, a_h = (a_x1 + a_x2) * 0.5, (a_y1 + a_y2) * 0.5, a_x2 - a_x1, a_y2 - a_y1

    gt_x1, gt_y1, gt_x2, gt_y2 = cls_output[:, 0], cls_output[:, 1], cls_output[:, 2], cls_output[:, 3]
    gt_x, gt_y, gt_w, gt_h = (gt_x1 + gt_x2) * 0.5, (gt_y1 +  gt_y2) * 0.5, gt_x2 - gt_x1, gt_y2 - gt_y1

    x = gt_x * a_w + a_x
    y = gt_y * a_h + a_y
    w = t.exp(gt_w) * a_w
    h = t.exp(gt_h) * a_h

    x1 = x - 0.5 * w
    y1 = x - 0.5 * h
    x2 = x + 0.5 * w
    y2 = x + 0.5 * h
    return t.stack((x1, y1, x2, y2), 1)
def cal_box(detection, image_shape, size=608):
    _, h, w = image_shape
    min_stride = min(size / h, size / w)
    _h, _w = math.ceil(h*min_stride), math.ceil(w*min_stride)
    if _h == size:
        dif = (size - _w) // 2
        detection[:, 0] = detection[:, 0] - dif
        detection[:, 2] = detection[:, 2] - dif
        detection[:, 0:4] = detection[:, 0:4] / min_stride

    elif _w == size:
        dif = (size - _h) // 2
        detection[:, 1] = detection[:, 1] - dif
        detection[:, 3] = detection[:, 3] - dif
        detection[:, 0:4] = detection[:, 0:4] / min_stride
    return detection


def iou(anchors, label):
    a_x1, a_y1, a_x2, a_y2 = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
    gt_x1, gt_y1, gt_x2, gt_y2 = label[0], label[1], label[2], label[3]

    insection = (t.min(a_x2, gt_x2) - t.max(a_x1, gt_x1)) * (t.min(a_y2, gt_y2) - t.max(a_y1, gt_y1))
    insection = t.clamp(insection, min=1e-8)
    union  = (a_x2 - a_x1) * (a_y2 - a_y1) + (gt_x2 - gt_x1) * (gt_y2 - gt_y1) - insection

    iou = insection / union
    return iou
def nms(cls, reg):

    val, id = t.max(cls, 1)
    filter_mask = val > 0.5
    if not t.sum(filter_mask) > 0:
        return None
    val = val[filter_mask]
    id = id[filter_mask]
    res = t.cat((reg, cls), 1)

    res = res[filter_mask]


    seq = t.argsort(val, descending=True)
    val = val[seq]
    id  = id[seq]
    res = res[seq, :]
    objects = []
    for box in res:
        objects.append(box)
        ious = iou(res[:, 0:4], box[0:4])
        iou_mask = ious > 0.5
        cls_mask = id[0] == id

        mask = (iou_mask + cls_mask) // 2
        mask = 1 - mask

        res = res[mask]
    objects = t.stack(objects, 0)
    return objects



