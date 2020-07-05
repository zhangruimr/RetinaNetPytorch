import torch as t

def iou(anchors, label):
    a_x1, a_y1, a_x2, a_y2 = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
    gt_x1, gt_y1, gt_x2, gt_y2 = label[0], label[1], label[2], label[3]

    insection = (t.min(a_x2, gt_x2) - t.max(a_x1, gt_x1)) * (t.min(a_y2, gt_y2) - t.max(a_y1, gt_y1))
    insection = t.clamp(insection, min=1e-8)
    union  = (a_x2 - a_x1) * (a_y2 - a_y1) + (gt_x2 - gt_x1) * (gt_y2 - gt_y1) - insection

    iou = insection / union
    return iou
def nms():
    pass

