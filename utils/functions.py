import torch as t

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



