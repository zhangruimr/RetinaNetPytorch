import torch as t
import os
import numpy as np
from pycocotools.coco import COCO
x = t.Tensor([0, 1,2,3]).byte()
print(x // 2)
"""
def getLabels(COCO, imageId):
    print(imageId)
    annId = COCO.getAnnIds(imageId, iscrowd=False)
    print(annId)
    labels_id = COCO.loadAnns(annId)
    labels = []
    if len(labels_id) <= 0:

        labels = np.zeros((0, 5))
        return labels
    for label_id in labels_id:
        if label_id['bbox'][2] < 1 or label_id['bbox'][3] < 1:
            continue
        label = np.zeros((5))
        label[0] = label_id['category_id'] - 1
        label[1:] = label_id['bbox']
        labels.append(label)
    labels = np.stack(labels, 0)
    #print(labels)
    return labels
road = os.path.join("/home/zr/COCO" , "annotations", "instances_train2017.json")
os.makedirs("../labels", exist_ok=True)
coco = COCO(road)
imageId = coco.getImgIds()
#print(imageId[0])
for i in range(len(imageId)):
    #print(i)
    filename = coco.loadImgs(imageId[i])[0]['file_name']
    wr = open(os.path.join("../labels", filename.split(".")[0])+".txt", 'w')
    label = getLabels(coco, imageId[i])
    np.savetxt(os.path.join("../labels", filename.split(".")[0])+".txt", label)
    wr.close()
"""
