from __future__ import division
import os
import random
import math
import torch as t
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from torch.nn import functional as F
from pycocotools.coco import COCO

class COCOTrain(Dataset):
    def __init__(self, path="/home/zr/COCO/", size = 608, flip=True):
        self.path = path
        self.flip = flip
        self.size = size

        self.dataType = "train2017"
        self.annfile = os.path.join(path, "annotations", "instances_{}.json".format(self.dataType))
        self.COCO = COCO(self.annfile)
        self.imagesId = self.COCO.getImgIds()

        self.classesId = self.COCO.getCatIds()
        self.classesName = [item['name'] for item in self.COCO.loadCats(self.classesId)]
        self.classMap = list(zip(self.classesId, self.classesName))

    def __getitem__(self, item):
        image , imagePath = self.getImage(item)
        labels = self.getLabels(item)
        image, labels = self.transform(image, labels, self.size, self.flip)
        return image.float(), labels.float(), imagePath

    def collate_fn(self, batch):
        images, labels, imagePaths = list(zip(*batch))
        images = t.stack(images, 0)
        Labels = []
        for i, label in enumerate(labels):
            per = t.zeros((len(label), 6))
            per[:, 0] = i
            per[:, 1:] = label
            Labels.append(per)
        labels = t.cat(Labels, 0)
        return images, labels, imagePaths

    def getImage(self, item):
        imageName = self.COCO.loadImgs(self.imagesId[item])[0]
        imagePath = os.path.join(self.path, self.dataType, imageName['file_name'])
        image = cv2.imread(imagePath)[:, :, ::-1]

        return image, imagePath

    def getLabels(self, item):
        annId = self.COCO.getAnnIds(self.imagesId[item], iscrowd=False)
        labels_id = self.COCO.loadAnns(annId)
        labels = []
        if len(labels_id) <= 0:
            labels = np.zeros((0, 5))
            return labels
        for label_id in labels_id:
            if label_id['bbox'][2] < 1 or label_id['bbox'][3] < 1:
                continue
            label = np.zeros((5))
            label[0] = label_id['category_id']
            label[1:] = label_id['bbox']
            labels.append(label)
        labels = np.stack(labels, 0)

        x, y, w, h =  labels[:, 1], labels[:, 2], labels[:, 3], labels[:, 4]
        labels[:, 3] = x + w
        labels[:, 4] = y + h
        #pycocotools class index start from 1, so -1
        labels[:, 0] = labels[:, 0] - 1
        return labels
    def transform(self, image, labels, size, flip):
        image = t.from_numpy(image / 255).float().permute((2, 0, 1)).contiguous()
        labels = t.from_numpy(labels).float()

        image, labels = self.resize(image, labels, size)
        image, labels = self.pad(image, labels, size)

        if flip:
            if random.random() < 0.5:
                image = t.flip(image, [2])
                cx = (labels[:, 1] + labels[:, 3]) * 0.5
                w = labels[:, 3] - labels[:, 1]

                cx = size - cx
                labels[:, 1] = cx - 0.5 * w
                labels[:, 3] = cx + 0.5 * w

        return image, labels
    def resize(self, image, labels, size):
        c, h, w = image.shape
        min_stride = min(size / h, size / w)
        image = F.interpolate(image.unsqueeze(0), (math.ceil(h * min_stride), math.ceil(w * min_stride))).squeeze(0)
        labels[:, 1:] = labels[:, 1:] * min_stride
        return image, labels
    def pad(self, image, labels, size):
        _, _h, _w = image.shape

        if _h == size:
            dif = (size - _w) // 2
            padding = (dif, size- _w - dif, 0, 0)
            labels[:, 1] = labels[:, 1] + dif
            labels[:, 3] = labels[:, 3] + dif

        elif _w == size:
            dif = (size - _h) // 2

            padding = (0, 0, dif, size - _h - dif)
            labels[:, 2] = labels[:, 2] + dif
            labels[:, 4] = labels[:, 4] + dif
        try:
            image = F.pad(image.unsqueeze(0), pad=padding).squeeze(0)
        except:
            print("!!!")
            image = t.zeros((3, size, size)).float()
            labels = t.zeros((0, 5)).float()
        return image, labels
    def __len__(self):
        return len(self.imagesId)


if __name__ == "__main__":
    a = COCOTrain()
    dataloader = DataLoader(dataset=a, batch_size = 1, shuffle=True, collate_fn=a.collate_fn, drop_last=True)
    for data, labels, paths in dataloader:
        image = data.permute((0, 2, 3, 1)).contiguous().numpy()[0]
        boxes = labels[labels[:,0]==0].numpy()
        for box in boxes:
            b, _, x1, y1, x2, y2 = box.tolist()
            print(box)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            # cv2.rectangle(image, x1, y1, x2, y2, (0, 255, 255), 2)
        cv2.imshow("win", image)
        cv2.waitKey(0)

