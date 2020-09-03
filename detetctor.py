import torch as t
from datasets.myDatasets import *
from models.model import *
import numpy as np
import cv2
import os
from config import Config
from utils.functions import *
import math
def detect(model, img, classname, colors, size=608):
    image = test_process(img, (size, size))

    if t.cuda.is_available():
        image = image.cuda()
    with t.no_grad():
        detection = model(image)[0]
    if detection is None:
        return img

    detection = cal_box(detection, img.shape, size)
    detection = clip_box(detection, size).cpu().numpy()

    for box in detection:

        x1, y1, x2, y2, score, cls = int(box[0]), int(box[1]), int(box[2]), int(box[3]), round(float(box[4]), 2), int(box[-1])
        cv2.rectangle(img, (x1, y1), (x2, y2), colors[cls], 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, classname[cls]+":"+str(score), (x1, y1+25), font, 1, colors[cls], 1)

    return img

if __name__ == "__main__":
    config = Config()
    weights = config.detectWeights
    detectDirs = config.detetcDirs
    detectResults = config.detectResults
    classNum = config.classnum
    colors = config.colors
    classname = config.classname
    size = config.size

    preTrain = config.preTrain
    os.makedirs(detectResults, exist_ok=True)
    filenames = os.listdir(detectDirs)


    model = RetinaNet(weights=preTrain, classNum=classNum)
    if t.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(t.load(weights))
    model.eval()

    for filename in filenames:
        road = detectDirs + filename

        img = cv2.imread(road)
        img = detect(model, img, classname, colors, size)
        cv2.imwrite(detectResults + filename, img)
