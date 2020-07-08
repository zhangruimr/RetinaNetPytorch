import torch as t
from datasets.datasets import *
from models.model import *
import numpy as np
import cv2
import os
from config import Config
from utils.functions import *

def detect(model, img, size=608):
    image = transform(img).unsqueeze(0)
    if image == None:
        return None
    with t.no_grad():
        detection = model(image)[0]
    detection = cal_box(detection, img.shape, size).numpy()
    for box in detection:
        x1, y1, x2, y2, score, cls = int(box[0]), int(box[1]), int(box[2]). int(box[3]), float('%.2f', score), int(box[-1])
        cv2.rectangle(img, (x1, y1), (x2, y2), colors[cls], 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "--", (x1, y1), font, 0.6, (0, 0, 0), 1)
    return img

if __name__ == "__main__":
    config = Config()
    weights = config.detectWeights
    detectDirs = config.detetcDirs
    detectResults = config.detectResults
    colors = config.colors
    classname = config.classname
    size = config.size

    preTrain = config.preTrain
    os.makedirs(detectResults, exist_ok=True)
    filenames = os.listdir(detectDirs)

    model = RetinaNet(weights=preTrain)
    if t.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(t.load(weights))
    model.eval()

    for filename in filenames:
        road = detectDirs + filename
        img = cv2.imread(road)
        img = detect(model, img, size)
        cv2.imwrite(detectResults + filename, img)
