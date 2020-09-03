import torch as t
from datasets.myDatasets import *
from models.model import *
import numpy as np
import cv2
import os
from config import Config
from utils.functions import *
import math
from detetctor import *
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
    model = RetinaNet(weights=preTrain, classNum=classNum)
    if t.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(t.load(weights))
    model.eval()

    video = cv2.VideoCapture("cat.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videowriter = cv2.VideoWriter(detectResults+"cat.avi", fourcc, 20.0, (int(video.get(4)), int(video.get(3))), True)
    freq = 0
    if video.isOpened():
        while cv2.waitKey(20) != ord(' '):
            ret, frame = video.read()
            if not ret:
                break
            img = detect(model, frame, classname, colors, size)
            videowriter.write(frame)
            cv2.imshow("win", frame)

