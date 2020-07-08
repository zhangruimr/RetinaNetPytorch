import torch as t
import numpy as np
import cv2
import os
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.data import DataLoader
from models.model import RetinaNet
from datasets.datasets import COCOTrain
from config import Config
from utils.functions import *
def train():
    config = Config()
    batch = config.batch
    size = config.size
    epoches = config.epoches
    preTrain = config.preTrain
    weights = config.weights
    os.makedirs(weights, exist_ok=True)

    model = RetinaNet(weights=preTrain)
    #for name, param in model.named_parameters():
    #    print(name)
    #    print(param)
    if t.cuda.is_available():
        print("----GPU-Training----")
        model = t.nn.DataParallel(model, device_ids=[0, 2])
        model = model.cuda()
    #model.load_state_dict(t.load("weights/epoch96.pth"))

    model.train()
    optimer = Adam(model.parameters(), lr=0.01)
    optimer.zero_grad()
    scheduler = lr_scheduler.MultiStepLR(optimer, [80, 90], 0.1)
    datasets = COCOTrain()
    dataloader = DataLoader(datasets, batch_size=batch, shuffle=True, collate_fn=datasets.collate_fn, drop_last=True)

    for epoch in range(epoches):
        print("epoch-{}".format(epoch))
        for i, (imgs, labels, paths) in enumerate(dataloader):
            print("--epoch-{}-batch-{}--".format(epoch, i))
            if t.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()
            loss = model(imgs, labels).mean()
            print("Loss:", loss)
            loss.backward()
            if (i + 1) % 4 == 0:
                optimer.step()
                optimer.zero_grad()
        scheduler.step()
        if (epoch+1) % 8 == 0:
            t.save(model.state_dict(), weights + "epoch{}.pth".format(epoch+1))
    t.save(model.state_dict(), weights + "Finally.pth")



if __name__ == "__main__":
    train()