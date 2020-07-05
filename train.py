import torch as t
import os
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.data import DataLoader
from models.model import RetinaNet
from datasets.datasets import COCOTrain, COCOServer
from config import Config
from utils.functions import *
def train():
    config = Config()
    epoches = config.epoches
    preTrain = config.preTrain
    os.makedirs("weights", exist_ok=True)

    model = RetinaNet(weights=preTrain)
    #for name, param in model.named_parameters():
    #    print(name)
    #    print(param)
    if t.cuda.is_available():
        print("^^^^GPU-Training^^^^")
        model = model.cuda()
    model.train()
    optimer = Adam(model.parameters(), lr=0.001)
    optimer.zero_grad()
    scheduler = lr_scheduler.MultiStepLR(optimer, [80, 90], 0.1)
    datasets = COCOServer()
    dataloader = DataLoader(datasets, batch_size=1, collate_fn=datasets.collate_fn, drop_last=True)

    for epoch in range(epoches):
        for i, (imgs, labels, paths) in enumerate(dataloader):
            if t.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()
            loss = model(imgs, labels)
            print(loss)
            loss.backward()
            if (i + 1) % 10 == 0:
                optimer.step()
                optimer.zero_grad()
        scheduler.step()
        if (epoch+1) % 8 == 0:
            t.save(model.parameters(), "weights/epoch{}.pth".format(epoch+1))
    t.save(model.parameters(), "weights/Finally.pth")



if __name__ == "__main__":
    train()