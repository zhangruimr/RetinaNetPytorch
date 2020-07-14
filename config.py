import random
class Config():
    detectWeights = "weights/epoch49.pth"
    detetcDirs = "/home/zr/GithubCode/data/samples/"
    detectResults = "/home/zr/GithubCode/data/results/"

    classnum = 20
    colors = [(random.randint(80, 200), random.randint(80, 200), random.randint(80, 200)) for i in range(classnum)]
    size = 608
    classname = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    start_lr = 0.0001
    lr_change = [40, 80]
    lr_decay = 0.1
    preTrain = "models/resnet101.pth"
    trainWeights = "weights/epoch49.pth"
    if not trainWeights == None:
        preTrain = None
    batch = 12
    epoches = 160
    weightsSave = "weights/"