import random
class Config():
    detectWeights = "weights/Finally.pth"
    detetcDirs = "/home/zr/gitCode/data/samples/"
    detectResults = "/home/zr/gitCode/data/results/"
    classnum = 90
    size = 608
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(classnum)]
    preTrain = "models/resnet101.pth"
    batch = 8
    epoches = 100
    weights = "weights/"