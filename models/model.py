import torch as t
import torch.nn as nn
from  utils.anchor import *
from subModels import *
from loss import loss
#初始化模型参数

class RetinaNet(nn.Module):
    def __init__(self, backbone = 101, weights="resnet101.pth"):
        super(RetinaNet, self).__init__()
        self.backbone = Backbone(backbone, weights)

        #resnet101是[512, 1024, 2048]
        self.fpn = FPN([512, 1024, 2048])
        self.cls = Classification(256)
        self.reg = Regression(256)
        self.baseAnchor = baseAnchor()
        self.loss = loss()
    def forward(self, input, labels = None):
        features = self.backbone(input)
        detectFeatures = self.fpn(features)
        all_anchor = generateAnchor(detectFeatures)
        #print(all_anchor.shape)
        classify = []
        regression = []
        for detectFeature in detectFeatures:
            classify.append(self.cls(detectFeature))
            regression.append(self.reg(detectFeature))
        classify = t.cat(classify, 1)
        regression = t.cat(regression, 1)
        if self.training == True:
            loss = self.loss(classify, regression, labels, all_anchor)
            return loss
        else:
            pass


if __name__ == "__main__":
    model = RetinaNet()
    x = t.ones((3, 3, 608, 608))
    model(x)
#权重下载后加载












