import torch as t
import torch.nn as nn
from  utils.anchor import *
from subModels import *
from loss import loss
from utils.functions import *
#初始化模型参数

class RetinaNet(nn.Module):
    def __init__(self, backbone = 101, weights="resnet101.pth"):
        super(RetinaNet, self).__init__()
        self.backbone = Backbone(backbone, weights)

        #resnet101是[512, 1024, 2048]
        self.fpn = FPN([512, 1024, 2048])
        self.cls = Classification(256)
        self.reg = Regression(256)

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

            results = []
            for i in range(len(regression)):
                encode_reg_output = decodeBox(all_anchor[i], regression[i])
                cls_output = classify[i]

                max_val, max_id = t.max(cls_output, 1)
                mask = max_val > 0.05
                if not t.sum(mask) > 0:
                    results.append(None)
                    continue
                encode_reg_output = encode_reg_output[mask]
                cls_output = cls_output[mask]

                results.append(nms(cls_output, encode_reg_output))
            return results






if __name__ == "__main__":
    model = RetinaNet()
    model.train()
    x = t.ones((3, 3, 608, 608))
    x = model(x)
#权重下载后加载












