import torch as t
import torch.nn as nn
import resnet
import math
def init(model):
    for name, param in model.named_parameters():
        if name.find("bias") >= 0:
            nn.init.constant_(param, 0)
        elif name.find("weight") >= 0:
            nn.init.normal_(param, 0, 0.01)
class Resnet(resnet.ResNet):

    #默认使用resnet101
    def __init__(self, block=resnet.Bottleneck, layers=[3, 4, 23, 3], weights=None):
        super(Resnet, self).__init__(block, layers)
        if not weights == None:
            self.load_state_dict(t.load(weights))
        print("-----加载預训练成功-----")
        del self.avgpool
        del self.fc

def Backbone(resnettype = 101, weights=None):

    block = resnet.Bottleneck
    layers = [3, 4, 23, 3]

    if resnettype == 101:
        model = Resnet(block, layers, weights)
        return model
    elif resnettype == 152:
        layers = [3, 8, 36, 3]
        model = Resnet(block, layers, weights)
        return model

class FPN(nn.Module):
    def __init__(self, inputChannels, outChannel=256):
        super(FPN, self).__init__()
        c3_channel, c4_channel, c5_channel = inputChannels

        self.p5_1 = nn.Conv2d(c5_channel, outChannel, (1, 1), 1, 0)
        self.p5_up = nn.Upsample(scale_factor=2, mode="nearest")
        self.p5_2 = nn.Conv2d(outChannel, outChannel, (3, 3), 1, 1)

        self.p4_1 = nn.Conv2d(c4_channel, outChannel, (1, 1), 1, 0)
        self.p4_up = nn.Upsample(scale_factor=2, mode="nearest")
        self.p4_2 = nn.Conv2d(outChannel, outChannel, (3, 3), 1, 1)

        self.p3_1 = nn.Conv2d(c3_channel, outChannel, (1, 1), 1, 0)
        self.p3_up = nn.Upsample(scale_factor=2, mode="nearest")
        self.p3_2 = nn.Conv2d(outChannel, outChannel, (3, 3), 1, 1)

        self.p6 = nn.Conv2d(c5_channel, outChannel, (3, 3), 2, 1)

        self.p7_relu = nn.ReLU()
        self.p7_conv = nn.Conv2d(outChannel, outChannel, (3, 3), 2, 1)
        init(self)
    def forward(self, inputs):
        c3, c4, c5 = inputs

        p5_x = self.p5_1(c5)
        p5_upsample = self.p5_up(p5_x)
        p5_x = self.p5_2(p5_x)

        p4_x = self.p4_1(c4)
        p4_x = p4_x + p5_upsample
        p4_upsample = self.p4_up(p4_x)
        p4_x = self.p4_2(p4_x)

        p3_x = self.p3_1(c3)
        p3_x = p3_x + p4_upsample
        p3_x = self.p3_2(p3_x)

        p6_x = self.p6(c5)
        #print(p6_x.shape)

        p7_x = self.p7_relu(p6_x)
        p7_x = self.p7_conv(p7_x)
       # print(p7_x.shape)
        output = [p3_x, p4_x, p5_x, p6_x, p7_x]
        #output = [p3_x, p4_x, p5_x]
        return output

class Classification(nn.Module):
    def __init__(self, inChannel=256, outChannel=256, anchorNum = 9, classNum = 90):
        super(Classification, self).__init__()

        self.anchorNum = anchorNum
        self.classNum = classNum
        self.layers = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, (3, 3), 1, 1),
            nn.ReLU(),
            nn.Conv2d(outChannel, outChannel, (3, 3), 1, 1),
            nn.ReLU(),
            nn.Conv2d(outChannel, outChannel, (3, 3), 1, 1),
            nn.ReLU(),
            nn.Conv2d(outChannel, outChannel, (3, 3), 1, 1),
            nn.ReLU()
        )
        init(self.layers)

        self.post_layer = nn.Conv2d(outChannel, anchorNum * classNum, (3, 3), 1, 1)
        nn.init.normal_(self.post_layer.weight, 0, 0.01)
        nn.init.constant_(self.post_layer.bias, -math.log((1 - 0.1) / 0.1))
        self.sig = nn.Sigmoid()


    def forward(self, input):
        output = self.layers(input)
        output = self.post_layer(output)
        output = self.sig(output)
        batch, channel, height, width = input.shape
        output = output.permute(0, 2, 3, 1).contiguous()
        output = output.view(batch, -1, self.classNum).contiguous()
        return output

class Regression(nn.Module):
    def __init__(self, inChannel=256, outChannel=256, anchorNum=9):
        super(Regression, self).__init__()
        self.anchorNum = anchorNum

        self.layers = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, (3, 3), 1, 1),
            nn.ReLU(),
            nn.Conv2d(outChannel, outChannel, (3, 3), 1, 1),
            nn.ReLU(),
            nn.Conv2d(outChannel, outChannel, (3, 3), 1, 1),
            nn.ReLU(),
            nn.Conv2d(outChannel, outChannel, (3, 3), 1, 1),
            nn.ReLU()
        )
        init(self.layers)

        self.post_layer = nn.Conv2d(outChannel, anchorNum * 4, (3, 3), 1, 1)
        nn.init.normal_(self.post_layer.weight, 0, 0.01)
        nn.init.constant_(self.post_layer.bias, -math.log((1 - 0.1) / 0.1))

    def forward(self, input):
        output = self.layers(input)
        output = self.post_layer(output)
        batch, channel, height, width = output.shape
        output = output.permute(0, 2, 3, 1).contiguous()

        output = output.view(batch, height, width, self.anchorNum, 4).contiguous()
        output = output.view(batch, -1, 4).contiguous()

        return output