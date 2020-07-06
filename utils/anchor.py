import numpy as np
import torch as t
import cv2
def baseAnchor(baseSize=16, ratios=[0.5, 1, 2], scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]):
    AnchorNum = len(ratios) * len(scales)
    anchor = np.zeros((AnchorNum, 4))
    scale = np.array(scales)
    ratio = np.array(ratios)

    l = baseSize * np.tile(scale, (3, 1)).T.flatten()
    areas = l * l
    anchor[:, 2] = np.sqrt(areas / np.tile(ratio, (1, 3)).flatten())
    anchor[:, 3] = anchor[:, 2] * np.tile(ratio, (1, 3)).flatten()

    #anchor[:,0:2] += 0.5
    #print(anchor[:, 2::4].shape)
    anchor[:, 0] -= anchor[:, 2] * 0.5
    anchor[:, 1] -= anchor[:, 3] * 0.5
    anchor[:, 2] = anchor[:, 2] * 0.5
    anchor[:, 3] = anchor[:, 3] * 0.5
    #print(anchor)
    return anchor

def layerAnchor(size, stride, baseAnchor):
    b, c, h, w = size

    anchorAnchor = baseAnchor
    #print(anchorAnchor)

    grid_x = np.arange(0, w) * stride
    grid_y = np.arange(0, h) * stride

    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    grid_x = grid_x.T + 0.5 * stride
    grid_y = grid_y.T + 0.5 * stride
    grid = np.tile(np.stack((grid_x, grid_y), -1).reshape((-1, 2)), (1, 2))

    grid = np.tile(np.expand_dims(grid, 1), (1, 9, 1))
    #print(grid)
    anchorAnchor = np.tile(np.expand_dims(anchorAnchor, 0), (grid.shape[0], 1, 1))
    layer_anchor = grid + anchorAnchor
    #print(layer_anchor)
    layer_anchor = np.reshape(layer_anchor, (-1, 4))
    layer_anchor = np.tile(layer_anchor, (b, 1, 1))
    return layer_anchor
def generateAnchor(rpnFeaturemaps, oriSize = 608,  ratios=[0.5, 1, 2], scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]):

    all_anchor = []
    for i, layer in enumerate(rpnFeaturemaps):
        base_anchor = baseAnchor(2 ** (i + 5), ratios, scales)
        size = layer.shape
        stride = oriSize // size[-1]
        all_anchor.append(layerAnchor(size, stride, base_anchor))
    all_anchor = t.from_numpy(np.concatenate(all_anchor, 1)).float()
    if t.cuda.is_available():
        return all_anchor.cuda()
    else:
        return all_anchor

if __name__ == "__main__":

    pic = cv2.imread("../test.jpg")
    pic = cv2.resize(pic, (608, 608))
    anchor = baseAnchor(2**7)
    anchor = layerAnchor((1, 3, 38, 38), 16, anchor)[0]
    #print(len(anchor))
    #cv2.imshow("win", pic)
    i = 0
    for box in anchor:
     #if i % 5 == 0:
      #print(box)
      x1, y1, x2, y2 = box.tolist()
      #print((int(x1) * 32, int(y1) * 32), (int(x2) * 32, int(y2) * 32))
      cv2.rectangle(pic, (int(x1) , int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
      cv2.circle(pic, (int( (int(x1)+int(x2))*0.5 ), int( (int(y1)+int(y2))*0.5 )), 2, (255, 0, 0), -1)
      #print((int( (int(x1)*32+int(x1)*32)*0.5 ), int( (int(y1)*32+int(y2)*32)*0.5 )))
      cv2.imshow("win", pic)
      cv2.waitKey(0)
      i += 1
