#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Create Time : 2022/3/11 10:24 
@Author : Gabriel
@File : anchors.py 
@Project:SSD
@About : 对anchor进行处理
'''
import numpy as np


class AnchorBox(object):
    def __init__(self, input_shape, min_size, max_size=None, aspect_ratios=None):
        self.input_shape = input_shape
        self.min_size = min_size
        self.max_size = max_size

        self.aspect_ratios = []
        for ar in aspect_ratios:
            self.aspect_ratios.append(ar)
            self.aspect_ratios.append(1. / ar)

    def call(self, layer_shape):
        # 获取输入进来的特征层的宽和高
        layer_h = layer_shape[0]
        layer_w = layer_shape[1]

        # 获取输入图片的宽和高
        img_h = self.input_shape[0]
        img_w = self.input_shape[1]

        box_ws = []
        box_hs = []

        # 按照aspect_ratios填充
        for ar in self.aspect_ratios:
            # 先填充一个小的正方形
            if ar == 1 and len(box_ws) == 0:
                box_ws.append(self.min_size)
                box_hs.append(self.min_size)
            # 再填充一个大的正方形
            elif ar == 1 and len(box_ws) > 0:
                box_ws.append(np.sqrt(self.min_size * self.max_size))
                box_hs.append(np.sqrt(self.min_size * self.max_size))
            elif ar != 1:
                box_ws.append(self.min_size * np.sqrt(ar))
                box_hs.append(self.min_size / np.sqrt(ar))

        # 获得所有先验框的宽高1/2
        box_ws = 0.5 * np.array(box_ws)
        box_hs = 0.5 * np.array(box_hs)

        # 每一个特征层对应的步长
        step_x = img_w / layer_w
        step_y = img_h / layer_h

        # 生成网格中心
        lin_x = np.linspace(0.5 * step_x, img_w - 0.5 * step_x, layer_w)
        lin_y = np.linspace(0.5 * step_y, img_h - 0.5 * step_y, layer_h)
        centers_x, centers_y = np.meshgrid(lin_x, lin_y)
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)

        # 获得先验框的左上角和右下角
        num_anchors_ = len(self.aspect_ratios)
        anchor_boxes = np.concatenate((centers_x, centers_y), axis=1)
        anchor_boxes = np.tile(anchor_boxes, (1, 2 * num_anchors_))

        anchor_boxes[:, ::4] -= box_ws
        anchor_boxes[:, 1::4] -= box_hs
        anchor_boxes[:, 2::4] += box_ws
        anchor_boxes[:, 3::4] += box_hs

        # 归一化
        anchor_boxes[:, ::2] /= img_w
        anchor_boxes[:, 1::2] /= img_h
        anchor_boxes = anchor_boxes.reshape(-1, 4)

        # 防止超界
        anchor_boxes = np.minimum(np.maximum(anchor_boxes, 0.0), 1.0)

        return anchor_boxes


def get_img_output_length(h, w):
    '''
    计算共享特征层的大小
    :param h: 高
    :param w: 宽
    :return:
    '''
    filter_size = [3, 3, 3, 3, 3, 3, 3, 3]
    padding = [1, 1, 1, 1, 1, 1, 0, 0]
    stride = [2, 2, 2, 2, 2, 2, 1, 1]
    feature_hs = []
    feature_ws = []

    for i in range(len(filter_size)):
        h = (h + 2 * padding[i] - filter_size[i]) // stride[i] + 1
        w = (w + 2 * padding[i] - filter_size[i]) // stride[i] + 1
        feature_hs.append(h)
        feature_ws.append(w)

    return np.array(feature_hs)[-6:], np.array(feature_ws)[-6:]


def get_anchors(input_shape=[300, 300], anchors_size=[30, 60, 111, 162, 213, 264, 315]):
    feature_hs, feature_ws = get_img_output_length(input_shape[0], input_shape[1])
    anchors_ratios = [[1, 2], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2], [1, 2]]
    anchors = []
    for i in range(len(feature_hs)):
        anchors.append(AnchorBox(input_shape, anchors_size[i], max_size=anchors_size[i + 1],
                                 aspect_ratios=anchors_ratios[i]).call([feature_hs[i], feature_ws[i]]))

    anchors = np.concatenate(anchors, 0)

    return anchors
