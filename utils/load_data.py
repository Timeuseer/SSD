#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Create Time : 2022/3/12 14:37 
@Author : Gabriel
@File : load_data.py 
@Project:SSD
@About : 数据加载与预处理
'''

import math
import cv2
import numpy as np

from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from random import shuffle
from utils.utils import cvtColor


class SSDDatasets(keras.utils.Sequence):
    def __init__(self, annotation_lines, input_shape, anchors, batch_size, num_classes,
                 train, overlap_threshold=0.5):
        self.annotation_lines = annotation_lines
        self.length = len(self.annotation_lines)
        self.input_shape = input_shape
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.train = train
        self.overlap_threshold = overlap_threshold

    def __len__(self):
        return math.ceil(len(self.annotation_lines) / float(self.batch_size))

    def __getitem__(self, index):
        img_data = []
        box_data = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            i = i % self.length
            '''
            训练时对数据进行随机增强
            验证时不对数据进行随机增强
            '''
            img, box = self.get_random_data(self.annotation_lines[i], self.input_shape, random=self.train)
            if len(box) != 0:
                boxes = np.array(box[:, :4], dtype=np.float32)
                boxes[:, [0, 2]] = boxes[:, [0, 2]] / self.input_shape[1]
                boxes[:, [1, 3]] = boxes[:, [1, 3]] / self.input_shape[0]
                one_hot_label = np.eye(self.num_classes - 1)[np.array(box[:, 4], np.int32)]
                box = np.concatenate([boxes, one_hot_label], axis=-1)

            box = self.assign_boxes(box)
            img_data.append(img)
            box_data.append(box)

        return preprocess_input(np.array(img_data)), np.array(box_data)

    def generate(self):
        i = 0
        while True:
            img_data = []
            box_data = []
            for b in range(self.batch_size):
                if i == 0:
                    np.random.shuffle(self.annotation_lines)
                '''
                训练时对数据进行随机增强
                验证时不对数据进行随机增强
                '''
                img, box = self.get_random_data(self.annotation_lines[i], self.input_shape, random=self.train)
                i = (i + 1) % self.length
                if len(box) != 0:
                    boxes = np.array(box[:, :4], dtype=np.float32)
                    boxes[:, [0, 2]] = boxes[:, [0, 2]] / self.input_shape[1]
                    boxes[:, [1, 3]] = boxes[:, [1, 3]] / self.input_shape[0]
                    one_hot_label = np.eye(self.num_classes - 1)[np.array(box[:, 4], np.int32)]
                    box = np.concatenate([boxes, one_hot_label], axis=-1)

                box = self.assign_boxes(box)
                img_data.append(img)
                box_data.append(box)

            yield preprocess_input(np.array(img_data)), np.array(box_data)

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=0.3, hue=0.1,
                        sat=1.5, val=1.5, random=True):
        line = annotation_line.split()
        '''
        读取图片并转换成RGB图像
        '''
        img = Image.open(line[0])
        img = cvtColor(img)

        '''
        获取图像的高宽和目标高宽
        '''
        iw, ih = img.size
        w, h = input_shape

        '''
        获得预测框
        '''
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            '''
            将图像多图的部分加上灰条
            '''
            img = img.resize((nw, nh), Image.BICUBIC)
            new_img = Image.new('RGB', (w, h), (128, 128, 128))
            new_img.paste(img, (dx, dy))
            img_data = np.array(new_img, np.float32)

            '''
            对真实框进行调整
            '''
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                bow_w = box[:, 2] - box[:, 0]
                bow_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(bow_w > 1, bow_h > 1)]

            return img_data, box

        '''
        对图像进行缩放并进行长和宽的扭曲
        '''
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw * new_ar)
        img = img.resize((nw, nh), Image.BICUBIC)

        '''
        将图像多图的部分加上灰条
        '''
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_img = Image.new('RGB', (w, h), (128, 128, 128))
        new_img.paste(img, (dx, dy))
        img = new_img

        '''
        翻转图像
        '''
        flip = self.rand() < 0.5
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        '''
        色域扭曲
        '''
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < 0.5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < 0.5 else 1 / self.rand(1, val)
        x = cv2.cvtColor(np.array(img, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 0] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        img_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        '''
        对真实框进行调整
        '''
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            bow_w = box[:, 2] - box[:, 0]
            bow_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(bow_w > 1, bow_h > 1)]

        return img_data, box

    def on_epoch_begin(self):
        shuffle(self.annotation_lines)

    def iou(self, box):
        '''
        计算出每个真实框与所有的先验框的IOU
        判断真实框与先验框的重合程度
        '''
        inter_upleft = np.maximum(self.anchors[:, :2], box[:2])
        inter_bottomleft = np.minimum(self.anchors[:, 2:4], box[2:])

        inter_wh = inter_bottomleft - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]

        '''
        真实框的面积
        '''
        area_true = (box[2] - box[0]) * (box[3] - box[1])

        '''
        先验框的面积
        '''
        area_gt = (self.anchors[:, 2] - self.anchors[:, 0]) * (self.anchors[:, 3] - self.anchors[:, 1])

        '''
        计算IOU
        '''
        union = area_true + area_gt - inter

        iou = inter / union

        return iou

    def encode_box(self, box, return_iou=True, variances=[0.1, 0.1, 0.2, 0.2]):
        '''
        计算当前真实框和先验框的重合情况
        iou         [self.num_anchors]
        encode_box  [self.num_anchors,5]
        '''
        iou = self.iou(box)
        encode_box = np.zeros((self.num_anchors, 4 + return_iou))

        '''
        找到每一个真实框，重合度较高的先验框
        可以由这一个先验框来负责真实框的预测
        '''
        assign_mask = iou > self.overlap_threshold

        '''
        如果没有一个重合程度大于self.overlap_threshold
        则选择重合度最大的为正样本
        '''
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True

        '''
        利用IOU进行赋值
        '''
        if return_iou:
            encode_box[:, -1][assign_mask] = iou[assign_mask]

        '''
        找到对应的先验框
        '''
        assign_anchors = self.anchors[assign_mask]

        '''
        逆向编码，将真实框转化为SSD预测结果的格式
        先计算真实框的中心和长宽
        '''
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]

        '''
        在计算重合度较高的先验框中心与长宽
        '''
        assign_anchors_center = (assign_anchors[:, 0:2] + assign_anchors[:, 2:4]) * 0.5
        assign_anchors_wh = (assign_anchors[:, 2:4] - assign_anchors[:, 0:2])

        '''
        逆向求取SSD应该有的预测结果
        先求取中心的预测结果，再求取宽高的预测结果
        存在改变数量级的参数，默认为[0.1, 0.1, 0.2, 0.2]
        '''
        encode_box[:, :2][assign_mask] = box_center - assign_anchors_center
        encode_box[:, :2][assign_mask] /= assign_anchors_wh
        encode_box[:, :2][assign_mask] /= np.array(variances)[:2]

        encode_box[:, 2:4][assign_mask] = np.log(box_wh / assign_anchors_wh)
        encode_box[:, 2:4][assign_mask] /= np.array(variances)[2:4]

        return encode_box.ravel()

    def assign_boxes(self, boxes):
        '''
        assignment分为三个部分:
        :4      的内容为网络应该有的回归预测结果
        4:-1    的内容为先验框所对应的种类，默认为背景
        -1      的内容为先验框中是否包含物体
        '''
        assignment = np.zeros((self.num_anchors, 4 + self.num_classes + 1))
        assignment[:, 4] = 1.0
        if len(boxes) == 0:
            return assignment

        '''
        对每一个真实框都计算IOU
        '''
        encode_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])

        '''
        encode_boxes的shape变换为: [num_true_box, num_anchors, 4 + 1]
        其中4为编码后的结果，1为IOU
        '''
        encode_boxes = encode_boxes.reshape(-1, self.num_anchors, 4 + 1)

        '''
        [num_anchors] 求取每一个先验框重合程度最大的真实框
        '''
        best_iou = encode_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encode_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]

        '''
        计算一共有多少个先验框满足要求
        '''
        assign_num = len(best_iou_idx)

        '''
        取出编码后的真实框
        '''
        encode_boxes = encode_boxes[:, best_iou_mask, :]

        '''
        对编码后的真实框进行赋值
        '''
        assignment[:, :4][best_iou_mask] = encode_boxes[best_iou_idx, np.arange(assign_num), :4]

        '''
        4代表为背景的概率，设置为0，因为这里取出的先验框都是有物体的
        '''
        assignment[:, 4][best_iou_mask] = 0
        assignment[:, 5:-1][best_iou_mask] = boxes[best_iou_idx, 4:]

        '''
        -1表示先验框是否有对应的物体
        '''
        assignment[:, -1][best_iou_mask] = 1

        return assignment
