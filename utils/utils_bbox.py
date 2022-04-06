#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Create Time : 2022/3/12 16:03 
@Author : Gabriel
@File : utils_bbox.py 
@Project:SSD
@About : 
'''

import numpy as np
import tensorflow as tf


class BBoxUtility(object):
    def __init__(self, num_classes, nms_threshold=0.45, top_k=300):
        self.num_classes = num_classes
        self._nms_threshold = nms_threshold
        self._top_k = top_k

    def ssd_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_img):
        '''
        把y轴放前面，这样有利于预测框和图像的宽高相乘
        '''
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_img:
            '''
            offset表示图像有效区域相对于图像左上角的偏移情况
            '''
            new_shape = np.round(image_shape * np.min(input_shape / image_shape))
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape

            box_yx = (box_yx - offset) * scale
            box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxs = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxs[..., 0:1], box_maxs[..., 1:2]],
                               axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)

        return boxes

    def decode_boxes(self, mbox_loc, anchors, variances):
        '''
        获得先验框的宽高
        '''
        anchor_w = anchors[:, 2] - anchors[:, 0]
        anchor_h = anchors[:, 3] - anchors[:, 1]

        '''
        获得先验框的中心点
        '''
        anchor_center_x = 0.5 * (anchors[:, 2] + anchors[:, 0])
        anchor_center_y = 0.5 * (anchors[:, 3] + anchors[:, 1])

        '''
        真实框聚类先验框中心的xy轴偏移情况
        '''
        decode_bbox_center_x = mbox_loc[:, 0] * anchor_w * variances[0]
        decode_bbox_center_x += anchor_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * anchor_h * variances[1]
        decode_bbox_center_y += anchor_center_y

        '''
        真实框的宽高
        '''
        decode_bbox_w = np.exp(mbox_loc[:, 2] * variances[2])
        decode_bbox_w *= anchor_h
        decode_bbox_h = np.exp(mbox_loc[:, 3] * variances[3])
        decode_bbox_h *= anchor_h

        '''
        获取真实框的左上角和右下角
        '''
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_w
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_h
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_w
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_h

        '''
        对真实框的左上角和右下角信息堆叠
        '''
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]), axis=-1)

        '''
        防止超出0和1
        '''
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)

        return decode_bbox

    def decode_box(self, predications, anchors, image_shape, input_shape, letterbox_img, variances=[0.1, 0.1, 0.2, 0.2],
                   confidence=0.5):
        '''
        :4  为回归预测结果
        '''
        mbox_loc = predications[:, :, :4]

        '''
        获得种类的置信度
        '''
        mbox_conf = predications[:, :, 4:]

        results = []

        '''
        对每一张图片进行处理
        '''
        for i in range(len(mbox_loc)):
            results.append([])
            '''
            利用回归结果对先验框进行解码
            '''
            decode_bbox = self.decode_boxes(mbox_loc[i], anchors, variances)

            for c in range(1, self.num_classes):  # 因为0为背景，所以跳过
                '''
                取出属于该类的所有框的置信度，并判断是否大于阈值
                '''
                c_confs = mbox_conf[i, :, c]
                c_confs_m = c_confs > confidence
                if len(c_confs[c_confs_m]) > 0:
                    '''
                    取出得分高于confidence的值
                    '''
                    boxes_to_process = decode_bbox[c_confs_m]
                    confs_to_process = c_confs[c_confs_m]

                    '''
                    进行IOU的非极大抑制
                    '''
                    idx = tf.image.non_max_suppression(tf.cast(boxes_to_process, tf.float32),
                                                       tf.cast(confs_to_process, tf.float3232),
                                                       self._top_k, iou_threshold=self._nms_threshold).numpy()

                    '''
                    取出在非极大抑制中较好的内容
                    '''
                    good_boxes = boxes_to_process[idx]
                    confs = confs_to_process[idx][:, None]
                    labels = (c - 1) * np.ones((len(idx), 1))

                    '''
                    将标签、置信度、框的位置进行堆叠
                    '''
                    c_pred = np.concatenate((good_boxes, labels, confs), axis=1)

                    results[-1].extend(c_pred)

                '''
                从有灰条的结果转化为没有灰条
                '''
                if len(results[-1]) > 0:
                    results[-1] = np.array(results[-1])
                    box_xy, box_wh = (results[-1][:, 0:2] + results[-1][:, 2:4]) / 2, results[-1][:, 2:4] - results[-1][
                                                                                                            :, 0:2]
                    results[-1][:, :4] = self.ssd_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_img)

            return results
