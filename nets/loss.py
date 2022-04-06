#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Create Time : 2022/3/10 11:15 
@Author : Gabriel
@File : loss.py 
@Project:SSD
@About : 
'''

import tensorflow as tf


class MultiboxLoss(object):
    def __init__(self, num_classes, alpha=1.0, neg_pos_ratio=3.0,
                 background_label_id=0, negatives_for_hard=100.0):
        self.num_classes = num_classes
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        if background_label_id != 0:
            raise Exception('Only 0 as background label id is supported')
        self.background_label_id = background_label_id
        self.negatives_for_hard = negatives_for_hard

    def _l1_smooth_loss(self, y_true, y_pred):
        abs_loss = tf.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred) ** 2
        l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
        return tf.reduce_sum(l1_loss, -1)

    def _softmax_loss(self, y_true, y_pred):
        y_pred = tf.maximum(y_pred, 1e-7)
        softmax_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

        return softmax_loss

    def compute_loss(self, y_true, y_pred):
        # y_true:[batch_size,8732,4+num_classes+1]
        # y_pred:[batch_size,8732,4+num_classes+1]
        num_boxes = tf.cast(tf.shape(y_true)[1], tf.float32)

        '''
        分类的loss
        [batch_size,8732,21] -> [batch_size,8732]
        '''
        conf_loss = self._softmax_loss(y_true[:, :, 4:-1], y_pred[:, :, 4:])

        '''
        位置的loss
        [batch_size,8732,4] -> [batch_size,8732]
        '''
        loc_loss = self._l1_smooth_loss(y_true[:, :, :4], y_pred[:, :, :4])

        '''
        获取所有正标签的loss
        '''
        pos_loc_loss = tf.reduce_sum(loc_loss * y_true[:, :, -1], axis=1)
        pos_conf_loss = tf.reduce_sum(conf_loss * y_true[:, :, -1], axis=1)

        '''
        获取每一张图中正样本的个数,shape=[batch_size,]
        '''
        num_pos = tf.reduce_sum(y_true[:, :, -1], axis=-1)

        '''
        获取每一张图中负样本的个数，shape=[batch_size,]
        '''
        num_neg = tf.minimum(num_pos * self.neg_pos_ratio, num_boxes - num_pos)

        # 找到大于0的值，
        pos_num_neg_mask = tf.greater(num_neg, 0)

        '''
        如果所有图中都没有正样本
        则默认选取100个先验框作为负样本
        '''
        has_min = tf.cast(tf.reduce_any(pos_num_neg_mask), tf.float32)
        num_neg = tf.concat(axis=0, values=[num_neg, [(1 - has_min) * self.negatives_for_hard]])

        '''
        求整个batch中负样本数量总和
        '''
        num_neg_batch = tf.reduce_sum(tf.boolean_mask(num_neg, tf.greater(num_neg, 0)))
        num_neg_batch = tf.cast(num_neg_batch, tf.int32)

        '''
        对预测结果进行判断，如果该先验框中没有包含物体，在它的不属于背景的预测概率过大时，就很难分类样本
        '''
        confs_start = 4 + self.background_label_id + 1
        confs_end = confs_start + self.num_classes - 1

        '''
        把不是背景的概率求和，求和后的概率越大，则越难分类,shape=[batch_size,8732]
        '''
        max_confs = tf.reduce_sum(y_pred[:, :, confs_start:confs_end], axis=2)

        '''
        只有没有包含物体的先验框才得到保留
        在整个batch里面选取最难分类的num_neg_batch个先验框作为负样本
        '''
        max_confs = tf.reshape(max_confs * (1 - y_pred[:, :, -1]), [-1])
        _, indices = tf.nn.top_k(max_confs, k=num_neg_batch)

        neg_conf_loss = tf.gather(tf.reshape(conf_loss, [-1]), indices)

        # 进行归一化
        num_pos = tf.where(tf.not_equal(num_pos, 0), num_pos, tf.ones_like(num_pos))
        total_loss = tf.reduce_sum(pos_conf_loss) + tf.reduce_sum(neg_conf_loss) + tf.reduce_sum(
            self.alpha * pos_loc_loss)
        total_loss /= tf.reduce_sum(num_pos)

        return total_loss
