#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Create Time : 2022/3/10 9:27 
@Author : Gabriel
@File : ssd.py 
@Project: SSD
@About : 搭建SSD模型
'''

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import models
from nets.vgg import vgg16
from nets.custom_layer import Normalize


def ssd(input_shape, num_classes=21):
    # 模型常用输入大小为[300,300,3]
    input_shape = layers.Input(shape=input_shape)

    # 生成backbone
    net = vgg16(input_shape)

    '''
    对提取到的主干特征进行处理
    '''

    # 对conv4_3进行l2标准化处理,shape = [38,38,512]
    net['conv4_3_norm'] = Normalize(20, name='conv4_3_norm')(net['conv4_3'])
    num_priors = 4
    # 对预测框进行处理
    # num_priors是每个网格点的先验框数量,4表示x,y,h,w的调整
    net['conv4_3_norm_mbox_loc'] = layers.Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same',
                                                 name='conv4_3_norm_mbox_loc')(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_loc_flat'] = layers.Flatten(name='conv4_3_norm_mbox_loc_flat')(net['conv4_3_norm_mbox_loc'])
    # 分类置信度
    net['conv4_3_norm_mbox_conf'] = layers.Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same',
                                                  name='conv4_3_norm_mbox_conf')(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_conf_flat'] = layers.Flatten(name='conv4_3_norm_mbox_conf_flat')(
        net['conv4_3_norm_mbox_conf'])

    # 对fc7层进行处理,shape=[19,19,1024]
    num_priors = 6
    # 对预测框进行处理
    # num_priors是每个网格点的先验框数量,4表示x,y,h,w的调整
    net['fc7_mbox_loc'] = layers.Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same',
                                        name='fc7_mbox_loc')(net['fc7'])
    net['fc7_mbox_loc_flat'] = layers.Flatten(name='fc7_mbox_loc_flat')(net['fc7_mbox_loc'])
    # 分类置信度
    net['fc7_mbox_conf'] = layers.Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same',
                                         name='fc7_mbox_conf')(net['fc7'])
    net['fc7_mbox_conf_flat'] = layers.Flatten(name='fc7_mbox_conf_flat')(
        net['fc7_mbox_conf'])

    # 对conv6_2进行处理,shape=[10,10,512]
    num_priors = 6
    # 对预测框进行处理
    # num_priors是每个网格点的先验框数量,4表示x,y,h,w的调整
    net['conv6_2_mbox_loc'] = layers.Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same',
                                            name='conv6_2_mbox_loc')(net['conv6_2'])
    net['conv6_2_mbox_loc_flat'] = layers.Flatten(name='conv6_2_mbox_loc_flat')(net['conv6_2_mbox_loc'])
    # 分类置信度
    net['conv6_2_mbox_conf'] = layers.Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same',
                                             name='conv6_2_mbox_conf')(net['conv6_2'])
    net['conv6_2_mbox_conf_flat'] = layers.Flatten(name='conv6_2_mbox_conf_flat')(
        net['conv6_2_mbox_conf'])

    # 对conv7_2进行处理,shape=[5,5,256]
    num_priors = 6
    # 对预测框进行处理
    # num_priors是每个网格点的先验框数量,4表示x,y,h,w的调整
    net['conv7_2_mbox_loc'] = layers.Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same',
                                            name='conv7_2_mbox_loc')(net['conv7_2'])
    net['conv7_2_mbox_loc_flat'] = layers.Flatten(name='conv7_2_mbox_loc_flat')(net['conv7_2_mbox_loc'])
    # 分类置信度
    net['conv7_2_mbox_conf'] = layers.Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same',
                                             name='conv7_2_mbox_conf')(net['conv7_2'])
    net['conv7_2_mbox_conf_flat'] = layers.Flatten(name='conv7_2_mbox_conf_flat')(
        net['conv7_2_mbox_conf'])

    # 对conv8_2进行处理,shape=[3,3,256]
    num_priors = 4
    # 对预测框进行处理
    # num_priors是每个网格点的先验框数量,4表示x,y,h,w的调整
    net['conv8_2_mbox_loc'] = layers.Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same',
                                            name='conv8_2_mbox_loc')(net['conv8_2'])
    net['conv8_2_mbox_loc_flat'] = layers.Flatten(name='conv8_2_mbox_loc_flat')(net['conv8_2_mbox_loc'])
    # 分类置信度
    net['conv8_2_mbox_conf'] = layers.Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same',
                                             name='conv8_2_mbox_conf')(net['conv8_2'])
    net['conv8_2_mbox_conf_flat'] = layers.Flatten(name='conv8_2_mbox_conf_flat')(
        net['conv8_2_mbox_conf'])

    # 对conv9_2进行处理,shape=[1,1,256]
    num_priors = 4
    # 对预测框进行处理
    # num_priors是每个网格点的先验框数量,4表示x,y,h,w的调整
    net['conv9_2_mbox_loc'] = layers.Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same',
                                            name='conv9_2_mbox_loc')(net['conv9_2'])
    net['conv9_2_mbox_loc_flat'] = layers.Flatten(name='conv9_2_mbox_loc_flat')(net['conv9_2_mbox_loc'])
    # 分类置信度
    net['conv9_2_mbox_conf'] = layers.Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same',
                                             name='conv9_2_mbox_conf')(net['conv9_2'])
    net['conv9_2_mbox_conf_flat'] = layers.Flatten(name='conv9_2_mbox_conf_flat')(
        net['conv9_2_mbox_conf'])

    # 对上面得到的结果进行堆叠
    net['mbox_loc'] = layers.Concatenate(axis=1, name='mbox_loc')([net['conv4_3_norm_mbox_loc_flat'],
                                                                   net['fc7_mbox_loc_flat'],
                                                                   net['conv6_2_mbox_loc_flat'],
                                                                   net['conv7_2_mbox_loc_flat'],
                                                                   net['conv8_2_mbox_loc_flat'],
                                                                   net['conv9_2_mbox_loc_flat']])

    net['mbox_conf'] = layers.Concatenate(axis=1, name='mbox_conf')([net['conv4_3_norm_mbox_conf_flat'],
                                                                     net['fc7_mbox_conf_flat'],
                                                                     net['conv6_2_mbox_conf_flat'],
                                                                     net['conv7_2_mbox_conf_flat'],
                                                                     net['conv8_2_mbox_conf_flat'],
                                                                     net['conv9_2_mbox_conf_flat']])

    # [8732,4]
    net['mbox_loc'] = layers.Reshape((-1, 4), name='mbox_loc_final')(net['mbox_loc'])
    # [8732,21]
    net['mbox_conf'] = layers.Reshape((-1, num_classes), name='mbox_conf_logics')(net['mbox_conf'])
    net['mbox_conf'] = layers.Activation('softmax', name='mbox_conf_final')(net['mbox_conf'])
    # [8732,25]
    net['predications'] = layers.Concatenate(axis=-1, name='predications')([net['mbox_loc'], net['mbox_conf']])

    model = models.Model(net['input'], net['predications'])

    return model
