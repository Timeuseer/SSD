#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Create Time : 2022/3/10 9:27 
@Author : Gabriel
@File : vgg.py 
@Project:SSD
@About : 搭建vgg网络框架
'''

from tensorflow.keras import layers


def vgg16(input_tensor):
    '''
    backbone 构建
    :param input_tensor: 输入
    :return:
    '''

    net = {}

    '''
    Block1
    [300,300,3] -> [150,150,64]
    '''
    net['input'] = input_tensor
    net['conv1_1'] = layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
                                   name='conv1_1')(net['input'])
    net['conv1_2'] = layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
                                   name='conv1_2')(net['conv1_1'])
    net['pool1'] = layers.MaxPool2D((2, 2), strides=(2, 2), padding='same',
                                    name='pool1')(net['conv1_2'])

    '''
    Block2
    [150,150,64] -> [75,75,128]
    '''
    net['conv2_1'] = layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',
                                   name='conv2_1')(net['pool1'])
    net['conv2_2'] = layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',
                                   name='conv2_2')(net['conv2_1'])
    net['pool2'] = layers.MaxPool2D((2, 2), strides=(2, 2), padding='same',
                                    name='pool2')(net['conv2_2'])

    '''
    Block3
    [75,75,128] -> [38,38,256]
    '''
    net['conv3_1'] = layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same',
                                   name='conv3_1')(net['pool2'])
    net['conv3_2'] = layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same',
                                   name='conv3_2')(net['conv3_1'])
    net['conv3_3'] = layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same',
                                   name='conv3_3')(net['conv3_2'])
    net['pool3'] = layers.MaxPool2D((2, 2), strides=(2, 2), padding='same',
                                    name='pool3')(net['conv3_3'])

    '''
    Block4
    [38,38,256] -> [19,19,512]
    '''
    net['conv4_1'] = layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same',
                                   name='conv4_1')(net['pool3'])
    net['conv4_2'] = layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same',
                                   name='conv4_2')(net['conv4_1'])
    net['conv4_3'] = layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same',
                                   name='conv4_3')(net['conv4_2'])
    net['pool4'] = layers.MaxPool2D((2, 2), strides=(2, 2), padding='same',
                                    name='pool4')(net['conv4_3'])

    '''
    Block5
    [19,19,512] -> [19,19,512]
    '''
    net['conv5_1'] = layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same',
                                   name='conv5_1')(net['pool4'])
    net['conv5_2'] = layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same',
                                   name='conv5_2')(net['conv5_1'])
    net['conv5_3'] = layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same',
                                   name='conv5_3')(net['conv5_2'])
    net['pool5'] = layers.MaxPool2D((3, 3), strides=(1, 1), padding='same',
                                    name='pool5')(net['conv5_3'])

    '''
    FC6
    [19,19,512] -> [19,19,1024]
    使用空洞卷积
    '''
    net['fc6'] = layers.Conv2D(1024, kernel_size=(3, 3), dilation_rate=(6, 6),
                               activation='relu', padding='same', name='fc6')(net['pool5'])

    '''
    FC7
    [19,19,1024] -> [19,19,1024]
    '''
    net['fc7'] = layers.Conv2D(1024, kernel_size=(1, 1), activation='relu', padding='same',
                               name='fc7')(net['fc6'])

    '''
    Block6
    [19,19,1024] -> [10,10,512]
    '''
    net['conv6_1'] = layers.Conv2D(256, kernel_size=(1, 1), activation='relu', padding='same',
                                   name='conv6_1')(net['fc7'])
    net['conv6_padding'] = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(net['conv6_1'])
    net['conv6_2'] = layers.Conv2D(512, kernel_size=(3, 3), strides=(2, 2), activation='relu',
                                   name='conv6_2')(net['conv6_padding'])

    '''
    Block7
    [10,10,512] -> [5,5,256]
    '''
    net['conv7_1'] = layers.Conv2D(128, kernel_size=(1, 1), activation='relu', padding='same',
                                   name='conv7_1')(net['conv6_2'])
    net['conv7_padding'] = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(net['conv7_1'])
    net['conv7_2'] = layers.Conv2D(256, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='valid',
                                   name='conv7_2')(net['conv7_padding'])

    '''
    Block8
    [5,5,256] -> [3,3,256]
    '''
    net['conv8_1'] = layers.Conv2D(128, kernel_size=(1, 1), activation='relu', padding='same',
                                   name='conv8_1')(net['conv7_2'])
    net['conv8_2'] = layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                   padding='valid', name='conv8_2')(net['conv8_1'])

    '''
    Block9
    [3,3,256] -> [1,1,256]
    '''
    net['conv9_1'] = layers.Conv2D(128, kernel_size=(1, 1), activation='relu', padding='same',
                                   name='conv9_1')(net['conv8_2'])
    net['conv9_2'] = layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                   padding='valid', name='conv9_2')(net['conv9_1'])

    return net
