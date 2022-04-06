#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Create Time : 2022/3/10 10:03 
@Author : Gabriel
@File : custom_layer.py 
@Project: SSD
@About :
'''

import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras import layers


class Normalize(layers.Layer):
    def __init__(self, scale, **kwargs):
        self.axis = 3
        self.scale = scale
        super(Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [layers.InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)
        init_gamma = self.scale * np.ones(shape)
        self.gamma = K.variable(init_gamma, name=f'{self.name}_gamma')

    def call(self, inputs, mask=None):
        output = K.l2_normalize(inputs, self.axis)
        output *= self.gamma

        return output
