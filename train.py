#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Create Time : 2022/3/13 14:49 
@Author : Gabriel
@File : train.py 
@Project:SSD
@About : 
'''

import yaml
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from nets.ssd import ssd
from nets.loss import MultiboxLoss
from utils.anchors import get_anchors
from utils.callbacks import (ExponentDecayScheduler, LossHistory,
                             ModelCheckpoint)
from utils.load_data import SSDDatasets
from utils.utils import get_classes

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
)

if __name__ == '__main__':
    '''
    加载配置文件
    '''
    config = yaml.load(open('config/train_config.yaml', 'r', encoding='utf-8'))

    '''
    获取classes和anchor
    '''
    class_names, num_classes = get_classes(classes_path=config['classes_path'])
    num_classes += 1
    anchors = get_anchors(input_shape=config['input_shape'], anchors_size=config['anchors_size'])

    model = ssd((config['input_shape'][0], config['input_shape'][1], 3), num_classes)

    if config['model_path'] != '':
        print(f'load weights {config["model_path"]}...')
        model.load_weights(config["model_path"], by_name=True, skip_mismatch=True)

    '''
    训练参数的设置
    logging:        表示tensorboard的保存地址
    checkpoint:     用于设置权值保存的细节，period用于修改多少epoch保存一次
    reduce_lr:      用于设置学习率下降的方式
    early_stopping: 用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    '''
    logging = TensorBoard(log_dir='logs/')
    checkpoint = ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    reduce_lr = ExponentDecayScheduler(decay_rate=0.92, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    loss_history = LossHistory('logs/')

    multiloss = MultiboxLoss(num_classes, neg_pos_ratio=3.0).compute_loss

    '''
    加载数据集
    '''
    with open(config['train_annotation_path']) as f:
        train_lines = f.readlines()
    with open(config['val_annotation_path']) as f:
        val_lines = f.readlines()

    num_train = len(train_lines)
    num_val = len(val_lines)

    if config['Freeze_Train']:
        freeze_layers = 17
        for i in range(freeze_layers): model.layers[i].trainable = False
        print(f'Freeze the first {freeze_layers} layers of total {len(model.layers)} layers.')

    '''
    主干特征提取网络特征通用，冻结训练可以加快速度，也可以防止训练初期破坏权值。
    Init_Epoch      为起始世代
    Freeze_Epoch    为冻结训练的世代
    Unfreeze_Epoch  总训练世代
    '''
    batch_size = config['Freeze_batch_size']
    lr = config['Freeze_lr']
    start_epoch = config['Init_Epoch']
    end_epoch = config['Freeze_Epoch']

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError('数据集过小，无法进行训练，请扩充数据集...')

    train_dataloader = SSDDatasets(train_lines, config['input_shape'], anchors, batch_size, num_classes, train=True)
    val_dataloader = SSDDatasets(val_lines, config['input_shape'], anchors, batch_size, num_classes, train=False)

    print(f'Train on {num_train} samples, val on {num_val} samples, with batch size {batch_size}...')
    model.compile(optimizer=Adam(lr=lr), loss=MultiboxLoss(num_classes, neg_pos_ratio=3.0).compute_loss)
    model.fit_generator(
        generator=train_dataloader,
        steps_per_epoch=epoch_step,
        validation_data=val_dataloader,
        validation_steps=epoch_step_val,
        epochs=end_epoch,
        initial_epoch=start_epoch,
        use_multiprocessing=True if config['num_workers'] != 0 else False,
        workers=config['num_workers'],
        callbacks=[logging, checkpoint, reduce_lr, early_stopping, loss_history]
    )

    if config['Freeze_Train']:
        for i in range(freeze_layers):
            model.layers[i].trainable = True

    batch_size = config['UnFreeze_batch_size']
    lr = config['UnFreeze_lr']
    start_epoch = config['Freeze_Epoch']
    end_epoch = config['UnFreeze_Epoch']

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError('数据集过小，无法进行训练，请扩充数据集...')

    train_dataloader = SSDDatasets(train_lines, config['input_shape'], anchors, batch_size, num_classes, train=True)
    val_dataloader = SSDDatasets(val_lines, config['input_shape'], anchors, batch_size, num_classes, train=False)

    print(f'Train on {num_train} samples, val on {num_val} samples, with batch size {batch_size}...')
    model.compile(optimizer=Adam(lr=lr), loss=MultiboxLoss(num_classes, neg_pos_ratio=3.0).compute_loss)
    model.fit_generator(
        generator=train_dataloader,
        steps_per_epoch=epoch_step,
        validation_data=val_dataloader,
        validation_steps=epoch_step_val,
        epochs=end_epoch,
        initial_epoch=start_epoch,
        use_multiprocessing=True if config['num_workers'] != 0 else False,
        workers=config['num_workers'],
        callbacks=[logging, checkpoint, reduce_lr, early_stopping, loss_history]
    )
