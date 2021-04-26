# -*- coding: utf-8 -*-
# @Time : 2020/11/30 16:11
# @Author : M
# @FileName: My_Callback.py
# @Dec : 

import tensorflow as tf
import keras
from Config.config import Config

config = Config()


class Mylosscallback(keras.callbacks.Callback):
    def __init__(self, log_dir):
        super(keras.callbacks.Callback, self).__init__()
        super().__init__()
        self.writer = tf.summary.FileWriter(log_dir)
        self.num = 0
        self.epoch = 0
        self.acc = 0

    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.num = self.num + 1
        self.losses = logs.get('loss')
        self.accuracy = logs.get('acc')
        self.val_loss = logs.get('val_loss')
        self.val_acc = logs.get('val_acc')
        # print('debug success!!!')
        summary = tf.compat.v1.Summary()  # tf.compat.v1.Summary()
        summary.value.add(tag='losses', simple_value=self.losses)
        summary.value.add(tag='accuracy', simple_value=self.accuracy)
        summary.value.add(tag='val_loss', simple_value=self.val_loss)
        summary.value.add(tag='val_acc', simple_value=self.val_acc)
        self.writer.add_summary(summary, self.num)
        self.writer.flush()

    def on_epoch_end(self, batch, logs={}):
        self.epoch = self.epoch + 1
        epoch_path = open(config.epoch_info_path, 'a', encoding='utf8')
        epoch_path.write("Epoch" + str(self.epoch) + "\t")
        epoch_path.write("Loss:" + str(logs.get('loss')))
        epoch_path.write("\t" + "ACC:" + str(logs.get('acc')))
        epoch_path.write('\n')
        epoch_path.close()
