# -*- coding: utf-8 -*-
# @Time : 2021/8/24 9:30
# @Author : M
# @FileName: TextRNN_Att_config.py
# @Dec : 

from config.base_config import BaseConfig


class Config(BaseConfig):
    def __init__(self, dataset, embedding, model_name):
        super(Config, self).__init__(dataset, embedding, model_name)
        self.hidden_size = 128  # lstm隐藏层
        self.num_layers = 2  # lstm层数
        self.hidden_size2 = 64
