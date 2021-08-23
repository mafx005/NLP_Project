# -*- coding: utf-8 -*-
# @Time : 2021/8/19 14:04
# @Author : M
# @FileName: CNN_config.py
# @Dec :

from Text_Classification.config.base_config import BaseConfig


class Config(BaseConfig):
    def __init__(self, dataset, embedding, model_name):
        super(Config, self).__init__(dataset, embedding, model_name)
        self.filter_sizes = [1, 2, 3, 4, 5]  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels数)
