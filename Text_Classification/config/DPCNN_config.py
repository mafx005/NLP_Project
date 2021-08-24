# -*- coding: utf-8 -*-
# @Time : 2021/8/24 14:39
# @Author : M
# @FileName: DPCNN_config.py
# @Dec : 

from config.base_config import BaseConfig


class Config(BaseConfig):
    def __init__(self, dataset, embedding, model_name):
        super(Config, self).__init__(dataset, embedding, model_name)
        self.num_filters = 256  # 卷积核数量(channels数)
