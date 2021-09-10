# -*- coding: utf-8 -*-
# @Time : 2021/8/24 17:42
# @Author : M
# @FileName: bert_config.py
# @Dec : 

from config.base_config import BaseConfig
from transformers import BertTokenizer


class Config(BaseConfig):
    def __init__(self, dataset, embedding, model_name):
        super(Config, self).__init__(dataset, embedding, model_name)
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = './bert_pretrain/ms'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
