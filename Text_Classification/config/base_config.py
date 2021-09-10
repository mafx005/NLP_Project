# -*- coding: utf-8 -*-
# @Time : 2021/8/20 10:08
# @Author : M
# @FileName: base_config.py
# @Dec : 

import torch
import numpy as np


class BaseConfig(object):
    def __init__(self, dataset, embedding, model_name):
        self.model_name = model_name
        self.train_path = dataset + '/train.txt'  # 训练集
        self.valid_path = dataset + '/valid.txt'  # 验证集
        self.test_path = dataset + '/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/labels.txt', encoding='utf-8').readlines()]  # 类别名单
        self.vocab_path = dataset + '/vocab.pkl'  # 词表
        self.save_path = dataset + '/saved_model/' + self.model_name + '.pth'  # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + embedding)["embeddings"].astype('float32')) \
            if embedding != 'random' else None  # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dropout = 0.5
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab = 10000  # 词表大小，在运行时赋值
        self.num_epochs = 3
        self.batch_size = 64
        self.pad_size = 128
        self.learning_rate = 1e-3
        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300  # 字向量维度
