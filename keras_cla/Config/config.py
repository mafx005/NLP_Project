# -*- coding: utf-8 -*-
# @Time : 2020/11/30 11:50
# @Author : M
# @FileName: config.py
# @Dec :

import os
from model_zoo.bigru import BiGRU
from model_zoo.DPCNN import DPCNN
from model_zoo.rnn_att import AttRNN
from model_zoo.textcnn import TextCNN


class Config(object):

    def __init__(self):
        self.word_maxlen = 400  # 词的最大长度
        self.char_maxlen = 400  # 字的最大长度
        self.embed_size = 300  # 词向量维度
        self.batch_size = 64
        self.epoch = 3
        self.word_num_words = 100000  # 词表的最大数量(word level)
        self.char_num_words = 5000  # 词表的最大数量 (char level)
        self.gpu_id = "0"  # 指定GPU的ID
        self.cla_type = 'Binary_class'  # Multiclass:多分类  Binary_class:二分类 Mutillabel:多标签
        self.cla_nums = 6  # 分类标签数目

        self.model = AttRNN  # 模型

        self.base_file = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.checkpoint_dir = os.path.join(self.base_file, 'save_model')  # 需要修改文件名
        self.epoch_info_path = os.path.join(self.checkpoint_dir, 'epoch_info.txt')
        self.result_info_path = os.path.join(self.checkpoint_dir, 'result.txt')  # 最终结果文件
        self.word_vocab_path = os.path.join(self.base_file, r'data\save_vocab\word_tok.pickle')  # 词表文件
        self.char_vocab_path = os.path.join(self.base_file, r'data\save_vocab\char_tok.pickle')  # 词表文件
        self.w2v_path = os.path.join(self.base_file, r'vectors\w2v.model')  # word2vec词向量文件
        self.glove_path = os.path.join(self.base_file, r'vectors\glove_300d.txt')  # glove词向量文件
        self.embedding_type = 'word2vec'  # word2vec glove w2v_glove
        self.train_path = os.path.join(self.base_file, r'data\train_sample.txt')  # 训练集文件
        self.valid_path = os.path.join(self.base_file, r'data\valid_sample.txt')  # 验证集文件
        self.test_path = os.path.join(self.base_file, r'data\test_sample.txt')  # 测试集文件
        self.badcase_path = os.path.join(self.base_file, r'data\bad_case\bad_case_data.csv')  # badcase输出路径

        self.word_char = 'word'  # word or char

        self.do_train = True  # 训练设置为True，预测时设置为False
        self.write2txt = True  # 训练设置为True，预测时设置为False
        self.do_predict = True
        self.write_badcase = True  # 是否将预测错误的数据写入文件
