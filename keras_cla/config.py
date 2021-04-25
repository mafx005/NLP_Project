# -*- coding: utf-8 -*-
# @Time : 2020/11/30 11:50
# @Author : M
# @FileName: config.py
# @Dec :

import os
from classification_model import BiGRU, TextCNN, DPCNN, AttRNN


class Config(object):

    def __init__(self):
        self.word_maxlen = 400
        self.char_maxlen = 400
        self.embed_size = 300
        self.batch_size = 64
        self.epoch = 3
        self.word_num_words = 100000
        self.char_num_words = 5000
        self.gpu_id = "0"
        self.cla_type = 'Binary_class'     # Multiclass:多分类     Binary_class:二分类
        self.cla_nums = 6       # 分类标签数目

        self.model = AttRNN  # 选择模型

        self.checkpoint_dir = 'save_model'  # 需要修改文件名
        self.epoch_info_path = os.path.join(self.checkpoint_dir, 'epoch_info.txt')
        self.result_info_path = os.path.join(self.checkpoint_dir, 'result.txt')
        self.word_vocab_path = 'save_vocab/word_tok.pickle'
        self.char_vocab_path = 'save_vocab/char_tok.pickle'
        self.w2v_path = 'vectors/w2v.model'
        self.glove_path = 'vectors/glove_300d.txt'
        self.embedding_type = 'word2vec'    #   word2vec glove w2v_glove
        # self.char_train_path = 'data/char/mongo_cail_train_char.txt'
        # self.char_valid_path = 'data/char/mongo_cail_valid_char.txt'
        # self.char_test_path = 'data/char/2020_gjcs_data.txt'
        self.train_path = 'data/final_train_word_shuffle.txt'  #
        self.valid_path = 'data/final_valid_word_shuffle.txt'    #
        self.test_path = 'data/2020_gjcs_data_word.txt'  # final_test_word_shuffle  2020_gjcs_data_word_new
        self.badcase_path = 'bad_case/bad_case_data.csv'    #   badcase输出路径
        
        self.word_char = 'word'  # word or char

        self.do_train = True   # 训练设置为   True，预测时设置为    False
        self.write2txt = True   # 训练设置为  True，预测时设置为   False
        self.do_predict = True
        self.write_badcase = False
