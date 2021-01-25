# -*- coding: utf-8 -*-
# @Time : 2021/1/23 13:55
# @Author : M
# @FileName: 共现矩阵.py
# @Dec : 

import jieba
import numpy as np

sent_data = ['自然语言处理是一个领域', '共现矩阵是自然语言处理的一个知识点']
# sent_data = ['I like deep learning .', ' I like NLP .', 'I enjoy flying .']


def build_vocab(data):
    vocab_list = []
    for i in range(len(data)):
        text = jieba.cut(data[i])
        text = ' '.join(text)
        # text = data[i]
        vocab_list += text.split()
    vocab_list = list(set(vocab_list))
    return vocab_list


def init_matrix(set_vocab):
    vocab_matrix = [['' for j in range(len(set_vocab) + 1)] for i in range(len(set_vocab) + 1)]
    vocab_matrix[0][1:] = np.array(set_vocab)
    vocab_matrix = list(map(list, zip(*vocab_matrix)))
    vocab_matrix[0][1:] = np.array(set_vocab)
    return vocab_matrix


def build_matrix(matrix, text):
    for row in range(1, len(matrix)):
        for col in range(1, len(matrix)):
            if matrix[0][row] == matrix[col][0]:
                matrix[col][row] = str(0)
            else:
                count = 0
                for word in text:
                    if matrix[0][row] in word and matrix[col][0] in word:
                    # if matrix[0][row] + ' ' + matrix[col][0] in text[word] or matrix[col][0] + ' ' + matrix[0][row]in text[word]:
                        count += 1
                    else:
                        continue
                matrix[col][row] = str(count)
    print(matrix)
    return matrix


vocab = build_vocab(sent_data)
# print(vocab)
vocab_matrix = init_matrix(vocab)
# print(vocab_matrix)
word_data = []
for i in sent_data:
    word_data.append(' '.join(jieba.cut(i)).split())
build_matrix(vocab_matrix, word_data)
