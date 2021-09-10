# -*- coding: utf-8 -*-
# @Time : 2021/9/10 9:43
# @Author : M
# @FileName: aeda.py
# @Dec : 

import jieba
import random

"""
paper title:    AEDA：AnEasier Data Augmentation Technique for Text Classification
paper url:  https://arxiv.org/pdf/2108.13230.pdf
paper code: https://github.com/akkarimi/aeda_nlp
"""


PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']
PUNC_RATIO = 0.3


def insert_punctuation_marks(sentence, punc_ratio=PUNC_RATIO):
    words = sentence.split(' ')
    new_line = []
    q = random.randint(1, int(punc_ratio * len(words) + 1))
    qs = random.sample(range(0, len(words)), q)

    for j, word in enumerate(words):
        if j in qs:
            new_line.append(PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS) - 1)])
            new_line.append(word)
        else:
            new_line.append(word)
    new_line = ' '.join(new_line)
    return new_line


if __name__ == '__main__':
    sentence = '这是一个测试句子。'
    word = ' '.join(jieba.lcut(sentence))
    print(insert_punctuation_marks(word))
