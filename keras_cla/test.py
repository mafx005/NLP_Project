# -*- coding: utf-8 -*-
# @Time : 2021/1/8 14:09
# @Author : M
# @FileName: test.py
# @Dec : 

import keras
import numpy as np

file_name = open(r'D:\CODE\chinese_eda\data_in\妨害公务_1W.txt', 'r', encoding='utf8')
text = []
label = []
count = 0
line = file_name.readline()
while line:
    count += 1
    if count == 5:
        break
    text.append(line.split('\t')[1])
    label.append(line.split('\t')[0])
    line = file_name.readline()
label = np.array(label)
one_hot_labels = keras.utils.to_categorical(label, num_classes=6)
print(label)
print(one_hot_labels)
