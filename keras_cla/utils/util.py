# -*- coding: utf-8 -*-

import re
import numpy as np
import os
import keras
import pickle
import jieba

stop_words = ['经鉴定', '抓获', '供述', '再查明', '庭审举证', '鉴定书', '以上事实', '民事诉讼原告人', '指控的事实', '陈述', '民事赔偿',
              '以上指控', '代理意见', '判决书', '检验报告', '指控的犯罪事实', '调取证据通知书', '述称', '综合分析', '综上', '光盘', '经统计', '审理过程中', '上述事实',
              '经侦查', '案件侦查', '上述指控', '案发后', '鉴定', '经查', '认定', '登记表', '检验', '证言', '照片', '笔录', '证据',
              '书证', '资料', '辩护意见', '证明', '勘验']


def data_preprocess(text_string):
    text_list = text_string.split('\n')
    print('total length', len(text_list))
    ajjbqk_start = 0
    ajjbqk_end = len(text_list)
    res_list = []
    for i in range(ajjbqk_start, len(text_list)):
        if re.search(r'审理查明(.*?)一致.*?(，|。|：|,|:)', text_list[i]):
            ajjbqk_start = ajjbqk_start
            continue
        elif '审理查明' in text_list[i] or '依法查明' in text_list[i] or '经依法审查' in text_list[i]:
            ajjbqk_start = i
            break
    if ajjbqk_start == 0:
        for i in range(ajjbqk_start, len(text_list)):
            if '检察院指控' in text_list[i] or '公诉机关指控' in text_list[i] or '指控：' in text_list[i] or '指控:' in text_list[i] or '指控事实：' in text_list[i]:
                ajjbqk_start = i
                break
    if ajjbqk_start == 0:
        for i in range(ajjbqk_start, len(text_list)):
            if '审理终结' in text_list[i] or '侦查终结' in text_list[i]:
                ajjbqk_start = i + 1
                break
    for i in range(ajjbqk_start, len(text_list)):
        if re.search(r'审理查明(.*?)一致.*?(，|。|：|,|:)', text_list[i]):
            ajjbqk_end = i
            break
        elif '本院认为' in text_list[i]:
            ajjbqk_end = i
            break
    else:
        ajjbqk_end = ajjbqk_end
    ajjbqk = text_list[ajjbqk_start:ajjbqk_end + 1] if ajjbqk_start == ajjbqk_end else text_list[
                                                                                       ajjbqk_start:ajjbqk_end]
    print('中间', len(ajjbqk))
    for i in ajjbqk:
        first_sent = i.split('，')[0]
        for j in stop_words:
            if j in first_sent:
                break
        else:
            res_list.append(i)
    print('最后', len(res_list))
    return res_list


def merge_fact_paragraphs(text_list):
    res_dict = dict(事实1='')
    count = 0
    for i in text_list:
        juhao_string = i.split('。')[0]
        first_douhao = '，'.join(juhao_string.split('，')[:2])
        if '审理查明' in first_douhao or '指控' in first_douhao:
            count += 1
            res_dict['事实' + str(count)] = i
        elif '年' in first_douhao or '月' in first_douhao or '日' in first_douhao:
            count += 1
            res_dict['事实' + str(count)] = i
        else:
            if count == 0:
                count += 1
                res_dict['事实' + str(count)] = res_dict['事实' + str(count)] + i
            else:
                res_dict['事实' + str(count)] = res_dict['事实' + str(count)] + '\n' + i
    # print(res_dict)
    return res_dict


def load_data(file_path, class_file, cla_type, cla_nums, word_char='word'):
    label_set = creat_label_set(class_file)
    if word_char == 'word':
        file_name = open(file_path, 'r', encoding='utf8')
    else:
        file_name = open(file_path, 'r', encoding='utf8')
    text = []
    label = []
    for t in file_name:
        try:
            assert len(t.split('\t')) == 2
        except:
            print('data error line is ', t)
            continue
        if word_char == 'word':
            text.append(cut_data(t.split('\t')[1]))
        else:
            text.append([x for x in t.split('\t')[1]])
        try:
            label.append(int((t.split('\t')[0]).replace(' ', '')))
        except:
            label_temp = t.split('\t')[0].replace(' ', '')
            label_temp = label_temp.split(',')
            label_zero = creat_label(label_temp, label_set)
            label.append(label_zero)
    label = np.array(label)
    assert cla_type in ['Multiclass', 'Binary_class', 'Mutillabel']
    if cla_type == 'Multiclass':
        label = keras.utils.to_categorical(label, num_classes=cla_nums)
    elif cla_type in ['Binary_class', 'Mutillabel']:
        label = label
    return text, label


def load_vocab(vocab_path, total_text, word_char='word', num_words=400):
    if word_char == 'word':
        if not os.path.exists(vocab_path):
            token = keras.preprocessing.text.Tokenizer(num_words=num_words)
            token.fit_on_texts(total_text)
            with open(vocab_path, 'wb')as f:
                pickle.dump(token, f, protocol=pickle.HIGHEST_PROTOCOL)
                print('++++++++++word词表保存完成++++++++++')
        else:
            with open(vocab_path, 'rb')as f:
                token = pickle.load(f)
                print('++++++++++word词表加载完成++++++++++')
    else:
        if not os.path.exists(vocab_path):
            token = keras.preprocessing.text.Tokenizer(num_words=num_words)
            token.fit_on_texts(total_text)
            with open(vocab_path, 'wb')as f:
                pickle.dump(token, f, protocol=pickle.HIGHEST_PROTOCOL)
                print('++++++++++char词表保存完成++++++++++')
        else:
            with open(vocab_path, 'rb')as f:
                token = pickle.load(f)
                print('++++++++++char词表加载完成++++++++++')
    return token


def cut_data(text):
    text_cut = ' '.join(jieba.cut(text))
    return text_cut


def creat_label_set(label_file_path):
    label_set = [x.strip() for x in open(label_file_path, encoding='utf8').readlines()]
    return label_set


def creat_label(label, label_set):
    label_str = [str(i) for i in label]
    label_zero = np.zeros(len(label_set))
    label_zero[np.in1d(label_set, label_str)] = 1
    return label_zero


def predict2half(predictions):
    return np.where(predictions > 0.5, 1.0, 0.0)


def predict2top(predictions):
    one_hots = []
    for prediction in predictions:
        one_hot = np.where(prediction == prediction.max(), 1.0, 0.0)
        one_hots.append(one_hot)
    return np.array(one_hots)


def predict2both(predictions):
    one_hots = []
    for prediction in predictions:
        one_hot = np.where(prediction > 0.75, 1.0, 0.0)
        if one_hot.sum() == 0:
            one_hot = np.where(prediction == prediction.max(), 1.0, 0.0)
        one_hots.append(one_hot)
    return np.array(one_hots)


def predict2tag(predictions, label_file_path):
    sets = creat_label_set(label_file_path)
    m = []
    for x in predictions:
        x_return = []
        for i in range(len(x)):
            if x[i] > 0.75:
                x_return.append(sets[i])
        m.append(x_return)
    return m


# if __name__ == '__main__':
#     label_file = '../data/accu.txt'
#     data_file = '../data/data_test.txt'
#     label_set = creat_label_set(label_file)
#     print(label_set)
#     with open(data_file, 'r',encoding='utf8')as f:
#         lines = f.readlines()
#         for i in range(len(lines)):
#             label = lines[i].split('\t')[0]
#             label = label.split(',')
#             print(label)
#             label_zero = creat_label(label,label_set)
#             print(label_zero)
