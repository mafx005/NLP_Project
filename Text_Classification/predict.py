# -*- coding: utf-8 -*-
# @Time : 2021/8/19 14:19
# @Author : M
# @FileName: predict.py
# @Dec : 

import jieba
import pickle
import torch
import numpy as np
from importlib import import_module

UNK, PAD = '<UNK>', '<PAD>'


def build_data(config, input_text, use_word):
    if use_word:
        tokenizer = lambda x: list(jieba.cut(x))  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    vocab = pickle.load(open(config.vocab_path, 'rb'))
    print(f"Vocab size: {len(vocab)}")
    lin = input_text.strip()
    if not lin:
        raise Exception('input is null')
    content = lin
    words_line = []
    token = tokenizer(content)
    seq_len = len(token)
    if config.pad_size:
        if len(token) < config.pad_size:
            token.extend([PAD] * (config.pad_size - len(token)))
        else:
            token = token[:config.pad_size]
            seq_len = config.pad_size
    # word to id
    for word in token:
        words_line.append(vocab.get(word, vocab.get(UNK)))
    ids = torch.LongTensor([words_line]).to(config.device)
    seq_len = torch.LongTensor(seq_len).to(config.device)
    return ids, seq_len


def predict2list(predictions, sets):
    res = []
    ans = {}
    predictions = torch.nn.Sigmoid()(predictions)
    for x in predictions:
        indices = [i for i in range(len(x)) if x[i] > 0.5]
        if len(indices) > 0:
            ans['accusation'] = ["|".join([sets[index] for index in indices])]
            ans['probability'] = [x[index] for index in indices]
            ans['probability'] = np.array(ans['probability']).tolist()
        else:
            ans['accusation'] = sets[x.tolist().index(max(x.tolist()))]
            ans['probability'] = max(x.tolist())
            ans['probability'] = np.array(ans['probability']).tolist()
        res.append(ans)
        ans = {}
    return res


def predict(config, input_text, use_word, cla_task_name='binary_cla'):
    data = build_data(config, input_text, use_word)
    with torch.no_grad():
        outputs = model(data)
    if cla_task_name == 'mutil_label':
        label = predict2list(outputs, config.class_list)
    else:
        label = torch.argmax(outputs)
    return label


if __name__ == '__main__':
    dataset = 'data'  # 存放数据集的目录
    cla_task_name = 'mutil_label'  # binary_cla(二分类),mutil_class(多分类),mutil_label(多标签分类)
    model_name = 'TextCNN'
    embedding = 'random'
    model = import_module('model_zoo.' + model_name)
    config = import_module('config.' + model_name + '_config').Config(dataset, embedding, model_name=model_name)
    vocab = pickle.load(open(config.vocab_path, 'rb'))
    config.n_vocab = len(vocab)
    model = model.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path, map_location=config.device))
    model.eval()
    test_text = '借条15、陈利民因生意需要向黄春元借人民币40，000元，大写肆万元整，月利息按8％计算。'
    use_word = True
    print(predict(config, test_text, use_word=use_word, cla_task_name=cla_task_name))
