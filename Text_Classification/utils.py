# -*- coding: utf-8 -*-
# @Time : 2021/8/19 14:11
# @Author : M
# @FileName: utils.py
# @Dec :

import os
import pickle
import torch
import time
import jieba
import numpy as np
from tqdm import tqdm
from datetime import timedelta
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

MAX_VOCAB_SIZE = 100000
UNK, PAD = '<UNK>', '<PAD>'


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    # print(file_path)
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[1]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config, use_word, cla_task_name='binary_cla'):
    if use_word:
        tokenizer = lambda x: list(jieba.cut(x))  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(config.vocab_path):
        vocab = pickle.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pickle.dump(vocab, open(config.vocab_path, 'wb'))
    print("Vocab size:", len(vocab))

    def load_dataset(path, pad_size=32):
        contents = []
        couont = 0
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                couont += 1
                if couont == 10:
                    break
                lin = line.strip()
                if not lin:
                    continue
                label, content = lin.split('\t')
                if cla_task_name == 'mutil_label':
                    label = mutil_label2onehot(label, config.class_list)
                else:
                    label = int(label)
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, label, seq_len))
        return contents  # [([...], 0), ([...], 1), ...]
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.valid_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        if len(batches) < batch_size:
            batch_size = len(batches)
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


class BertData(Dataset):
    def __init__(self, config, file_name, cla_task_name='binary_cla'):
        self.data = self.build_bert_dataset(config, file_name, cla_task_name)

    @staticmethod
    def build_bert_dataset(config, file_name, cla_task_name):
        datas = []
        with open(file_name, 'r', encoding='utf8')as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                label, content = lin.split('\t')
                if cla_task_name == 'mutil_label':
                    label = mutil_label2onehot(label, config.class_list)
                else:
                    label = int(label)
                datas.append((content, label))
        return datas

    def __getitem__(self, idx):
        text = self.data[idx][0]
        label = self.data[idx][1]
        return text, label

    def __len__(self):
        return len(self.data)


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def acc_score(true_labels, pred_labels):
    return accuracy_score(true_labels, pred_labels)


def predict2both(predictions):
    one_hots = []
    predictions = torch.nn.Sigmoid()(predictions)
    predictions = predictions.cpu()
    for prediction in predictions:
        one_hot = np.where(prediction > 0.5, 1.0, 0.0)
        if one_hot.sum() == 0:
            one_hot = np.where(prediction == prediction.max(), 1.0, 0.0)
        one_hots.append(one_hot)
    return np.array(one_hots)


def mutil_label_acc_score(true_labels, predictions):
    pred_labels = predict2both(predictions)
    if torch.is_tensor(true_labels):
        true_labels = true_labels.numpy()
    acc = [(true_labels[i] == pred_labels[i]).min() for i in range(len(pred_labels))]
    return sum(acc) / len(acc)


def mutil_label_f1_score(true_labels, predictions):
    pred_labels = predict2both(predictions)
    f1_micro = f1_score(true_labels, pred_labels, average='micro')
    f1_macro = f1_score(true_labels, pred_labels, average='macro')
    return f1_micro, f1_macro


def mutil_label2onehot(label, labels_list):
    num_classes = len(labels_list)
    onehot_labels = np.zeros(num_classes, dtype=int)
    label_list = list(map(lambda x: x, label.split(',')))
    for j in label_list:
        index = labels_list.index(j)
        onehot_labels[index] = 1
    return onehot_labels


class FGM():
    """
    对抗训练，对embedding添加扰动
    """

    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        # print(emb_name)
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        # print(emb_name)
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


if __name__ == '__main__':
    label_list = [x.strip() for x in open('./data/sample_label.txt', encoding='utf-8').readlines()]
    text = '1,2\t多标签数据格式，类别间用逗号隔开'
    label, text = text.split('\t')
    print(mutil_label2onehot(label, label_list))
    pass
