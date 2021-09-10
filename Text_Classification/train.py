# -*- coding: utf-8 -*-
# @Time : 2021/8/19 14:19
# @Author : M
# @FileName: train.py
# @Dec : 

import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from importlib import import_module
from tensorboardX import SummaryWriter
from transformers import BertTokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import build_dataset, build_iterator, get_time_dif, BertData
from utils import acc_score, classification_report, confusion_matrix, mutil_label_acc_score
from utils import mutil_label_f1_score


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=42):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样


def train(config, model, train_iter, dev_iter, test_iter, model_name):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=0, last_epoch=-1)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        scheduler.step()  # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            if 'bert' not in model_name.lower():
                outputs = model(trains)
            else:
                inputs = tokenizer(trains, max_length=config.pad_size, padding=True, truncation=True, return_tensors='pt')
                input_ids = inputs['input_ids'].to(config.device)
                token_type_ids = inputs['token_type_ids'].to(config.device)
                attention_mask = inputs['attention_mask'].to(config.device)
                label = labels.to(config.device)
                outputs = model(input_ids, attention_mask, token_type_ids, label)
            model.zero_grad()
            if cla_task_name == 'mutil_label':
                loss = nn.BCEWithLogitsLoss()(outputs, labels.float())
            else:
                loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                if cla_task_name == 'mutil_label':
                    train_acc = mutil_label_acc_score(true, outputs.data)
                    dev_acc, dev_loss = evaluate_mutil_label_task(model, dev_iter, model_name)
                else:
                    predic = torch.max(outputs.data, 1)[1].cpu()
                    train_acc = acc_score(true, predic)
                    dev_acc, dev_loss = evaluate(config, model, dev_iter, model_name)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    test(config, model, test_iter, model_name)


def test(config, model, test_iter, model_name):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    if cla_task_name == 'mutil_label':
        test_acc, test_loss, f1_micro, f1_macro = evaluate_mutil_label_task(model, test_iter, model_name, test=True)
        msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
        print(msg.format(test_loss, test_acc))
        print("Mutil label f1_macro_score...")
        print(f1_macro)
        print("Mutil label f1_micro_score...")
        print(f1_micro)
    else:
        test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
        msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
        print(msg.format(test_loss, test_acc))
        print("Precision, Recall and F1-Score...")
        print(test_report)
        print("Confusion Matrix...")
        print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, model_name, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            if 'bert' in model_name.lower():
                outputs = model(texts)
            else:
                inputs = tokenizer(texts, max_length=config.pad_size, padding=True, truncation=True, return_tensors='pt')
                input_ids = inputs['input_ids'].to(config.device)
                token_type_ids = inputs['token_type_ids'].to(config.device)
                attention_mask = inputs['attention_mask'].to(config.device)
                label = labels.to(config.device)
                outputs = model(input_ids, attention_mask, token_type_ids, label)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = acc_score(labels_all, predict_all)
    if test:
        report = classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


def evaluate_mutil_label_task(model, data_iter, model_name, test=False):
    model.eval()
    loss_total = 0
    predict_all = []
    labels_all = []
    with torch.no_grad():
        for texts, labels in data_iter:
            if 'bert' not in model_name.lower():
                outputs = model(texts)
            else:
                inputs = tokenizer(texts, max_length=config.pad_size, padding=True, truncation=True, return_tensors='pt')
                input_ids = inputs['input_ids'].to(config.device)
                token_type_ids = inputs['token_type_ids'].to(config.device)
                attention_mask = inputs['attention_mask'].to(config.device)
                label = labels.to(config.device)
                outputs = model(input_ids, attention_mask, token_type_ids, label)
            loss = nn.BCEWithLogitsLoss()(outputs, labels.float())
            loss_total += loss
            labels = labels.data.cpu().numpy()
            labels_all.extend(labels)
            predict_all.extend(outputs.data)
    predict_all = torch.stack(predict_all)
    acc = mutil_label_acc_score(labels_all, predict_all)
    if test:
        f1_micro, f1_macro = mutil_label_f1_score(labels_all, predict_all)
        return acc, loss_total / len(data_iter), f1_micro, f1_macro
    return acc, loss_total / len(data_iter)


if __name__ == '__main__':
    set_seed(42)

    dataset = 'data'  # 存放数据集的目录
    cla_task_name = 'mutil_label'  # binary_cla(二分类),mutil_class(多分类),mutil_label(多标签分类)

    model_name = 'bert'  # 模型选择
    embedding = 'random'
    model = import_module('model_zoo.' + model_name)
    config = import_module('config.' + model_name + '_config').Config(dataset, embedding, model_name)

    model = model.Model(config).to(config.device)
    if 'bert' not in model_name.lower():
        init_network(model)
        print("Loading data...")
        vocab, train_data, dev_data, test_data = build_dataset(config, use_word=True, cla_task_name=cla_task_name)
        train_iter = build_iterator(train_data, config)
        dev_iter = build_iterator(dev_data, config)
        test_iter = build_iterator(test_data, config)
        config.n_vocab = len(vocab)
        train(config, model, train_iter, dev_iter, test_iter, model_name)
    else:
        print("Loading data for bert...")
        tokenizer = BertTokenizer.from_pretrained(config.bert_path)  # 设置预训练模型的路径
        train_data = BertData(config, config.train_path, cla_task_name=cla_task_name)
        valid_data = BertData(config, config.valid_path, cla_task_name=cla_task_name)
        test_data = BertData(config, config.test_path, cla_task_name=cla_task_name)
        train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_data, batch_size=config.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=config.batch_size, shuffle=True)
        train(config, model, train_dataloader, valid_dataloader, test_dataloader, model_name)
